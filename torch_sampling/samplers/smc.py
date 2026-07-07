import torch
import numpy as np
from typing import NamedTuple
from tqdm import tqdm
from scipy.interpolate import interp1d

from ..targets.target_distribution import TargetDistribution
from ..targets.annealed_target import AnnealedTarget
from ..kernels import Kernel, MALAKernel, MHKernel, HMCKernel
from .base import Sampler
from ..utils.kernel_state import KernelState

class SMCState(NamedTuple):
    inner_state: KernelState
    log_weights: torch.Tensor

class SMC(Sampler):
    """
    Sequential Monte Carlo (SMC) sampler.

    This algorithm samples from a sequence of distributions that gradually transition
    from a simple proposal distribution to the complex target distribution. It uses
    importance sampling, resampling, and MCMC propagation steps.
    """
    def __init__(
        self,
        target: TargetDistribution,
        beta_schedule: torch.Tensor,
        kernel: Kernel,
        n_particles: int = 100,
        n_leapfrog_steps: int = 1,
        step_size: float = 0.1,
        resampling_threshold: float = 0.9,
        resampling_method: str = "systematic",
        adaptation_rate: float = 0.05,
        target_acceptance: float = None,
        verbose: bool = False,
        compile: bool = True
    ) -> None:
        """
        Initialize the SMC sampler.

        Arguments:
            target: The target distribution to sample from.
            beta_schedule: A schedule of inverse temperatures (betas) for annealing.
            kernel: An MCMC kernel (e.g., LA, MALA, HMC) for the mutation step.
            resampling_threshold: The threshold for the effective sample size (ESS) below which resampling is triggered (as a fraction of the total number of particles).
            verbose: If True, display a progress bar.
        """
        super().__init__(target=target, verbose=verbose)
        self.target = target
        self.beta_schedule = beta_schedule
        self.kernel = kernel
        self.n_particles = n_particles
        self.n_leapfrog_steps = n_leapfrog_steps
        self.register_buffer("step_size", torch.tensor(step_size, requires_grad=False))
        self.resampling_threshold = resampling_threshold
        self.resampling_method = resampling_method
        self.adaptation_rate = adaptation_rate
        self.verbose = verbose

        if self.resampling_threshold > 0.0 and self.resampling_method not in ["multinomial", "systematic"]:
            raise ValueError("Invalid resampling method. Must be 'multinomial' or 'systematic'.")

        if target_acceptance is None:
            if isinstance(kernel, MALAKernel):
                self.target_acceptance = 0.574
            elif isinstance(kernel, MHKernel):
                self.target_acceptance = 0.234
            elif isinstance(kernel, HMCKernel):
                self.target_acceptance = 0.651
            else:
                print("Warning: Unknown kernel type for PT, using default target acceptance 0.0")
                self.target_acceptance = 0.0

        self._kernel_fn = lambda state, step_size, beta: self.kernel.step(
            AnnealedTarget(self.target, beta),
            state,
            step_size
        )

        if target_acceptance is None:
            if isinstance(kernel, MALAKernel):
                self.target_acceptance = 0.574
            elif isinstance(kernel, MHKernel):
                self.target_acceptance = 0.234
            elif isinstance(kernel, HMCKernel):
                self.target_acceptance = 0.651
            else:
                print("Warning: Unknown kernel type for PT, using default target acceptance 0.0")
                self.target_acceptance = 0.0

        if compile:
            self._kernel_fn = torch.compile(self._kernel_fn)

        self._current_state = None

    def _init_state(self, x: torch.Tensor, beta: torch.Tensor):
        """Initialize state for all replicas."""

        ann_target = AnnealedTarget(self.target, beta)
        grad, log_prob = ann_target.grad_log_prob(x, return_log_prob=True)
        inner = KernelState(x, log_prob, grad)
        
        log_weights = torch.full((x.shape[0],), np.log(1.0 / x.shape[0]), device=x.device, dtype=x.dtype)
            
        self._current_state = SMCState(inner, log_weights)
    
    def _adapt_step_size(self, kernel_log_accept: torch.Tensor):
        """
        Adapt the step size based on acceptance rates.
        """
        kernel_log_accept = torch.nan_to_num(kernel_log_accept, nan=-float('inf'))
        kernel_log_accept = torch.clamp(kernel_log_accept, max=0.0, min=-10.0)
        batch_mean_accept = torch.exp(kernel_log_accept).mean()
        diff = batch_mean_accept - self.target_acceptance
        factor = torch.exp(self.adaptation_rate * diff)
        self.step_size.mul_(factor)

    def _effective_sample_size(self, norm_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate the effective sample size (ESS).

        Arguments:
            norm_weights: (n_groups, n_samples,) tensor of normalized weights (sum to 1).

        Returns:
            ess: (n_groups,) tensor of effective sample sizes.
        """
        return 1.0 / torch.sum(norm_weights**2, dim=1)  # (n_groups,)

    def _resample_if_needed(
        self,
        x: torch.Tensor,
        log_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Perform resampling if the ESS is below the threshold.
        Resamples independent groups of particles in parallel.

        Arguments:
            x: (n_samples, dim) tensor of particles.
            log_weights: (n_samples,) tensor of log-weights.
        
        Returns:
            x_resampled: (n_samples, dim) tensor of resampled particles.
            log_weights_resampled: (n_samples,) tensor of resampled log-weights.
        """
        n_samples = x.shape[0]
        n_groups = n_samples // self.n_particles
        if n_groups < 1:
            n_groups = 1

        # Reshape to (n_groups, n_particles, dim) and (n_groups, n_particles)
        # We clone x here to avoid modifying the input tensor in-place later
        x_view = x.view(n_groups, self.n_particles, -1)
        log_weights_view = log_weights.view(n_groups, self.n_particles)

        # Normalize weights for ESS calculation and resampling
        weights = torch.softmax(log_weights_view, dim=1)  # (n_groups, n_particles)
        ess = self._effective_sample_size(weights)        # (n_groups,)

        # Identify which groups need resampling
        # Note: We use self.n_particles because threshold is relative to group size
        mask_resample = ess < (self.n_particles * self.resampling_threshold) # (n_groups,)

        # If no groups need resampling, return early
        if not mask_resample.any():
            return x, log_weights
       
        # We work on a clone to ensure we return a new tensor and don't mutate input x
        x_resampled = x_view.clone() # (n_groups, n_particles, dim)
        log_weights_resampled = log_weights_view.clone() # (n_groups, n_particles)

        # Extract only the groups that need resampling
        # Shape: (n_groups_to_resample, n_particles)
        weights_to_resample = weights[mask_resample] # (n_groups_to_resample, n_particles)

        if self.resampling_method == "multinomial":
            # Multinomial resampling: generate indices for the groups that need it
            # indices shape: 
            indices = torch.multinomial(weights_to_resample, self.n_particles, replacement=True) # (n_groups_to_resample, n_particles)
        elif self.resampling_method == "systematic":
            n_groups_to_resample = weights_to_resample.shape[0]
            
            offset = torch.rand(n_groups_to_resample, 1, device=weights_to_resample.device) # (n_groups_to_resample, 1)
            steps = torch.arange(self.n_particles, device=weights_to_resample.device).unsqueeze(0) # (1, n_particles)
            
            # u = (offset + 0, offset + 1, ...) / n_particles
            u = (offset + steps) / self.n_particles

            # Calculate CDF
            cumulative_sum = torch.cumsum(weights_to_resample, dim=1) # (n_groups_to_resample, n_particles)
            cumulative_sum[..., -1] = 1.0  # Numerical stability fix

            # searchsorted finds indices such that cumulative_sum[i] >= u
            indices = torch.searchsorted(cumulative_sum, u) # (n_groups_to_resample, n_particles)
            
            # Clamp indices to handle potential float precision edge cases
            indices = torch.clamp(indices, max=self.n_particles - 1)

        # Gather the particles using the indices
        # We need to expand indices to match x dimensions: (n_groups_subset, n_particles, dim)
        subset_x = x_resampled[mask_resample] # (n_groups_to_resample, n_particles, dim)
        
        # Helper to gather along the particle dimension (dim=1)
        # expand indices to: (n_groups_subset, n_particles, dim)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, subset_x.size(-1)) # (n_groups_to_resample, n_particles, dim)
        
        # Perform the gather
        x_resampled[mask_resample] = torch.gather(subset_x, 1, gather_indices) 

        # Reset weights for resampled groups to uniform (log(1) = 0.0)
        # Note: Depending on your convention, uniform log weights might be -log(N) or 0.
        # Following your snippet, we use 0.0.
        log_weights_resampled[mask_resample] = 0.0

        # Flatten back to original shapes
        x_resampled = x_resampled.view(n_samples, -1)
        log_weights_resampled = log_weights_resampled.view(n_samples)
        
        return x_resampled, log_weights_resampled
    
    def step(
        self,
        beta_prev: torch.Tensor,
        beta_curr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one full step of the SMC algorithm: reweight, resample, and mutate.

        Arguments:
            beta_prev: The previous beta value in the annealing schedule.
            beta_curr: The current beta value in the annealing schedule.

        Returns:
            x_new: (n_samples, dim) tensor of updated particles.
            log_weights_new: (n_samples,) tensor of updated log-weights.
            log_increments: (n_samples,) tensor of incremental log-weights.
        """

        # Propagate           
        x = self._current_state.inner_state.x
        new_inner, kernel_acc = self._kernel_fn(
            self._current_state.inner_state, self.step_size, beta_curr
        )
        # Reweight

        target_log_prob_x = self.target.log_prob(x) # (n_samples, 1)
        # target_log_prob_x = new_inner.log_prob / (beta_curr + 1e-6) # (n_samples, 1)

        log_increments = (beta_curr - beta_prev)*(target_log_prob_x).squeeze(-1) # (n_samples,)
        new_log_weights = self._current_state.log_weights + log_increments # (n_samples,)

        if self.adaptation_rate > 0.0:
            self._adapt_step_size(kernel_acc)        

        # Resample
        if self.resampling_threshold > 0.0:
            new_x, new_log_weights = self._resample_if_needed(
                new_inner.x,
                new_log_weights
            )
            new_inner = new_inner._replace(x=new_x)

        del self._current_state
        self._current_state = SMCState(new_inner, new_log_weights)

        return self._current_state.inner_state.x, self._current_state.log_weights, log_increments

    def forward(
        self,
        x0: torch.Tensor,
        return_trajectory: bool = False,
        return_log_weights: bool = False,
        return_log_increments: bool = False
    ) -> torch.Tensor:
        """
        Run the full SMC algorithm from an initial set of particles.

        Arguments:
            x0: (n_samples, dim) tensor of initial particles.
            return_trajectory: If True, returns the trajectory of particles at each annealing step.
            return_log_weights: If True, returns the log weights at each annealing step.
            return_log_increments: If True, returns the log weight increments at each annealing step.

        Returns:
            - If return_trajectory is True: (n_steps, n_samples, dim) tensor.
            - If return_trajectory is False: (n_samples, dim) tensor of final particles.
            - Additionally, if return_log_weights is True, returns (n_steps, n_samples) tensor of log weights at each step.
            - Additionally, if return_log_increments is True, returns (n_steps-1, n_samples) tensor of log weight increments at each step.
        """


        self._init_state(x0, self.beta_schedule[0])

        x = x0.clone()
        log_weights = self._current_state.log_weights.clone()

        if self.verbose:
            pbar = tqdm(total=len(self.beta_schedule) - 1, desc="SMC")

        if return_trajectory:
            xs = [x.clone().to("cpu")]

        if return_log_weights:
            log_weights_list = [log_weights.clone().to("cpu")]
        
        if return_log_increments:
            log_increments_list = []

        beta_prev = self.beta_schedule[0]
        for t in range(len(self.beta_schedule) - 1):
            beta_curr = self.beta_schedule[t+1]
            x, log_weights, log_increments = self.step(beta_prev, beta_curr)
            beta_prev = beta_curr

            if return_log_weights:
                log_weights_list.append(log_weights.clone().to("cpu"))
            
            if return_log_increments:
                log_increments_list.append(log_increments.clone().to("cpu"))

            if return_trajectory:
                xs.append(x.clone().to("cpu"))
            
            if self.verbose:
                pbar.update(1)

        if self.verbose:
            pbar.close()
        
        if return_log_weights:
            log_weights = torch.stack(log_weights_list, dim=0)

        if return_log_increments:
            log_increments = torch.stack(log_increments_list, dim=0)

        result = []
        if return_trajectory:
            xs = torch.stack(xs, dim=0)
            result.append(xs)
        else:
            result.append(x)

        if return_log_weights:
            result.append(log_weights)

        if return_log_increments:
            result.append(log_increments)

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)

    def sample(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate samples from the target distribution.

        Arguments:
            n_samples: The number of particles (samples) to generate.
            device: The device to place the tensors on.
            dtype: The data type of the tensors.

        Returns:
            (n_samples, dim) tensor of final samples.
        """
        # Initial particles are typically drawn from a simple distribution like N(0, I)
        x0 = torch.randn(n_samples, self.dim, device=device, dtype=dtype)
        return self.forward(x0, return_trajectory=False)

    def sample_trajectory(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate a trajectory of samples at each annealing step.

        Arguments:
            n_samples: The number of particles (samples) to generate.
            device: The device to place the tensors on.
            dtype: The data type of the tensors.

        Returns:
            (n_steps + 1, n_samples, dim) tensor of sample trajectories.
        """
        x0 = torch.randn(n_samples, self.dim, device=device, dtype=dtype)
        return self.forward(x0, return_trajectory=True)

    def run_rounds(
        self, 
        x0: torch.Tensor = None,
        n_samples: int = None,
        n_rounds: int = None,
        max_schedule_length: int = None,
        final_schedule_length: int = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Perform warmup for the SMC sampler by running multiple rounds of SMC to adaptively determine the beta schedule.

        Arguments:
            n_samples: The number of particles to use in each SMC run.
            max_schedule_length: The maximum allowed length of the beta schedule. If None, it will be set to 2^(n_rounds + 1).
            n_rounds: The number of rounds to run. If None, it will be determined based on max_schedule_length.
            final_schedule_length: The desired final length of the beta schedule. If None, it will be set to max_schedule_length.
            device: The device to place the tensors on.
            verbose: If True, display a progress bar.

        Returns:
            schedules_list: List of beta schedules from each round.
            Lambda_list: List of global barrier estimates from each round.
        """

        # x0 and n_samples cannot both be None
        if x0 is None and n_samples is None:
            raise ValueError("Either x0 or n_samples must be provided.")

        if n_rounds is None and max_schedule_length is None:
            raise ValueError("Either n_rounds or max_schedule_length must be provided.")

        if x0 is None:
            x0 = torch.randn(n_samples, self.target.dim, device=device)
        
        if n_rounds is None:
            n_rounds = int(np.ceil(np.log2(max_schedule_length))) - 1
        
        if max_schedule_length is None:
            max_schedule_length = 2**(n_rounds + 1)

        if final_schedule_length is None:
            final_schedule_length = max_schedule_length

        new_schedule_length = 2
        new_beta_schedule = torch.linspace(0.0, 1.0, new_schedule_length, device=device)

        Lambda_list = []
        schedules_list = [new_beta_schedule.cpu()]
        sqrt_D_list = []
        self.beta_schedule = new_beta_schedule.clone()
        
        if verbose:
            pbar = tqdm(total=n_rounds)

        for round_idx in range(n_rounds):

            # Run SMC to get log_g values
            _, log_weights, log_increments = self.forward(
                x0=torch.randn(n_samples, self.dim, device=device),
                return_trajectory=False,
                return_log_weights=True,
                return_log_increments=True
            )

            # Update beta schedule
            # if it's the last round
            if round_idx == n_rounds - 1:
                new_schedule_length = final_schedule_length
            else:
                new_schedule_length = min(2 * len(self.beta_schedule), max_schedule_length)
            new_beta_schedule, Lambda, sqrt_D = update_beta_schedule_smc(
                old_schedule=self.beta_schedule,
                log_weights=log_weights,
                log_increments=log_increments,
                new_schedule_length=new_schedule_length
            )
            self.beta_schedule = new_beta_schedule.clone()

            schedules_list.append(new_beta_schedule.cpu())
            Lambda_list.append(Lambda.cpu())
            sqrt_D_list.append(sqrt_D.cpu())

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()
        
        return schedules_list, Lambda_list, sqrt_D_list

def update_beta_schedule_smc(
    old_schedule: torch.Tensor,
    log_weights: torch.Tensor,
    log_increments: torch.Tensor,
    new_schedule_length: int,
) -> torch.Tensor:
    """
    Update the beta schedule based on the log_g values. 
    Follows Algorithm 3 from "Optimized Annealed Sequential Monte Carlo Samplers".

    Arguments:
        old_schedule: (n_old_steps,) tensor of old beta schedule.
        log_weights: (n_old_steps, n_samples) tensor of log weights at each step.
        log_increments: (n_old_steps-1, n_samples,) tensor of log weight increments at each step.
        new_schedule_length: the desired length of the new beta schedule.   

    Returns:
        new_schedule: (new_schedule_length,) tensor of new beta schedule.
        Lambda: (n_old_steps,) tensor of cumulative barrier estimates.
    """

    log_g_1 = compute_log_g(log_weights, log_increments, exponent=1) # (n_old_steps-1,)
    log_g_2 = compute_log_g(log_weights, log_increments, exponent=2) # (n_old_steps-1,)

    # print("log_g_1:", log_g_1.min().item(), log_g_1.max().item())
    # print("log_g_2:", log_g_2.min().item(), log_g_2.max().item())

    D = log_g_2 - 2*log_g_1 # (n_old_steps-1,)
    # print("D:", D.min().item(), D.max().item())
    D = torch.clamp(D, min=0.0) # (n_old_steps-1,)
    sqrt_D = torch.sqrt(D) # (n_old_steps-1,)
    sqrt_D = torch.cat([torch.tensor([0.0], device=sqrt_D.device, dtype=sqrt_D.dtype), sqrt_D], dim=0) # (n_old_steps,)

    # cumsum of D
    Lambda = torch.cumsum(sqrt_D, dim=0) # (n_old_steps,)

    # print("Lambda:", Lambda.min().item(), Lambda.max().item())

    Lambda_norm = Lambda / Lambda[-1] # (n_old_steps,)
    Lambda_norm_np = Lambda_norm.cpu().numpy()
    old_schedule_np = old_schedule.cpu().numpy()

    _, unique_idx = np.unique(Lambda_norm_np, return_index=True)
    # sort unique_idx
    unique_idx = np.sort(unique_idx)
    x = old_schedule_np[unique_idx]
    y = Lambda_norm_np[unique_idx]

    try: 
        spline_inv = interp1d(y, x, kind='cubic', fill_value="extrapolate")
    except Exception as e:
        print(f"Error in interp1d: {e}. Falling back to linear interpolation.")
        spline_inv = interp1d(y, x, kind='linear', fill_value="extrapolate")

    # new betas are equally spaced in Lambda space
    u = np.linspace(0, 1, new_schedule_length) # (new_schedule_length,)
    new_schedule = spline_inv(u) # (new_schedule_length,)
    # clip to [0, 1]
    new_schedule = np.clip(new_schedule, 0.0, 1.0)
    # sort from smallest to largest
    new_schedule = np.sort(new_schedule)
    new_schedule = torch.tensor(new_schedule, device=old_schedule.device, dtype=old_schedule.dtype)

    return new_schedule, Lambda, sqrt_D

def compute_log_g(
    log_weights: torch.Tensor,
    log_increments: torch.Tensor,
    exponent: int = 1
) -> torch.Tensor:
    """
    Compute the log_g_t values from the log weights and log increments.
    WARNING: The log_g_t values computed here are NOT the same as in the paper, they are
    normalized. So we are computing log_g_t - log_g_0.

    Arguments:
        log_weights: (n_steps, n_samples,) tensor of log weights at each step.
        log_increments: (n_steps-1, n_samples, ) tensor of log weight increments at each step.
        exponent: The exponent to use in the computation (1 or 2).

    Returns:
        log_g: (n_steps, ) tensor of log_g_t values.
    """

    if exponent not in [1, 2]:
        raise ValueError("Exponent must be 1 or 2.")
    
    log_sum = torch.logsumexp(log_weights[0:-1, :] + exponent*log_increments, dim=1) # (n_steps-1,)
    log_norm = torch.logsumexp(log_weights, dim=1) # (n_steps,)

    log_g = log_sum - log_norm[0:-1] # (n_steps-1,)
    
    return log_g