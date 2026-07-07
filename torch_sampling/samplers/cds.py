import torch
import math
from typing import NamedTuple, Callable

from ..utils.kernel_state import KernelState

from ..targets.target_distribution import TargetDistribution
from ..targets.conditional_target import ConditionalTarget
from ..targets.gaussian import Gaussian

from ..kernels import MALAKernel, HMCKernel, MHKernel

from .base import Sampler
from .pt import PT

from tqdm import tqdm

MIN_TIME = 1e-10
TOL = 1e5

class PISDEState(NamedTuple):
    inner_state: KernelState
    z: torch.Tensor
    time: torch.Tensor

def _predictor_corrector_step(
    state: PISDEState,
    base_target: TargetDistribution,
    time_next: torch.Tensor,
    noise_var: torch.Tensor,
    corrector_kernel: Callable = None,
    corrector_step_size: torch.Tensor = None, 
    n_corrector_steps: int = 1,
) -> torch.Tensor:
    """
    Perform a single backward integration step.

    Arguments:
        state: current state containing x, z, log_prob_x, grad_log_prob_x, time
        time_next: next time step
        noise_var: noise variance at the current time step

    Returns:
        new_state: updated state after the step
        new_corrector_step_size: updated corrector step size
    """

    x, log_prob_x, grad_log_prob_x = state.inner_state
    z = state.z
    time = state.time
    
    time_clamp = torch.clamp(time, min=MIN_TIME, max=1.0)
    dtime = torch.clamp(time_next - time, min=0.0)

    # Euler-Maruyama update
    noise = torch.randn_like(x, device=x.device, dtype=x.dtype)
    drift_coeff = dtime / time_clamp
    drift = drift_coeff * ((x - z) + 0.5 * noise_var * grad_log_prob_x)
    diff_coeff = torch.sqrt(noise_var * dtime)
    diff = diff_coeff * noise
    new_x = x + drift + diff

    if n_corrector_steps > 0:
        new_x, new_log_prob_x, new_grad_log_prob_x, new_step_size = corrector_kernel(
            new_x,
            z,
            corrector_step_size,
            time_next
        )
        corrector_step_size = new_step_size.detach()
    else:
        time_next_clamp = torch.clamp(time_next, min=MIN_TIME, max=1.0)
        inv_time_next = 1.0 / time_next_clamp
        x0_next = z + inv_time_next * (new_x - z)  # (batch_size, dim)
        new_grad_log_prob_x, new_log_prob_x = base_target.grad_log_prob(x0_next, return_log_prob=True)  # (batch_size, dim)
        new_grad_log_prob_x = torch.clamp(new_grad_log_prob_x, min=-TOL, max=TOL)
    
    new_inner = KernelState(
        x=new_x,
        log_prob=new_log_prob_x,
        grad=new_grad_log_prob_x
    )

    new_state = PISDEState(
        inner_state=new_inner,
        z=z,
        time=time_next
    )

    return new_state, corrector_step_size

def mala_corrector_step(
    base_target: TargetDistribution,
    x: torch.Tensor,
    z: torch.Tensor,
    step_size: torch.Tensor,
    time: torch.Tensor,
    target_acceptance: float,
    adaptation_rate: float,
    log_prob_x: torch.Tensor = None,
    grad_log_prob_x: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performs a single Metropolis-Adjusted Langevin Algorithm (MALA) step with adaptive step size scaling.
    
    Arguments:
        base_target: the target distribution to sample from
        x: (..., dim) tensor of current samples
        z: (..., dim) tensor of reference samples
        step_size: current step size scaling factor
        time: current time
        target_acceptance: target acceptance rate for adaptation
        adaptation_rate: adaptation rate for step size adaptation
        log_prob_x: (..., 1) tensor of current log probabilities at x (optional)
        grad_log_prob_x: (..., dim) tensor of current gradients of log probability at x (optional)
    
    Returns:
        new_x: (..., dim) tensor of updated samples after MALA step
        new_log_prob: (..., 1) tensor of updated log probabilities at new_x
        new_grad_log_prob: (..., dim) tensor of updated gradients of log probability at new_x
        new_step_size: updated step size scaling factor after adaptation
    """



    time_clamp = torch.clamp(time, min=MIN_TIME, max=1.0)
    inv_time = 1.0 / time_clamp

    if log_prob_x is None or grad_log_prob_x is None:
        x0 = z + inv_time * (x - z) # (..., dim)
        grad_log_prob_x, log_prob_x = base_target.grad_log_prob(x0, return_log_prob=True) # (..., dim)
        grad_log_prob_x = torch.clamp(grad_log_prob_x, min=-TOL, max=TOL)

    is_valid_logprob = torch.isfinite(log_prob_x)  # (..., 1)
    is_valid_grad = torch.isfinite(grad_log_prob_x).all(dim=-1).unsqueeze(-1)  # (..., 1)
    is_valid_step = is_valid_logprob & is_valid_grad  # (..., 1)
    
    noise = torch.randn_like(x, device=x.device, dtype=x.dtype)  # (..., dim)

    drift_x = step_size * grad_log_prob_x  # (..., dim)
    diff_scale = torch.sqrt(2 * step_size * time_clamp)  # (1,)
    y = x + drift_x + diff_scale * noise  # (..., dim)

    y0 = z + inv_time * (y - z)  # (batch_size, dim)
    grad_log_prob_y, log_prob_y = base_target.grad_log_prob(y0, return_log_prob=True)  # (..., dim)
    grad_log_prob_y = torch.clamp(grad_log_prob_y, min=-TOL, max=TOL)

    drift_y = step_size * grad_log_prob_y  # (..., dim)

    norm_fwd = ((y - (x + drift_x)) ** 2).sum(dim=-1, keepdim=True)  # (..., 1)
    norm_bwd = ((x - (y + drift_y)) ** 2).sum(dim=-1, keepdim=True)  # (..., 1)

    log_q_dif = (1.0 / (4.0 * step_size * time_clamp + MIN_TIME)) * (norm_fwd - norm_bwd) # (..., 1)

    log_acceptance = log_prob_y - log_prob_x + log_q_dif  # (..., 1)
    log_acceptance = torch.where(is_valid_step, log_acceptance, torch.tensor(-float('inf'), device=x.device))

    # Accept/Reject
    rand_val = torch.rand_like(log_acceptance).log()  # (..., 1)
    mask = rand_val < log_acceptance  # (..., 1)

    x = torch.where(mask, y, x)  # (..., dim)
    log_prob_x = torch.where(mask, log_prob_y, log_prob_x)  # (..., 1)
    grad_log_prob_x = torch.where(mask, grad_log_prob_y, grad_log_prob_x)  # (..., dim)

    # Update step size
    if adaptation_rate > 0.0:
        log_acceptance = torch.nan_to_num(log_acceptance, nan=-float('inf')) # (..., 1)
        acceptance = torch.exp(log_acceptance) # (..., 1)
        acceptance = torch.clamp(acceptance, min=0.0, max=1.0)
        avg_acc = acceptance.mean()  # (1,)
        diff = avg_acc - target_acceptance  # (1,)
        step_size = step_size * torch.exp(adaptation_rate * diff)  # (1,)
    
    return x.detach(), log_prob_x.detach(), grad_log_prob_x.detach(), step_size.detach()

def mala_corrector(
    base_target: TargetDistribution,
    x: torch.Tensor,
    z: torch.Tensor,
    step_size: torch.Tensor,
    time: torch.Tensor,
    target_acceptance: float,
    adaptation_rate: float,
    n_corrector_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performs Metropolis-Adjusted Langevin Algorithm (MALA) steps with adaptive step size scaling.
    
    Arguments:
        base_target: the target distribution to sample from
        x: (batch_size, dim) tensor of current samples
        z: (batch_size, dim) tensor of reference samples
        step_size: current step size scaling factor
        time: current time
        target_acceptance: target acceptance rate for adaptation
        adaptation_rate: adaptation rate for step size adaptation
        n_corrector_steps: number of MALA corrector steps to perform
    
    Returns:
        new_x: (batch_size, dim) tensor of updated samples after MALA step
        new_log_prob: (batch_size, 1) tensor of updated log probabilities at new_x
        new_grad_log_prob: (batch_size, dim) tensor of updated gradients of log probability at new_x
        new_step_size: updated step size scaling factor after adaptation
    """ 

    time_clamp = torch.clamp(time, min=MIN_TIME, max=1.0)
    inv_time = 1.0 / time_clamp

    x0 = z + inv_time * (x - z) # (batch_size, dim)
    grad_log_prob_x, log_prob_x = base_target.grad_log_prob(x0, return_log_prob=True) # (batch_size, dim)
    grad_log_prob_x = torch.clamp(grad_log_prob_x, min=-TOL, max=TOL)

    for _ in range(n_corrector_steps):

        x, log_prob_x, grad_log_prob_x, step_size = mala_corrector_step(
            base_target,
            x,
            z,
            step_size,
            time,
            target_acceptance,
            adaptation_rate,
            log_prob_x,
            grad_log_prob_x
        )

    return x.detach(), log_prob_x.detach(), grad_log_prob_x.detach(), step_size.detach()


class CDS(Sampler):
    def __init__(
        self, 
        target : TargetDistribution,
        time_schedule: torch.Tensor,
        noise_schedule: torch.Tensor,
        corrector_mode: str = "mala",
        corrector_steps: int = 1,
        corrector_leapfrog_steps: int = 5,
        corrector_step_size: float = 0.1,
        corrector_adaptation_rate: float = 0.05,
        corrector_target_acceptance: float = None,
        jump_steps: int = 0, 
        jump_beta_schedule: torch.Tensor = None,
        jump_ref_std: float = 1.0,
        jump_step_mode: str = "mala",
        jump_step_size: float = None,
        jump_swap_mode: str = "nrpt",
        jump_swap_every: int = 1,
        jump_leapfrog_steps: int = 5,
        jump_adaptation_rate: float = 0.05,
        jump_target_acceptance: float = None,
        compile: bool = True,
        verbose : bool = False,
    ):
        """
        Progressive Interpolation by simulating an SDE.

        Arguments:
            target: target distribution to sample from
            time_schedule: tensor of time steps
            base_noise_var: base noise variance
            noise_schedule_type: type of noise schedule ("linear", "quadratic", "cosine")
            corrector_mode: type of corrector ("mala" or "hmc")
            corrector_mh: whether to use Metropolis-Hastings in MALA corrector
            corrector_steps: number of corrector steps to perform
            corrector_adaptation_rate: adaptation rate for corrector step size
            corrector_target_acceptance: target acceptance rate for corrector adaptation
            corrector_step_size: initial step size for corrector
            n_leapfrog_steps: number of leapfrog steps for HMC corrector
            compile: whether to compile the step function
            verbose: whether to print progress
        """
        super().__init__(target=target,verbose=verbose)
        
        # Time schedule
        self.time_schedule = time_schedule
        
        # Noise schedule
        self.noise_schedule = noise_schedule

        # if noise schedule and time_schedule have not the same length, raise error
        if len(self.noise_schedule) != len(self.time_schedule):
            raise ValueError("noise_schedule and time_schedule must have the same length.")
        
        # Corrector settings
        if corrector_mode not in ["mala"]:
            raise ValueError(f"Invalid corrector_mode: {corrector_mode}. Supported modes are: 'mala'.")
        self.corrector_mode = corrector_mode
        self.corrector_steps = corrector_steps
        self.corrector_leapfrog_steps = corrector_leapfrog_steps
        self.corrector_adaptation_rate = corrector_adaptation_rate
        self.corrector_target_acceptance = corrector_target_acceptance
        if self.corrector_target_acceptance is None:
            if self.corrector_mode in ["mala"]:
                self.corrector_target_acceptance = 0.574
        self.register_buffer("corrector_step_size", torch.tensor(corrector_step_size, requires_grad=False))
        
        if self.corrector_mode == "mala":
            self._corrector_kernel = lambda x, z, step_size, time: mala_corrector(
                self.target,
                x,
                z,
                step_size,
                time,
                target_acceptance=self.corrector_target_acceptance,
                adaptation_rate=self.corrector_adaptation_rate,
                n_corrector_steps=self.corrector_steps
            )
        
        # Jump settings
        if jump_steps > 0 and jump_beta_schedule is None:
            raise ValueError(f"jump_beta_schedule must be provided when jump_steps > 0.")

        self.jump_steps = jump_steps
        self.jump_ref_std = jump_ref_std
        self.jump_beta_schedule = jump_beta_schedule
        self.jump_step_mode = jump_step_mode
        self.jump_step_size = jump_step_size
        self.jump_swap_mode = jump_swap_mode
        self.jump_swap_every = jump_swap_every
        self.jump_leapfrog_steps = jump_leapfrog_steps
        self.jump_adaptation_rate = jump_adaptation_rate
        self.jump_target_acceptance = jump_target_acceptance

        self._compile = compile
        
        self._predictor_corrector_step = lambda state, time_next, noise_var, step_size: _predictor_corrector_step(
            state,
            self.target,
            time_next,
            noise_var,
            corrector_kernel=self._corrector_kernel,
            corrector_step_size=step_size,
            n_corrector_steps=self.corrector_steps,
        )    

        if compile:
            self._predictor_corrector_step = torch.compile(self._predictor_corrector_step)

        self._current_state : PISDEState = None

    def _init_state(
        self,
        x: torch.Tensor, 
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> None:
        
        # If time == 0, then log_prob is just target.log_prob(x)

        if time.item() == 0.0:
            div = 0.0 # We assume x == z at time 0
        else:
            time_clamp = torch.clamp(time, min=MIN_TIME, max=1.0)
            div = 1.0 / time_clamp
        x0 = z + div * (x - z)
        grad_log_prob_xt, log_prob_xt = self.target.grad_log_prob(x0, return_log_prob=True)
        inner_state = KernelState(
            x=x,
            log_prob=log_prob_xt,
            grad=grad_log_prob_xt
        )
        self._current_state = PISDEState(
            inner_state=inner_state,
            z=z,
            time=time
        )
    
    def _perform_jump(
        self,
        time: torch.Tensor,
        return_log_acceptance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform jump steps.

        Arguments:
            time: current time step
            return_log_acceptance: whether to return log acceptance rates

        Returns:
            new_x: (batch_size, dim)
            log_acceptance: (n_steps, n_replicas-1)   
        """

        if self.jump_step_size is None:
            jump_step_size = time.item() * self.corrector_step_size.item()
        else:
            jump_step_size = self.jump_step_size

        x = self._current_state.inner_state.x
        z = self._current_state.z

        n_replicas = len(self.jump_beta_schedule)
        z_replicas = z.unsqueeze(1).repeat(1, n_replicas, 1) # (n_samples, n_replicas, dim)
        z_replicas_flat = z_replicas.reshape(-1, self.target.dim) # (n_samples * n_replicas, dim)

        cond_target = ConditionalTarget(self.target, time, z_replicas_flat).to(x.device)
        mean = z.mean(dim=0)
        std = self.jump_ref_std * torch.ones(self.target.dim, device=x.device) / math.sqrt(time.item() + 1e-3)
        ref_target = Gaussian(
            mean=mean,
            std=std
        )

        if self.jump_step_mode == "mala":
            jump_kernel = MALAKernel(noise_scale=time.item())
        elif self.jump_step_mode == "hmc":
            jump_kernel = HMCKernel(n_leapfrog_steps=self.jump_leapfrog_steps, momentum_scale=time.item())
        else:
            jump_kernel = MHKernel()

        pt_sampler = PT(
            target=cond_target,
            reference=ref_target, 
            kernel=jump_kernel,
            beta_schedule=self.jump_beta_schedule,
            step_size=jump_step_size,
            swap_mode=self.jump_swap_mode,
            swap_every=self.jump_swap_every,
            adaptation_rate=self.jump_adaptation_rate,
            target_acceptance=self.jump_target_acceptance,
            verbose=self.verbose,
            compile=self._compile
        ).to(x.device)

        x0 = z.clone()
        out = pt_sampler(x0, n_steps=self.jump_steps, return_log_acceptance=return_log_acceptance)
        
        if return_log_acceptance:
            samples, log_acceptance = out
        else:
            samples = out
            log_acceptance = None

        new_x = samples[:, -1, :]  # Take samples from the last replica

        self._init_state(
            x=new_x,
            z=z,
            time=time
        )

        return new_x, log_acceptance
    
    def step(
        self,
        idx: int,
    ) -> torch.Tensor:
        """
        Step function.
        
        Arguments:
            idx: index of the current time step in the schedule
        
        Returns:
            x_new: (batch_size, dim)
        """

        time_next = self.time_schedule[idx + 1] if idx + 1 < len(self.time_schedule) else torch.tensor(1.0, device=self._current_state.time.device)
        noise_var = self.noise_schedule[idx]

        new_state, new_step_size = self._predictor_corrector_step(
            state=self._current_state,
            time_next=time_next,
            noise_var=noise_var,
            step_size=self.corrector_step_size
        )
        
        self._current_state = new_state
        self.corrector_step_size = new_step_size.detach()

        return self._current_state.inner_state.x

    def forward(
        self, 
        x0 : torch.Tensor,
        z : torch.Tensor = None,
        return_trajectory : bool = False,
        return_jump_log_acceptance : bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the kernel.
        
        Arguments:
            x0: starting point (batch_size, dim)
            z: Samples from the reference distribution, shape (batch_size, dim). If None, it will be set to x0.
            return_trajectory: if True, returns the trajectory of points, otherwise returns only the last point.
        
        Returns:
            xs: (n_steps + 1, batch_size, dim) if return_trajectory is True,
            x: (batch_size, dim) if return_trajectory is False
            jump_log_acceptance: (n_jump_steps, n_replicas-1) if return_jump_log_acceptance is True, else None
        """

        n_steps = len(self.time_schedule)

        if z is None:
            z = x0.clone()
        x = x0.clone()

        if return_trajectory:
            xs = [x.clone().to("cpu")]

        self._init_state(
            x=x,
            z=z,
            time=self.time_schedule[0]
        )

        if self.jump_steps > 0:
            x, jump_log_acceptance = self._perform_jump(self.time_schedule[0], return_log_acceptance=return_jump_log_acceptance)
            if return_trajectory:
                xs.append(x.clone().to("cpu"))

        if self.verbose:
            pbar = tqdm(total = n_steps, desc="Progressive Interpolation SDE")

        for i in range(0, n_steps):
            if self.verbose:
                pbar.update(1)
            # Sample from p_t(x | z)
            x = self.step(i)
            if return_trajectory:
                xs.append(x.clone().to("cpu"))
    
        if self.verbose:
            pbar.close()
        
        res = []
        if return_trajectory:
            res.append(torch.stack(xs, dim=0))  # (n_steps + 1, batch_size, dim)
        else:
            res.append(x)
        if return_jump_log_acceptance:
            if self.jump_steps > 0:
                res.append(jump_log_acceptance)
            else:
                res.append(None)
        return res[0] if len(res) == 1 else tuple(res)


    def sample_trajectory(
        self,
        n_samples : int = 1,
        z : torch.Tensor = None,
        device : torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Sample a trajectory of points starting from x0.

        Arguments:
            n_samples: number of samples to generate
            z: Samples from the reference distribution, shape (n_samples, dim). If None, it will be set to x0.
            device: device to place the initial tensor on
        
        Returns:
            xs: (n_steps + 1, n_samples, dim)
        """
        x0 = self.build_initial_point(
            n_samples=n_samples,
            device=device,
            dtype=torch.float32
        )
        return self.forward(x0=x0, z=z, return_trajectory=True)

    def sample(
        self, 
        n_samples : int = 1,
        z : torch.Tensor = None,
        device : torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Sample a single point starting from x0.
        
        Arguments:
            n_samples: number of samples to generate
            z: Samples from the reference distribution, shape (n_samples, dim). If None, it will be set to x0.
            device: device to place the initial tensor on
        
        Returns:
            x: (n_samples, dim)
        """
        x0 = self.build_initial_point(
            n_samples=n_samples,
            device=device,
            dtype=torch.float32
        )
        return self.forward(x0=x0, z=z, return_trajectory=False)