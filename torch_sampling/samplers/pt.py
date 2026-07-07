import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from tqdm import tqdm
from scipy.interpolate import interp1d

from ..targets.target_distribution import TargetDistribution
from ..targets.annealed_target_pt import AnnealedTargetPT
from ..kernels import Kernel, MALAKernel, MHKernel, HMCKernel
from .base import Sampler
from ..utils.kernel_state import KernelState

def swap_replicas(
    ann_target: AnnealedTargetPT,
    state: KernelState,
    swap_mode: str = "nrpt",
    iter_id: int = 0,
    indices: Optional[torch.Tensor] = None
) -> Tuple[KernelState, torch.Tensor]:
    """
    Swap replicas.
    
    Arguments:
        ann_target: AnnealedTargetPT instance.
        state: Current KernelState with samples, log_probs, and grads.
        swap_mode: "pt" for Parallel Tempering (all adjacent pairs), "nrpt" for Non-Reversible PT (alternating pairs).
        iter_id: Current iteration index, used for NRPT to alternate pairs.
        indices: Optional tensor of shape (n_samples, n_replicas) indicating specific replica indices for each sample.

    Returns:
        state: updated state after swaps.
        log_acceptance: Tensor of shape (n_replicas-1,) with swap log acceptance probabilities.
    """

    x, log_prob, grad_log_prob = state.x, state.log_prob, state.grad
    n_replicas = x.shape[1]

    # Recover raw log prob, recall that:
    # log_prob = beta * log_p + (1 - beta) * log_p_ref = beta * (log_p - log_p_ref) + log_p_ref
    # Thus, log_ratio = log_p - log_p_ref = (log_prob - log_p_ref) / beta
    betas = ann_target.beta_schedule.to(x.device).view(1, -1) # (1, n_replicas)
    grad_log_prob_ref, log_prob_ref = ann_target.grad_log_prob_reference(x, return_log_prob=True) # (n_samples, n_replicas, 1), (n_samples, n_replicas, dim)
    # log_ratio = ( state.log_prob.squeeze(-1) - log_prob_ref ) / betas # (n_samples, n_replicas)
    # grad_log_ratio = ( state.grad - grad_log_prob_ref ) / betas.view(1, -1, 1) # (n_samples, n_replicas, dim)
    grad_log_ratio, log_ratio = ann_target.grad_log_ratio(x, return_log_ratio=True)

    # Identify Partners
    all_partners = [(i, i+1) for i in range(n_replicas - 1)]
    if swap_mode == "pt":
        partners = [(i, i+1) for i in range(n_replicas - 1)]
    else:
        # NRPT alternates between even and odd pairs
        offset = iter_id % 2
        partners = [(i, i+1) for i in range(offset, n_replicas - 1, 2)]

    new_x = x.clone()
    new_log_prob = log_prob.clone()
    new_grad_log_prob = grad_log_prob.clone() if grad_log_prob is not None else None

    new_indices = indices.clone() if indices is not None else None

    with torch.no_grad():
        # Compute Swaps
        log_acceptance_list = []
        for i, j in all_partners:
            # log alpha = (beta_j - beta_i) * (log_ratio(x_i) - log_ratio(x_j))
            d_beta = betas[:, j] - betas[:, i] # (1,)
            d_log_prob = log_ratio[:, i] - log_ratio[:, j] # (n_samples,)
            log_acc = (d_beta * d_log_prob).squeeze() # (n_samples,)
            
            log_acceptance_list.append(log_acc.mean()) # Store average log acceptance for diagnostics

            if (i, j) in partners:
            
                # Determine which batch items swap
                rand_val = torch.rand_like(log_acc).log() # (n_samples,)
                mask = rand_val < log_acc # (n_samples,)
                mask = mask.view(-1, 1) # (n_samples, 1)
                
                # Swap x
                val_i = x[:, i].clone() # (n_samples, dim)
                val_j = x[:, j].clone() # (n_samples, dim)
                new_x[:, i, :] = torch.where(mask, val_j, val_i) # (n_samples, dim)
                new_x[:, j, :] = torch.where(mask, val_i, val_j) # (n_samples, dim)

                if new_indices is not None:
                    idx_i = indices[:, i].clone()
                    idx_j = indices[:, j].clone()
                    new_indices[:, i] = torch.where(mask.squeeze(), idx_j, idx_i)
                    new_indices[:, j] = torch.where(mask.squeeze(), idx_i, idx_j)
                
                # Swap log_probs
                lp_i = log_prob[:, i].clone() # (n_samples, 1)
                lp_j = log_prob[:, j].clone() # (n_samples, 1)
                log_ratio_i = log_ratio[:, i].clone() # (n_samples, 1)
                log_ratio_j = log_ratio[:, j].clone() # (n_samples, 1)
                log_prob_ref_i = log_prob_ref[:, i].clone() # (n_samples, 1)
                log_prob_ref_j = log_prob_ref[:, j].clone() # (n_samples, 1)
                
                new_log_prob[:, i] = torch.where(mask, log_ratio_j * betas[:, i] + log_prob_ref_j, lp_i)
                new_log_prob[:, j] = torch.where(mask, log_ratio_i * betas[:, j] + log_prob_ref_i, lp_j)

                # Swap grads if exist
                if grad_log_prob is not None:
                    g_i = grad_log_prob[:, i].clone() # (n_samples, dim)
                    g_j = grad_log_prob[:, j].clone() # (n_samples, dim)

                    grad_log_ratio_i = grad_log_ratio[:, i].clone() # (n_samples, dim)
                    grad_log_ratio_j = grad_log_ratio[:, j].clone() # (n_samples, dim)

                    grad_log_ref_i = grad_log_prob_ref[:, i].clone() # (n_samples, dim)
                    grad_log_ref_j = grad_log_prob_ref[:, j].clone() # (n_samples, dim)

                    new_grad_log_prob[:, i] = torch.where(mask, grad_log_ratio_j * betas[:, i] + grad_log_ref_j, g_i)
                    new_grad_log_prob[:, j] = torch.where(mask, grad_log_ratio_i * betas[:, j] + grad_log_ref_i, g_j)

    log_acc = torch.stack(log_acceptance_list) # (n_replicas-1,)

    new_x = new_x.detach()
    new_log_prob = new_log_prob.detach()
    new_grad_log_prob = new_grad_log_prob.detach() if new_grad_log_prob is not None else None
    log_acc = log_acc.detach()

    return KernelState(new_x, new_log_prob, new_grad_log_prob), log_acc, new_indices

class PT(Sampler):
    def __init__(
        self, 
        target: TargetDistribution,
        beta_schedule: torch.Tensor,
        kernel: Kernel,
        reference: TargetDistribution = None,
        step_size: float = 0.1, 
        swap_mode: str = "nrpt", # "pt" or "nrpt"
        swap_every: int = 1,
        adaptation_rate: float = 0.05,
        target_acceptance: float = None, # Set based on kernel if None
        verbose: bool = False,
        compile: bool = True
    ):
        super().__init__(target=target, verbose=verbose)
        self.reference = reference
        self.beta_schedule = beta_schedule
        self.kernel = kernel
        self.n_replicas = len(beta_schedule)
        self.swap_mode = swap_mode
        self.swap_every = swap_every
        
        # Adaptation params
        self.adaptation_rate = adaptation_rate

        self.register_buffer(
            "step_sizes", 
            torch.full(
                (1, self.n_replicas, 1), 
                step_size, 
                dtype=torch.float32,
                requires_grad=False
            )
        )

        self.ann_target = AnnealedTargetPT(
            target=self.target, 
            reference=self.reference,
            beta_schedule=self.beta_schedule
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

        self._compile = compile

        self._configure_kernel_fn()
        self._configure_swap_fn()
        
        if self._compile:
            self._kernel_fn = torch.compile(self._kernel_fn)
            self._swap_fn = torch.compile(self._swap_fn)

        self._current_state: Optional[KernelState] = None

        self._indices: Optional[torch.Tensor] = None        # (n_samples, n_replicas)
        self._particle_states: Optional[torch.Tensor] = None # (n_samples, n_replicas) 0=Neutral, 1=Ref, 2=Target
        self._trip_counts: Optional[torch.Tensor] = None    # (n_samples, n_replicas)

    def _replace_ann_target(self, new_ann_target: AnnealedTargetPT):
        self.ann_target = new_ann_target
        self._configure_kernel_fn()
        self._configure_swap_fn()
    
    def _configure_kernel_fn(self):
        self._kernel_fn = lambda state, step_sizes: self.kernel.step(self.ann_target, state, step_sizes)
        if self._compile:
            self._kernel_fn = torch.compile(self._kernel_fn)

    def _configure_swap_fn(self):
        self._swap_fn = lambda state, iter_id, indices: swap_replicas(
            self.ann_target, state, self.swap_mode, iter_id, indices
        )
        if self._compile:
            self._swap_fn = torch.compile(self._swap_fn)
    
    def _replace_beta_schedule(self, new_schedule: torch.Tensor):
        self.beta_schedule = new_schedule
        new_ann_target = AnnealedTargetPT(
            target=self.target,
            reference=self.reference,
            beta_schedule=new_schedule
        )
        self._replace_ann_target(new_ann_target)
 
    def to(self, device: torch.device) -> 'PT':
        self = super().to(device)
        new_ann_target = AnnealedTargetPT(
            target=self.target,
            reference=self.reference,
            beta_schedule=self.beta_schedule
        ).to(device)
        self._replace_ann_target(new_ann_target)
        return self

    @property
    def dim(self):
        return self.target.dim

    def _init_state(self, x: torch.Tensor) -> None:
        """
        Initialize state for all replicas.
        
        Arguments:
            x: Tensor of shape (n_samples, n_replicas, dim) with initial samples for each replica.
        """

        grad, log_prob = self.ann_target.grad_log_prob(x, return_log_prob=True)
        self._current_state = KernelState(x, log_prob, grad)

        # Initialize Tracking
        n_samples, n_replicas = x.shape[0], x.shape[1]
        device = x.device
        
        # permutation[b, k] = particle ID at ladder k
        self._indices = torch.arange(n_replicas, device=device).expand(n_samples, -1).clone()
        
        # State: 0 = Unknown/Mid, 1 = Touched Index 0 (Ref), 2 = Touched Index N-1 (Target)
        self._particle_states = torch.zeros((n_samples, n_replicas), dtype=torch.int8, device=device)
        
        # Set initial states based on position
        # Particles starting at index 0 have touched ref
        self._particle_states[:, 0] = 1 
        # Particles starting at index N-1 have touched target
        self._particle_states[:, -1] = 2 
        
        self._trip_counts = torch.zeros((n_samples, n_replicas), dtype=torch.int32, device=device)

    def _adapt_step_sizes(self, kernel_log_accept: torch.Tensor) -> None:
        """
        Adapt step sizes for each replica based on acceptance rates.

        Arguments:
            kernel_log_accept: Tensor of shape (n_samples, n_replicas) with log acceptance probabilities.    
        """

        kernel_log_accept = kernel_log_accept.detach()

        kernel_log_accept = torch.nan_to_num(kernel_log_accept, nan=-float('inf'))
        kernel_log_accept = torch.clamp(kernel_log_accept, max=0.0, min=-10.0)
        avg_accept = torch.exp(kernel_log_accept).mean(dim=0) # (n_replicas,)
        diff = avg_accept - self.target_acceptance # (n_replicas,)
        update_factor = torch.exp(self.adaptation_rate * diff).view(1, -1, 1) # (1, n_replicas, 1)
        self.step_sizes = self.step_sizes * update_factor # (1, n_replicas, 1)

    def step(
        self, 
        step_id: int = 0,
        use_tracking: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A single PT step: kernel updates, step size adaptation, and swaps.

        Arguments:
            step_id: Current step index.
        
        Returns:
            samples: Tensor of shape (n_samples, n_replicas, dim) with updated samples.
            log_acceptance: Tensor of shape (n_replicas-1,) with log acceptance probabilities during swaps.        
        """
            
        new_state, kernel_acc = self._kernel_fn(
            self._current_state, self.step_sizes
        )
            
        if self.adaptation_rate > 0.0:
            self._adapt_step_sizes(kernel_acc)
            
        log_acc = torch.full((len(self.beta_schedule)-1,), float('-inf'), device=self._current_state.x.device)
        if step_id % self.swap_every == 0 and len(self.beta_schedule) >= 2:
            new_state, log_acc, new_indices = self._swap_fn(
                new_state,
                step_id,
                self._indices
            )
            self._indices = new_indices

            if use_tracking:
                # We identify which particles are now at the ends of the ladder
                batch_idx = torch.arange(self._indices.shape[0], device=self._indices.device)
            
                # IDs of particles currently at Reference (Index 0)
                ids_at_ref = self._indices[:, 0] 
                # IDs of particles currently at Target (Index -1)
                ids_at_target = self._indices[:, -1]
                
                # Logic for Round Trip (0 -> N-1 -> 0)
                
                # Check particles now at Target (N-1)
                # If they previously touched Ref (State 1), mark as Touched Target (State 2)
                # (If we were counting 1-way trips, we would count here too, but standard is round trip)
                mask_ref_to_target = (self._particle_states[batch_idx, ids_at_target] == 1)
                self._particle_states[batch_idx, ids_at_target] = torch.where(
                    mask_ref_to_target, 
                    torch.tensor(2, device=self._indices.device, dtype=torch.int8),
                    self._particle_states[batch_idx, ids_at_target]
                )
                # Ensure anyone at target is at least state 2 (even if from unknown)
                self._particle_states[batch_idx, ids_at_target] = torch.max(
                    self._particle_states[batch_idx, ids_at_target], 
                    torch.tensor(2, device=self._indices.device, dtype=torch.int8)
                )

                # Check particles now at Reference (0)
                # If they previously touched Target (State 2), Increment Count and reset to State 1
                mask_target_to_ref = (self._particle_states[batch_idx, ids_at_ref] == 2)
                
                self._trip_counts[batch_idx, ids_at_ref] += mask_target_to_ref.int()
                
                # Reset state to 1 (Touched Ref) for anyone currently at Ref
                self._particle_states[batch_idx, ids_at_ref] = 1
        
        # self._current_state = new_state
        x = new_state.x.detach()
        log_prob = new_state.log_prob.detach()
        grad = new_state.grad.detach() if new_state.grad is not None else None

        self._current_state = KernelState(x, log_prob, grad)

        return self._current_state.x, log_acc

    def forward(
        self, 
        x0: torch.Tensor, 
        n_steps: int = 1, 
        return_trajectory: bool = False,
        return_log_acceptance: bool = False,
        return_round_trip_counts: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        
        Arguments:
            x0: Initial samples, shape (n_samples, dim) or (n_samples, n_replicas, dim)
            n_steps: Number of PT steps to perform.
            return_trajectory: If True, returns the full trajectory of samples.
            return_log_acceptance: If True, returns the log acceptance probabilities during swaps.
            return_round_trip_counts: If True, returns the round trip counts for each particle. 
        Returns:
            samples: Final samples after n_steps, shape (n_samples, n_replicas, dim) or (n_steps+1, n_samples, n_replicas, dim) if return_trajectory is True.
            log_acceptances (optional): Log acceptance probabilities during swaps, shape (n_replicas-1,) if return_log_acceptance is True.     
            round_trip_counts (optional): Round trip counts for each particle, shape (n_samples, n_replicas) if return_round_trip_counts is True.
        """
        
        if self.verbose:
            pbar = tqdm(total=n_steps, desc="PT")
        
        # Ensure x0 is (n_samples, n_replicas, dim)
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0).repeat(self.n_replicas, 1, 1) # (n_replicas, n_samples, dim)
            x0 = x0.permute(1, 0, 2).contiguous() # (n_samples, n_replicas, dim)

        self._init_state(x0)

        x = x0.clone()
        
        xs = [x.cpu()] if return_trajectory else []
        sum_log_acc = torch.zeros(len(self.beta_schedule)-1, device=x.device) if return_log_acceptance else None
        use_tracking = return_round_trip_counts

        for i in range(n_steps):
            x, log_acc = self.step(i, use_tracking) # x: (n_samples, n_replicas, dim), log_acc: (n_replicas-1,)
            
            if return_log_acceptance:
                sum_log_acc += log_acc
            if return_trajectory:
                xs.append(x.detach().cpu())
            
            if self.verbose: pbar.update(1)

        if self.verbose: pbar.close()

        res = []
        if return_trajectory:
            res.append(torch.stack(xs))
        else:
            res.append(x)
            
        if return_log_acceptance:
            res.append(sum_log_acc / n_steps) # Return average log acceptance over steps

        if return_round_trip_counts:
            res.append(self._trip_counts.cpu())

        return res[0] if len(res) == 1 else tuple(res)

    def run_rounds(
        self, 
        x0 : torch.Tensor = None,
        n_samples: int = None,
        n_rounds: int = None,
        max_n_steps: int = None,
        n_steps_per_round: int = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Optimize the beta schedule over multiple rounds.

        Arguments:
            x0: Initial samples, shape (n_samples, dim) or (n_samples, n_replicas, dim). If None, samples are drawn from standard normal.
            n_samples: Number of samples to use if x0 is None.
            n_rounds: Number of rounds to perform. If None, determined by max_n_steps.
            max_n_steps: Maximum number of steps per round. If None, determined by n_rounds.
            n_steps_per_round: Fixed number of steps per round. If None, doubles each round starting from 2.
            device: Device to perform computations on.
            verbose: If True, displays progress bars.

        Returns:
            schedules_list: List of beta schedules after each round.
            Lambda_list: List of Lambda tensors after each round.
            rejection_rates_list: List of rejection rates after each round.        

        """

        # x0 and n_samples cannot both be None
        if x0 is None and n_samples is None:
            raise ValueError("Either x0 or n_samples must be provided.")
        
        # n_rounds and max_n_steps cannot both be None
        if n_rounds is None and max_n_steps is None:
            raise ValueError("Either n_rounds or max_n_steps must be provided.")
        
        if x0 is not None:
            if x0.dim() == 2:
                x0 = x0.unsqueeze(0).repeat(self.n_replicas, 1, 1) # (n_replicas, n_samples, dim)
                x0 = x0.permute(1, 0, 2).contiguous() # (n_samples, n_replicas, dim)
            n_samples = x0.shape[0]
        else:
            x0 = torch.randn(n_samples, self.n_replicas, self.target.dim, device=device)

        if n_rounds is None:
            n_rounds = int(np.ceil(np.log2(max_n_steps))) - 1
        if max_n_steps is None:
            max_n_steps = 2**(n_rounds + 1)

        schedule_length = len(self.beta_schedule)
        schedules_list = [self.beta_schedule.cpu()]
        Lambda_list = []
        rejection_rates_list = []
        
        current_n_steps = n_steps_per_round if n_steps_per_round else 2
        
        for i in range(n_rounds):
            
            if verbose:
                print(f"Round {i+1}/{n_rounds}")

            _, log_acceptance = self.forward(
                x0=x0,
                n_steps=current_n_steps,
                return_trajectory=False,
                return_log_acceptance=True,
            )

            # Update Schedule
            current_n_steps = min(2 * current_n_steps, max_n_steps) if n_steps_per_round is None else n_steps_per_round
            
            new_schedule, Lambda, rej_rates = update_beta_schedule_pt(
                old_schedule=self.beta_schedule,
                log_acceptance=log_acceptance,
                new_schedule_length=schedule_length
            )
            
            # self.beta_schedule = new_schedule.to(device)
            self._replace_beta_schedule(new_schedule.to(device))
            
            schedules_list.append(new_schedule.cpu())
            Lambda_list.append(Lambda.cpu())
            rejection_rates_list.append(rej_rates.cpu())
                        
        return schedules_list, Lambda_list, rejection_rates_list

def update_beta_schedule_pt(
    old_schedule: torch.Tensor,
    log_acceptance: torch.Tensor,
    new_schedule_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update the beta schedule based on log acceptance probabilities using Gradient Scaling logic.

    Arguments:
        old_schedule: Tensor of shape (n_replicas,) with the current beta schedule.
        log_acceptance: Tensor of shape (n_replicas-1) with log acceptance probabilities during swaps.
        new_schedule_length: Desired length of the new beta schedule.

    Returns:
        new_schedule: Tensor of shape (new_schedule_length,)
        Lambda: Tensor of shape (n_replicas,) with cumulative rejection rates.
        rejection_rates: Tensor of shape (n_replicas-1,) with rejection rates between replicas.    
    """
    
    # Average over steps and batch
    # We want acceptance probability between replica i and i+1
    
    # Clamp and exp
    acc_probs = torch.exp(torch.clamp(log_acceptance, max=0.0)) # (n_replicas-1,)    
    rejection_rates = 1.0 - acc_probs # (n_replicas-1,)
    
    # Construct Lambda (Cumulative rejection)
    # Prepend 0 for the first replica
    rejection_rates_full = torch.cat([torch.tensor([0.0], device=rejection_rates.device), rejection_rates]) # (n_replicas,)
    Lambda = torch.cumsum(rejection_rates_full, dim=0) # (n_replicas,)

    # Interpolate
    Lambda_norm = Lambda / Lambda[-1] # (n_replicas,)
    x_old = old_schedule.detach().cpu().numpy() # (n_replicas,)
    y_old = Lambda_norm.detach().cpu().numpy() # (n_replicas,)
    
    # Ensure strict monotonicity for interpolation
    _, unique_idx = np.unique(y_old, return_index=True) 
    x_old = x_old[np.sort(unique_idx)]
    y_old = y_old[np.sort(unique_idx)]
    
    try:
        f_inv = interp1d(y_old, x_old, kind='cubic', fill_value="extrapolate") 
    except:
        f_inv = interp1d(y_old, x_old, kind='linear', fill_value="extrapolate")
        
    u = np.linspace(0, 1, new_schedule_length) # (new_schedule_length,)
    f_inv_u = f_inv(u) # (new_schedule_length,)
    f_inv_u = np.clip(f_inv_u, 0.0, 1.0) # Ensure within [0, 1]
    f_inv_u = np.sort(f_inv_u)
    new_schedule = torch.tensor(f_inv_u, device=old_schedule.device, dtype=old_schedule.dtype) # (new_schedule_length,)
    new_schedule = torch.clamp(new_schedule, 0.0, 1.0) # Ensure within [0, 1]
    
    return new_schedule, Lambda, rejection_rates