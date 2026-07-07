import torch
from typing import Optional

from ..targets.target_distribution import TargetDistribution
from ..utils.kernel_state import KernelState
from .base import Sampler
from ..kernels.mh_kernel import MHKernel


class MH(Sampler):
    """
    Adaptive Metropolis-Hastings (Random Walk Metropolis) sampler.
    """
    def __init__(
        self, 
        target: TargetDistribution,
        proposal_std: float = 1.0,
        target_acceptance: float = 0.234, # Optimal for high-dim RWM
        adaptation_rate: float = 0.05,
        compile: bool = False,
        verbose: bool = False
    ) -> None:
        super().__init__(target=target, verbose=verbose)
        self.register_buffer("proposal_std", torch.tensor(proposal_std))
        self.target_acceptance = target_acceptance
        self.adaptation_rate = adaptation_rate
        
        # State caching
        self._current_state: Optional[KernelState] = None

        self._kernel = MHKernel()
        self._kernel_fn = lambda state, proposal_std: self._kernel.step(
            self.target, state, proposal_std
        )
        if compile:
            self._kernel_fn = torch.compile(self._kernel_fn)

    def _init_state(self, x: torch.Tensor):
        """Initialize state with log_prob."""
        log_prob = self.target.log_prob(x)
        self._current_state = KernelState(x, log_prob)

    def step(self) -> torch.Tensor:

        # Run Step
        new_state, log_accept = self._kernel_fn(
            self._current_state, 
            self.proposal_std
        )
        
        self._current_state = new_state

        # --- Adaptation Logic ---
        if self.adaptation_rate > 0.0:
            self._adapt_step_size(log_accept)
                
        return self._current_state.x

    def _adapt_step_size(self, log_accept: torch.Tensor):
        """
        Adjust proposal_std based on acceptance rate.
        """
        # Calculate mean acceptance probability
        batch_mean_accept = torch.exp(torch.clamp(log_accept, max=0.0)).mean()
        
        # Update based on difference from target
        diff = batch_mean_accept - self.target_acceptance
        factor = torch.exp(self.adaptation_rate * diff) 
        self.proposal_std = self.proposal_std * factor

        # In-place update
        self.proposal_std.mul_(factor)
