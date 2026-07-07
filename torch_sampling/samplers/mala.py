import torch
from typing import Optional

from ..targets.target_distribution import TargetDistribution
from ..utils.kernel_state import KernelState
from .base import Sampler
from ..kernels.mala_kernel import MALAKernel

class MALA(Sampler):
    def __init__(
        self, 
        target: TargetDistribution,
        step_size: float = 0.1,
        noise_scale: float = 1.0,
        target_acceptance: float = 0.574,
        adaptation_rate: float = 0.05,
        verbose: bool = False,
        compile: bool = False
    ) -> None:
        super().__init__(target=target, verbose=verbose)
        self.register_buffer("step_size", torch.tensor(step_size))
        self.target_acceptance = target_acceptance
        self.adaptation_rate = adaptation_rate
        
        # State caching
        self._current_state: Optional[KernelState] = None

        self._kernel = MALAKernel(noise_scale=noise_scale)
        self._kernel_fn = lambda state, step_size: self._kernel.step(
            self.target, state, step_size
        )
        if compile:
            self._kernel_fn = torch.compile(self._kernel_fn)

    def _init_state(self, x: torch.Tensor):
        """Initialize state with gradients if not exists."""
        grad, log_prob = self.target.grad_log_prob(x, return_log_prob=True)
        self._current_state = KernelState(x, log_prob, grad)

    def step(self) -> torch.Tensor:
        # Run Step
        new_state, log_accept = self._kernel_fn(
            self._current_state, 
            self.step_size
        )
        
        self._current_state = new_state

        # --- Adaptation Logic ---
        if self.adaptation_rate > 0.0:
            self._adapt_step_size(log_accept)
                
        return self._current_state.x

    def _adapt_step_size(self, log_accept: torch.Tensor):
        """
        Adaptation performed entirely using Tensors on-device.
        """
        # Calculate mean acceptance
        batch_mean_accept = torch.exp(torch.clamp(log_accept, max=0.0)).mean()
        
        # Calculate factor
        diff = batch_mean_accept - self.target_acceptance
        factor = torch.exp(self.adaptation_rate * diff)
        
        # 3. In-place update of the buffer
        self.step_size.mul_(factor)
        