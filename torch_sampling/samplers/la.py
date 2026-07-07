import torch
from typing import Optional

from ..targets.target_distribution import TargetDistribution
from ..utils.kernel_state import KernelState
from .base import Sampler
from ..kernels.la_kernel import LAKernel

class LA(Sampler):
    def __init__(
        self, 
        target: TargetDistribution,
        step_size: float = 0.1,
        verbose: bool = False,
        compile: bool = False
    ) -> None:
        super().__init__(target=target, verbose=verbose)
        self.register_buffer("step_size", torch.tensor(step_size))

        # State caching
        self._current_state: Optional[KernelState] = None

        self._kernel = LAKernel()
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
        new_state, _ = self._kernel_fn(
            self._current_state, 
            self.step_size
        )
        
        self._current_state = new_state
                
        return self._current_state.x