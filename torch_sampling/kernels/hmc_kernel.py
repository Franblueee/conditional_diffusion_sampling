import torch

from ..utils.kernel_state import KernelState
from typing import Tuple
from .base import Kernel
from .functional import hmc_step

class HMCKernel(Kernel):
    def __init__(
        self, 
        n_leapfrog_steps: int = 5,
        momentum_scale: float = 1.0, 
        compile: bool = False
    ) -> None:
        self._step_fn = lambda target, state, step_size: hmc_step(
            target, state, step_size, n_leapfrog_steps=n_leapfrog_steps, momentum_scale=momentum_scale
        )
        self._step_fn = torch.compile(self._step_fn) if compile else self._step_fn

    def init_state(self, target, x: torch.Tensor) -> KernelState:
        grad, log_prob = target.grad_log_prob(x, return_log_prob=True)
        return KernelState(x, log_prob, grad)

    def step(self, target, state: KernelState, step_size: torch.Tensor) -> Tuple[KernelState, torch.Tensor]:
        return self._step_fn(target, state, step_size)