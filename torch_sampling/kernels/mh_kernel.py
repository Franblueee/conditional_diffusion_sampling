import torch

from ..utils.kernel_state import KernelState
from typing import Tuple
from .base import Kernel
from .functional import mh_step

class MHKernel(Kernel):
    def __init__(
        self, 
        compile: bool = False
    ) -> None:
        self._step_fn = torch.compile(mh_step) if compile else mh_step

    def init_state(self, target, x: torch.Tensor) -> KernelState:
        log_prob = target.log_prob(x)
        return KernelState(x, log_prob)

    def step(self, target, state: KernelState, proposal_std: torch.Tensor) -> Tuple[KernelState, torch.Tensor]:
        return self._step_fn(target, state, proposal_std)