import torch
from typing import Optional

from ..targets.target_distribution import TargetDistribution
from ..utils.kernel_state import KernelState
from .base import Sampler
from ..kernels.hmc_kernel import HMCKernel

class HMC(Sampler):
    def __init__(
        self, 
        target: TargetDistribution,
        step_size: float = 0.1,
        n_leapfrog_steps: int = 5,
        target_acceptance: float = 0.651, # Optimal for HMC
        adaptation_rate: float = 0.05,
        momentum_scale: float = 1.0,
        compile: bool = False,
        verbose: bool = False
    ) -> None:
        super().__init__(target=target, verbose=verbose)
        self.register_buffer("step_size", torch.tensor(step_size))
        self.n_leapfrog_steps = n_leapfrog_steps
        self.target_acceptance = target_acceptance
        self.adaptation_rate = adaptation_rate
        
        self._current_state: Optional[Sampler] = None
        
        self._kernel = HMCKernel(n_leapfrog_steps=n_leapfrog_steps, momentum_scale=momentum_scale, compile=compile)
        self._kernel_fn = lambda state, step_size: self._kernel.step(
            self.target, state, step_size
        )

        if compile:
            self._kernel_fn = torch.compile(self._kernel_fn)
            

    def _init_state(self, x: torch.Tensor):
        grad, log_prob = self.target.grad_log_prob(x, return_log_prob=True)
        self._current_state = KernelState(x, log_prob, grad)

    def step(self) -> torch.Tensor:

        new_state, log_accept = self._kernel_fn(
            self._current_state, 
            self.step_size,
        )
        
        self._current_state = new_state

        if self.adaptation_rate > 0.0:
            self._adapt_step_size(log_accept)
            
        return self._current_state.x

    def _adapt_step_size(self, log_accept: torch.Tensor):
        batch_mean_accept = torch.exp(torch.clamp(log_accept, max=0.0)).mean()
        diff = batch_mean_accept - self.target_acceptance
        factor = torch.exp(self.adaptation_rate * diff)
        self.step_size.mul_(factor)