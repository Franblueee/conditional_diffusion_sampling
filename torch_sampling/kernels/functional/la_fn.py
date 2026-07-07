import torch
from typing import Tuple

from ...targets.target_distribution import TargetDistribution
from ...utils.kernel_state import KernelState

def la_step(
    target: TargetDistribution,
    state: KernelState,
    step_size: torch.Tensor,
    noise_scale: float = 1.0
) -> Tuple[KernelState, torch.Tensor]:
    x, log_prob_x, grad_log_prob_x = state.x, state.log_prob, state.grad # (batch_size, ..., dim), (batch_size, ..., 1), (batch_size, ..., dim)
    
    noise = torch.randn_like(x) # (batch_size, ..., dim)
    x_new = x + step_size * grad_log_prob_x + torch.sqrt(2.0 * step_size * noise_scale) * noise # (batch_size, ..., dim)
    
    # 2. Target details
    grad_log_prob_x_new, log_prob_x_new = target.grad_log_prob(x_new, return_log_prob=True) # (batch_size, ..., dim), (batch_size, ..., 1)

    new_state = KernelState(
        x=x_new.detach(), 
        log_prob=log_prob_x_new.detach(), 
        grad=grad_log_prob_x_new.detach()
    )
    
    return new_state, None