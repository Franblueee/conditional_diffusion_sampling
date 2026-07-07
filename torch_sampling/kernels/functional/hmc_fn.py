import torch
from typing import Tuple

from ...targets.target_distribution import TargetDistribution
from ...utils.kernel_state import KernelState

def hmc_step(
    target: TargetDistribution,
    state: KernelState,
    step_size: torch.Tensor,
    n_leapfrog_steps: int,
    momentum_scale: float = 1.0,
) -> Tuple[KernelState, torch.Tensor]:
    
    x_init, log_prob_init, grad_init = state.x, state.log_prob, state.grad # (batch_size, dim), (batch_size, 1), (batch_size, dim)
    
    # Sample initial momentum
    p_init = torch.randn_like(x_init) # (batch_size, dim)
    K_init = 0.5 * (p_init ** 2).sum(dim=-1, keepdim=True) # (batch_size, 1)
    H_init = -log_prob_init + K_init # (batch_size, 1)

    x = x_init.clone() # (batch_size, dim)
    p = p_init.clone() # (batch_size, dim)
    grad = grad_init.clone() # (batch_size, dim)

    # Half step for momentum
    p = p + 0.5 * step_size * grad # (batch_size, dim)

    # Leapfrog steps
    for i in range(n_leapfrog_steps):
        x = x + step_size * momentum_scale * p

        if i != n_leapfrog_steps - 1:
            grad, _ = target.grad_log_prob(x, return_log_prob=True)
            p = p + step_size * grad
    
    # Final half step for momentum
    grad, log_prob = target.grad_log_prob(x, return_log_prob=True)
    p = p + 0.5 * step_size * grad

    # Negate momentum for symmetry
    p = -p

    K_prop = 0.5 * (p ** 2).sum(dim=-1, keepdim=True) # (batch_size, 1)
    H_prop = -log_prob + K_prop # (batch_size, 1)

    log_acceptance = H_init - H_prop # (batch_size, 1)

    # filter nans in log acceptance
    is_valid_logprob = torch.isfinite(log_prob) # (batch_size, 1)
    is_valid_grad = torch.isfinite(grad).all(dim=-1).unsqueeze(-1) # (batch_size, 1)
    is_valid_step = is_valid_logprob & is_valid_grad
    log_acceptance = torch.where(is_valid_step, log_acceptance, torch.tensor(-float('inf'), device=x.device))

    rand_val = torch.rand_like(log_acceptance).log()
    accept = rand_val < log_acceptance
    
    final_x = torch.where(accept, x, x_init)
    final_log_prob = torch.where(accept, log_prob, log_prob_init)
    final_grad = torch.where(accept, grad, grad_init)

    new_state = KernelState(x=final_x.detach(), log_prob=final_log_prob.detach(), grad=final_grad.detach())
    
    return new_state, log_acceptance.detach()