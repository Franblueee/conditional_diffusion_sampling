import torch
from typing import Tuple

from ...targets.target_distribution import TargetDistribution
from ...utils.kernel_state import KernelState

def mala_step(
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

    with torch.no_grad():
        
        is_valid_logprob = torch.isfinite(log_prob_x_new) # (batch_size, ..., 1)
        is_valid_grad = torch.isfinite(grad_log_prob_x_new).all(dim=-1).unsqueeze(-1) # (batch_size, ..., 1)

        is_valid_step = is_valid_logprob & is_valid_grad # (batch_size, ..., 1) 
        
        diff_bwd = x - x_new - step_size * grad_log_prob_x_new # (batch_size, ..., dim)
        log_q_x_given_x_new = - (0.25 / (step_size * noise_scale + 1e-6)) * (diff_bwd ** 2).sum(dim=-1, keepdim=True) # (batch_size, ..., 1)

        diff_fwd = x_new - x - step_size * grad_log_prob_x # (..., dim)
        log_q_x_new_given_x = - (0.25 / (step_size * noise_scale + 1e-6)) * (diff_fwd ** 2).sum(dim=-1, keepdim=True) # (batch_size, ..., 1)

        log_q_dif = log_q_x_given_x_new - log_q_x_new_given_x # (batch_size, ..., 1)
        log_acceptance = log_prob_x_new - log_prob_x + log_q_dif # (batch_size, ..., 1)
        
        log_acceptance = torch.where(is_valid_step, log_acceptance, torch.tensor(-float('inf'), device=x.device)) # (batch_size, ..., 1)

        rand_val = torch.rand_like(log_acceptance).log() # (batch_size, ..., 1)
        accept = rand_val < log_acceptance # (batch_size, ..., 1)
        accept = accept & is_valid_step # (batch_size, ..., 1)
        
        final_x = torch.where(accept, x_new, x) # (batch_size, ..., dim)
        final_log_prob = torch.where(accept, log_prob_x_new, log_prob_x) # (batch_size, ..., 1)
        final_grad = torch.where(accept, grad_log_prob_x_new, grad_log_prob_x) # (batch_size, ..., dim)

        new_state = KernelState(
            final_x.detach(), 
            final_log_prob.detach(), 
            final_grad.detach()
        )
        
        # Also detach log_acceptance so adaptation doesn't track history
        return new_state, log_acceptance.detach()