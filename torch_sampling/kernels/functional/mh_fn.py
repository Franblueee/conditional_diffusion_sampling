import torch
from typing import Tuple

from ...targets.target_distribution import TargetDistribution
from ...utils.kernel_state import KernelState

def mh_step(
    target: TargetDistribution,
    state: KernelState,
    proposal_std: torch.Tensor,
) -> Tuple[KernelState, torch.Tensor]:
    
    x, log_prob_x = state.x, state.log_prob

    # Direct tensor multiplication
    x_new = x + proposal_std * torch.randn_like(x)
    
    log_prob_x_new = target.log_prob(x_new)
    log_acceptance = log_prob_x_new - log_prob_x
    
    rand_val = torch.rand_like(log_acceptance).log()
    accept = rand_val < log_acceptance
    
    final_x = torch.where(accept, x_new, x)
    final_log_prob = torch.where(accept, log_prob_x_new, log_prob_x)

    new_state = KernelState(x=final_x.detach(), log_prob=final_log_prob.detach())
    
    return new_state, log_acceptance.detach()