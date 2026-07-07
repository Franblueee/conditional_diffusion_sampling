import torch
from .target_distribution import TargetDistribution
from .inf_uniform import InfUniform

class AnnealedTarget(TargetDistribution):
    """
    A lightweight wrapper that scales the base target by a scalar beta.
    """
    def __init__(
        self,
        target: TargetDistribution, 
        beta: torch.Tensor,
        reference: TargetDistribution = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.reference = reference
        self.beta = beta 

        if self.reference is None:
            # Default reference is an infinite uniform distribution
            dim = self.target.dim
            self.reference = InfUniform(dim=dim)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        target_log_prob = self.target.log_prob(x)
        reference_log_prob = self.reference.log_prob(x)

        log_prob = target_log_prob * self.beta + reference_log_prob * (1 - self.beta)
        return log_prob

    def grad_log_prob(self, x: torch.Tensor, return_log_prob: bool = False):
        if return_log_prob:
            grad_target, lp_target = self.target.grad_log_prob(x, return_log_prob=True)
            grad_reference, lp_reference = self.reference.grad_log_prob(x, return_log_prob=True)
            grad = grad_target * self.beta + grad_reference * (1 - self.beta)
            lp = lp_target * self.beta + lp_reference * (1 - self.beta)
            return grad, lp
        else:
            grad_target, lp_target = self.target.grad_log_prob(x, return_log_prob=True)
            grad_reference, lp_reference = self.reference.grad_log_prob(x, return_log_prob=True)
            grad = grad_target * self.beta + grad_reference * (1 - self.beta)
            return grad