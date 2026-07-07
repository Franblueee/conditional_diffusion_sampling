import torch
from .target_distribution import TargetDistribution

class InfUniform(TargetDistribution):
    """
    Multivariate Uniform distribution defined over the entire R^d space.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._dim = dim
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: shape (batch_size, dim)
        
        Returns:
            log_prob: shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        log_prob = torch.zeros(batch_size, 1, device=x.device)
        return log_prob
    
    def grad_log_prob(self, x: torch.Tensor, return_log_prob: bool = False):
        """
        Arguments:
            x: shape (batch_size, dim)
            return_log_prob: if True, also return log_prob
        
        Returns:
            grad_log_prob: shape (batch_size, dim)
            log_prob (optional): shape (batch_size, 1)
        """
        batch_size, dim = x.shape
        grad_log_prob = torch.zeros(batch_size, dim, device=x.device)
        
        if return_log_prob:
            log_prob = self.log_prob(x)
            return grad_log_prob, log_prob
        else:
            return grad_log_prob