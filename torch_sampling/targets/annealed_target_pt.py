import torch
from .target_distribution import TargetDistribution
from .inf_uniform import InfUniform

class AnnealedTargetPT(TargetDistribution):
    def __init__(
        self, 
        target: TargetDistribution, 
        beta_schedule: torch.Tensor,
        reference: TargetDistribution = None
    ) -> None:
        super().__init__()
        self.target = target
        self.reference = reference
        # Keep beta_schedule as a buffer-like attribute
        self.beta_schedule = beta_schedule 

        if self.reference is None:
            # Default reference is an infinite uniform distribution
            dim = self.target.dim
            self.reference = InfUniform(dim=dim)

    def to(self, device: torch.device):
        super().to(device)
        self.beta_schedule = self.beta_schedule.to(device)
        self.target = self.target.to(device)
        self.reference = self.reference.to(device)
        return self

    @property
    def n_replicas(self) -> int:
        return len(self.beta_schedule)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)
        
        Returns:
            log_prob: shape (batch_size, n_replicas, 1)
        """

        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim) # (batch_size*n_replicas, dim)
        
        log_prob_target = self.target.log_prob(flat_x) # (batch_size*n_replicas, 1)
        log_prob_target = log_prob_target.view(batch_size, n_replicas, 1) # (batch_size, n_replicas, 1)

        log_prob_reference = self.reference.log_prob(flat_x) # (batch_size*n_replicas, 1)
        log_prob_reference = log_prob_reference.view(batch_size, n_replicas, 1) # (batch_size, n_replicas, 1)

        betas = self.beta_schedule.to(x.device).view(1, n_replicas, 1) # (1, n_replicas, 1)
        log_prob = log_prob_target * betas + log_prob_reference * (1 - betas) # (batch_size, n_replicas, 1)
        return log_prob

    def grad_log_prob(self, x: torch.Tensor, return_log_prob: bool = False):
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)

        Returns:
            grad_log_prob: shape (batch_size, n_replicas, dim)
            log_prob (optional): shape (batch_size, n_replicas, 1)
        """
        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim) # (n_replicas*batch_size, dim)
        betas = self.beta_schedule.to(x.device).view(1, n_replicas, 1) # (1, n_replicas, 1)

        if return_log_prob:
            grad_target, lp_target = self.target.grad_log_prob(flat_x, return_log_prob=True) # (n_replicas*batch_size, dim), (n_replicas*batch_size, 1)
            grad_target = grad_target.view(batch_size, n_replicas, dim) # (batch_size, n_replicas, dim)
            lp_target = lp_target.view(batch_size, n_replicas, 1) # (batch_size, n_replicas, 1)

            grad_reference, lp_reference = self.reference.grad_log_prob(flat_x, return_log_prob=True) # (n_replicas*batch_size, dim), (n_replicas*batch_size, 1)
            grad_reference = grad_reference.view(batch_size, n_replicas, dim)
            lp_reference = lp_reference.view(batch_size, n_replicas, 1)

            grad = grad_target * betas + grad_reference * (1 - betas) # (batch_size, n_replicas, dim)
            lp = lp_target * betas + lp_reference * (1 - betas)

            return grad, lp
        else:
            grad_target, lp_target = self.target.grad_log_prob(flat_x, return_log_prob=True) # (n_replicas*batch_size, dim), (n_replicas*batch_size, 1)
            grad_target = grad_target.view(batch_size, n_replicas, dim) # (batch_size, n_replicas, dim)
            
            grad_reference, lp_reference = self.reference.grad_log_prob(flat_x, return_log_prob=True) # (n_replicas*batch_size, dim), (n_replicas*batch_size, 1)
            grad_reference = grad_reference.view(batch_size, n_replicas, dim)

            grad = grad_target * betas + grad_reference * (1 - betas)

            return grad
    
    def log_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)
        
        Returns:
            log_ratio: shape (batch_size, n_replicas, 1)
        """

        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim)
        log_prob_target = self.target.log_prob(flat_x)
        log_prob_reference = self.reference.log_prob(flat_x)
        log_ratio = log_prob_target - log_prob_reference
        log_ratio = log_ratio.view(batch_size, n_replicas, 1)
        return log_ratio

    def grad_log_ratio(self, x: torch.Tensor, return_log_ratio: bool = False):
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)

        Returns:
            grad_log_ratio: shape (batch_size, n_replicas, dim)
            log_ratio (optional): shape (batch_size, n_replicas, 1)
        """
        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim)

        if return_log_ratio:
            grad_target, lp_target = self.target.grad_log_prob(flat_x, return_log_prob=True)
            grad_reference, lp_reference = self.reference.grad_log_prob(flat_x, return_log_prob=True)

            grad_ratio = grad_target - grad_reference
            lp_ratio = lp_target - lp_reference

            grad_ratio = grad_ratio.view(batch_size, n_replicas, dim)
            lp_ratio = lp_ratio.view(batch_size, n_replicas, 1)

            return grad_ratio, lp_ratio
        else:
            grad_target = self.target.grad_log_prob(flat_x, return_log_prob=False)
            grad_reference = self.reference.grad_log_prob(flat_x, return_log_prob=False)

            grad_ratio = grad_target - grad_reference
            grad_ratio = grad_ratio.view(batch_size, n_replicas, dim)

            return grad_ratio

    
    def log_prob_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)
        
        Returns:
            log_prob_target: shape (batch_size, n_replicas, 1)
        """

        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim)
        log_prob_target = self.target.log_prob(flat_x)
        log_prob_target = log_prob_target.view(batch_size, n_replicas, 1)
        return log_prob_target
    
    def grad_log_prob_target(self, x: torch.Tensor, return_log_prob: bool = False):
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)

        Returns:
            grad_log_prob_target: shape (batch_size, n_replicas, dim)
            log_prob_target (optional): shape (batch_size, n_replicas, 1)
        """
        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim)

        if return_log_prob:
            grad_target, lp_target = self.target.grad_log_prob(flat_x, return_log_prob=True)
            grad_target = grad_target.view(batch_size, n_replicas, dim)
            lp_target = lp_target.view(batch_size, n_replicas, 1)
            return grad_target, lp_target
        else:
            grad_target = self.target.grad_log_prob(flat_x, return_log_prob=False)
            grad_target = grad_target.view(batch_size, n_replicas, dim)
            return grad_target
    
    def log_prob_reference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)
        
        Returns:
            log_prob_reference: shape (batch_size, n_replicas, 1)
        """

        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim)
        log_prob_reference = self.reference.log_prob(flat_x)
        log_prob_reference = log_prob_reference.view(batch_size, n_replicas, 1)
        return log_prob_reference
    
    def grad_log_prob_reference(self, x: torch.Tensor, return_log_prob: bool = False):
        """
        Arguments:
            x: shape (batch_size, n_replicas, dim)

        Returns:
            grad_log_prob_reference: shape (batch_size, n_replicas, dim)
            log_prob_reference (optional): shape (batch_size, n_replicas, 1)
        """
        batch_size, n_replicas, dim = x.shape
        flat_x = x.view(-1, dim)

        if return_log_prob:
            grad_reference, lp_reference = self.reference.grad_log_prob(flat_x, return_log_prob=True)
            grad_reference = grad_reference.view(batch_size, n_replicas, dim)
            lp_reference = lp_reference.view(batch_size, n_replicas, 1)
            return grad_reference, lp_reference
        else:
            grad_reference = self.reference.grad_log_prob(flat_x, return_log_prob=False)
            grad_reference = grad_reference.view(batch_size, n_replicas, dim)
            return grad_reference