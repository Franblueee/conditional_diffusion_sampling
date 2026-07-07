import torch
from .target_distribution import TargetDistribution

class ConditionalTarget(TargetDistribution):
    def __init__(
        self, 
        target: TargetDistribution,
        time: float | torch.Tensor,
        z: torch.Tensor = None
    ) -> None:
        """
        Conditional target distribution \pi_t(x_t | z) defined by x_t = (1-t)z + tx.
        
        Arguments:
            target: the target distribution to sample from
            time: interpolation factor (0, 1]
            z: initial condition tensor. Shape can be (dim,) or (batch_size, dim)
        """
        super().__init__()
        self.target = target
        
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float32)
        self.time = time
        self.z = z

    def set_z(self, z: torch.Tensor):
        """Update the condition z. Accepts (dim,) or (batch_size, dim)."""
        self.z = z

    @property
    def dim(self) -> int:
        return self.target.dim
    
    @property
    def n_particles(self) -> int:
        return self.target.n_particles

    def _get_broadcasted_z(self, device: torch.device):
        if self.z is None:
            raise ValueError("Parameter 'z' must be set before calling this method.")
        
        z = self.z.to(device)
        # If z is (dim,), unsqueeze to (1, dim) so it broadcasts across the batch
        if z.dim() == 1:
            return z.unsqueeze(0)
        return z

    def log_prob(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xt: (batch_size, dim) tensor
        """
        z = self._get_broadcasted_z(xt.device)
        
        # Invert the interpolation: x = (xt - (1-t)z) / t
        x = (xt - (1.0 - self.time) * z) / self.time
        # x = z + (xt - z) / self.time
        
        # log p(xt) = log p(x) - dim * log(t)
        return self.target.log_prob(x)

    def grad_log_prob(self, xt: torch.Tensor, return_log_prob: bool = False) -> torch.Tensor | tuple:
        """
        Arguments:
            xt: (batch_size, dim) tensor
        """
        z = self._get_broadcasted_z(xt.device)
        x = (xt - (1.0 - self.time) * z) / self.time
        # x = z + (xt - z) / self.time
        
        out = self.target.grad_log_prob(x, return_log_prob=return_log_prob)
        
        if return_log_prob:
            grad_log_prob_x, log_prob_x = out
            grad_log_prob_xt = grad_log_prob_x
            log_prob_xt = log_prob_x
            return grad_log_prob_xt, log_prob_xt
        
        return out

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Arguments:
            num_samples: Number of samples to draw. 
                         If z is a batch, it should match z.shape[0].
        """
        z = self._get_broadcasted_z(self.time.device)
        target_samples = self.target.sample(num_samples)
        
        # x_t = (1-t)z + tx
        return (1.0 - self.time) * z + self.time * target_samples