import torch
from .target_distribution import TargetDistribution

class Uniform(TargetDistribution):
    """
    Multivariate Uniform distribution defined over a hypercube.
    """
    def __init__(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
    ):
        """
        Arguments:
            low: (dim,) tensor representing the lower bound of the hypercube.
            high: (dim,) tensor representing the upper bound of the hypercube.
        """
        super().__init__()
        
        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape.")
        
        # Register buffers so they move with the model (to GPU/CPU)
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        
        self._configure_distribution()

    def _configure_distribution(self):
        """
        Initializes the underlying PyTorch distribution.
        reinterpreted_batch_ndims=1 treats the last dimension as part of the event.
        """
        base_dist = torch.distributions.Uniform(low=self.low, high=self.high)
        self.distribution = torch.distributions.Independent(base_dist, reinterpreted_batch_ndims=1)
    
    def to(self, device: torch.device):
        super().to(device)
        self._configure_distribution()
        return self

    @property
    def dim(self) -> int:
        return self.low.shape[0]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # view(-1, 1) ensures the output shape is (batch_size, 1)
        return self.distribution.log_prob(x).view(-1, 1)

    def grad_log_prob(self, x, return_log_prob = False):
        # The gradient of the log-probability of a uniform distribution is zero within the support
        grad = torch.zeros_like(x)
        if return_log_prob:
            log_prob = self.log_prob(x)
            return grad, log_prob
        return grad


    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))