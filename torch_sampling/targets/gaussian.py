import torch
from .target_distribution import TargetDistribution

class Gaussian(TargetDistribution):
    """
    Multivariate Gaussian distribution with support for full or diagonal covariance.
    """
    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor = None,
        prec: torch.Tensor = None,
        std: torch.Tensor = None
    ):
        """
        Arguments:
            mean: (dim,) tensor representing the mean.
            cov: (dim, dim) tensor representing the full covariance matrix.
            prec: (dim, dim) tensor representing the precision matrix (inverse covariance).
            std: (dim,) tensor representing the standard deviation for a diagonal covariance.
        """
        super().__init__()
        self.register_buffer("mean", mean)
        
        if cov is not None:
            self.register_buffer("cov", cov)
            self.prec = None
            self.scale_tril = None
        elif prec is not None:
            self.register_buffer("prec", prec)
            self.cov = None
            self.scale_tril = None
        elif std is not None:
            self.register_buffer("scale_tril", torch.diag(std))
            self.cov = None
            self.prec = None
        else:
            raise ValueError("One of 'cov', 'prec', or 'std' must be provided.")

        self._configure_distribution()

    def _configure_distribution(self):
        self.distribution = torch.distributions.MultivariateNormal(
            loc=self.mean,
            covariance_matrix=self.cov,
            scale_tril=self.scale_tril,
            precision_matrix=self.prec
        )
    
    def to(self, device: torch.device):
        super().to(device)
        self._configure_distribution()
        return self

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))