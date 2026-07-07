import torch
from .target_distribution import TargetDistribution

class GaussianMixture(TargetDistribution):
    """
    Gaussian Mixture (GM) distribution.
    """
    def __init__(
        self,
        means: torch.Tensor,
        covs: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        """
        Arguments:
            means: (n_modes, dim) tensor of means for each Gaussian component
            covs: (n_modes, dim, dim) tensor of covariance matrices for each Gaussian component
            weights: (n_modes,) tensor of weights for each Gaussian component
        """
        super().__init__()
        n_modes = means.shape[0]
        if weights is None:
            weights = torch.ones(n_modes, dtype=torch.float32) / n_modes
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]
    
    @property
    def distribution(self) -> torch.distributions.Distribution:
        """
        Returns the underlying distribution.
        """
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(probs=self.weights, validate_args=False),
            component_distribution=torch.distributions.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
            ),
            validate_args=False,
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of the input x
        Arguments:
            x: (batch_size, dim)
        Returns:
            log_prob: (batch_size, 1)
        """
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the Gaussian mixture distribution.
        Arguments:
            num_samples: number of samples to generate
        Returns:
            samples: (num_samples, dim)
        """
        return self.distribution.sample(torch.Size((num_samples,)))