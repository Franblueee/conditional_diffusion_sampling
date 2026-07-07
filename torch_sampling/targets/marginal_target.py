import torch
from sampling.targets.target_distribution import TargetDistribution

class MarginalTarget(TargetDistribution):

    def __init__(
        self, 
        target : TargetDistribution,
        time : float | torch.Tensor,
        sigma: float = 1.0,
        n_samples: int = 100
    ) -> None:
        """
        The marginal target distribution $\pi_t(x)$, defined by $\pi_t(x) = \int \pi_t(x | z) \pi_{ref}(z) dz$.

        Arguments:
            target: the target distribution to sample from
            time: the time parameter, a float or a tensor of shape (1,) representing the interpolation factor
            sigma: the standard deviation of the noise added to the samples from the reference distribution
            n_samples: the number of samples to use for the marginalization        
        """
        super().__init__()
        self.target = target
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float32, device=target.device)
        self.time = time
        self.n_samples = n_samples
        self.sigma = sigma  # Standard deviation of the noise added to the samples from the reference distribution
        self.eps = 1e-6  # Small value to avoid division by zero in log_prob and grad_log_prob

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of the input x.

        Arguments:
            x: (batch_size, dim) tensor of input samples
        
        Returns:
            log_prob_x: (batch_size, 1) tensor of log probabilities
        """
        time = self.time.to(x.device)
        batch_size = x.shape[0]
        dim = x.shape[1]
        z = self.sigma*torch.randn(self.n_samples, batch_size, dim, device=x.device)  # (n_samples, batch_size, dim)
        x = x.unsqueeze(0).expand(self.n_samples, -1, -1)  # (n_samples, batch_size, dim)
        div = 1 / (1 - time + self.eps)
        x_0 = div*x - div*time*z # (n_samples, batch_size, dim)
        x_0 = x_0.reshape(-1, dim)  # (n_samples * batch_size, dim)
        log_prob = self.target.log_prob(x_0) - dim * torch.log(1 - time) # (n_samples*batch_size, 1)
        log_prob = log_prob.reshape(self.n_samples, batch_size)  # (n_samples, batch_size)
        log_prob = log_prob.mean(dim=0)  # Sum over the samples
        return log_prob

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability of the input x.

        Arguments:
            x: (batch_size, dim) tensor of input samples
        
        Returns:
            log_prob_x: (batch_size, 1) tensor of log probabilities
        """
        time = self.time.to(x.device)
        batch_size = x.shape[0]
        dim = x.shape[1]
        z = self.sigma*torch.randn(self.n_samples, batch_size, dim, device=x.device)  # (n_samples, batch_size, dim)
        x = x.unsqueeze(0).expand(self.n_samples, -1, -1)  # (n_samples, batch_size, dim)
        div = 1 / (1 - time + self.eps)
        x_0 = div*x - div*time*z # (n_samples, batch_size, dim)
        x_0 = x_0.reshape(-1, dim)  # (n_samples * batch_size, dim)
        log_prob = self.target.log_prob(x_0) - dim * torch.log(1 - time) # (n_samples*batch_size, 1)
        prob = torch.exp(log_prob)
        prob = prob.reshape(self.n_samples, batch_size)
        prob = prob.mean(dim=0)
        return prob
    
    def grad_log_prob(self, x: torch.Tensor, return_log_prob: bool = False) -> torch.Tensor:
        """
        Returns the gradient of log_prob.

        Arguments:
            x: (batch_size, dim) tensor of input samples
            return_log_prob: whether to return the log probability as well
            
        Returns:
            grad_log_prob_x: (batch_size, dim) tensor of gradients of log probabilities
            log_prob_x: (batch_size, 1) tensor of log probabilities, if return_log_prob is True
        """
        raise NotImplementedError("MarginalTarget does not support gradient computation yet.")
