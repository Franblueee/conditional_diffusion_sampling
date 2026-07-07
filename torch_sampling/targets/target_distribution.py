import torch

class TargetDistribution(torch.nn.Module):
    """
    Base class for target distributions.
    """
    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs)

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the distribution.
        Returns:
            dim: int
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of the input x
        Arguments:
            x: (batch_size, dim)
        Returns:
            log_prob: (batch_size, 1)
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability of the input x
        Arguments:
            x: (batch_size, dim)
        Returns:
            prob: (batch_size, 1)
        """
        return self.log_prob(x).exp()        

    def grad_log_prob(self, x: torch.Tensor, return_log_prob: bool = False) -> torch.Tensor:
        """
        Returns the gradient of log_prob
        Arguments:
            x: (batch_size, dim)
        return_log_prob: if True, also returns the log probability
        Returns:
            grad_log_prob: (batch_size, dim)
            log_prob: (batch_size, 1) if return_log_prob is True
        """
        x_ = x.detach()  # (batch_size, 1, dim)
        x_.requires_grad = True
        log_prob_x = self.log_prob(x_) # (batch_size, 1)
        grad_log_prob_x = torch.autograd.grad(
            outputs=log_prob_x,
            inputs=x_,
            grad_outputs=torch.ones_like(log_prob_x),
            create_graph=False,
        )[0]
        if return_log_prob:
            return grad_log_prob_x, log_prob_x
        return grad_log_prob_x
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the distribution.
        Arguments:
            num_samples: number of samples to generate
        Returns:
            samples: (num_samples, dim)
        """
        raise NotImplementedError("Subclasses should implement this method.")