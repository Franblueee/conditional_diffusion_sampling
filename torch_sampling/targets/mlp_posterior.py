import math
import torch

from sampling.targets.target_distribution import TargetDistribution

class MLPPosterior(TargetDistribution):
    """
    """
    def __init__(
        self,
        n_data: int = 500,
        d_in: int = 20,
        d_hidden : int = 25,
        prior_std : float = 1.0,
        likelihood_std : float = 0.1
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        
        # Hyperparameters
        self.prior_std = prior_std
        self.s_n = likelihood_std
        
        # Calculate total dimension: (W1: d_in*d_hidden) + (b1: d_hidden) + (W2: d_hidden*1)
        self._dim = self.d_in * self.d_hidden + self.d_hidden + self.d_hidden

        # Generate random data for regression
        self._theta_true = torch.randn(1, self._dim) * prior_std # (1, dim)
        with torch.no_grad():
            X_train = torch.randn(n_data, d_in) # (n_data, d_in)
            y_train = self.forward(X_train, self._theta_true).squeeze(0) + \
                          torch.randn(n_data) * likelihood_std # (n_data,)
            
            X_test = torch.randn(n_data, d_in) # (n_data, d_in)
            y_test = self.forward(X_test, self._theta_true).squeeze(0) + \
                          torch.randn(n_data) * likelihood_std # (n_data,)        
        
        self.register_buffer('X_train', X_train)
        self.register_buffer('y_train', y_train)
        self.register_buffer('X_test', X_test)
        self.register_buffer('y_test', y_test)

    @property
    def dim(self) -> int:
        return self._dim

    def _unpack_params(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Theta shape: (batch_size, dim)
        Returns: W1, b1, W2 reshaped for batch matrix multiplication
        """
        d_in, d_hidden = self.d_in, self.d_hidden
        
        # 1. Weights from Input to Hidden
        idx1 = d_in * d_hidden
        w1 = theta[:, :idx1].view(-1, d_in, d_hidden)
        
        # 2. Biases for Hidden layer
        idx2 = idx1 + d_hidden
        b1 = theta[:, idx1:idx2].view(-1, 1, d_hidden)
        
        # 3. Weights from Hidden to Output (assuming 1D output)
        w2 = theta[:, idx2:].view(-1, d_hidden, 1)
        
        return w1, b1, w2

    def forward(
        self,
        X: torch.Tensor,
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MLP output for a batch of weights theta at inputs X.
        
        Arguments:
            X: Input data of shape (n_data, d_in)
            theta: Weights of shape (n_samples, dim)

        Returns:
            y_hat: Predicted outputs of shape (batch_size, n_data)        
        """

        w1, b1, w2 = self._unpack_params(theta)
        # w1: (n_samples, d_in, d_hidden)
        # b1: (n_samples, 1, d_hidden)
        # w2: (n_samples, d_hidden, 1)

        X_in = X.unsqueeze(0).expand(theta.size(0), -1, -1) # (n_samples, n_data, d_in)
        
        # z1 = X_in @ w1 + b1  # (n_samples, n_data, d_hidden)
        z1 = torch.bmm(X_in, w1) / math.sqrt(self.d_in) + b1 # (n_samples, n_data, d_hidden)
        a1 = torch.relu(z1)  # (n_samples, n_data, d_hidden)
        y_hat = torch.bmm(a1, w2) / math.sqrt(self.d_hidden) # (n_samples, n_data, 1)
        
        return y_hat.squeeze(-1) # (n_samples, n_data)
    
    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculates log p(theta) for Gaussian prior N(0, prior_std^2)

        Arguments:
            theta: Parameters of shape (n_samples, dim)

        Returns:
            log_prior: Log-prior of shape (n_samples, 1)
        """
        log_prior = -0.5 * torch.sum(theta**2, dim=-1) / (self.prior_std**2)
        return log_prior.unsqueeze(-1) # (n_samples, 1)

    def log_likelihood(
        self,
        X: torch.Tensor, 
        y_true: torch.Tensor,
        theta: torch.Tensor
    ) -> torch.Tensor:
        
        y_pred = self.forward(X, theta) # (n_samples, n_data)
        mse = torch.sum((y_pred - y_true)**2, dim=-1) # (n_samples,)
        log_likelihood = -0.5 * mse / (self.s_n**2) # (n_samples,)
        log_likelihood += -0.5 * X.size(0) * math.log(2 * math.pi * self.s_n**2)
        return log_likelihood.unsqueeze(-1) # (n_samples, 1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculates log p(theta | Data) proportional to log p(y | X_{train}, theta) + log p(theta)

        Arguments:
            theta: Parameters of shape (n_samples, dim)

        Returns:
            log_posterior: Unnormalized log-posterior of shape (n_samples, 1)
        """
        log_prior = self.log_prior(theta) # (n_samples, 1)
        log_likelihood = self.log_likelihood(
            self.X_train,
            self.y_train,
            theta
        )
        
        log_posterior = log_likelihood + log_prior # (n_samples, 1)
        return log_posterior
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the prior distribution N(0, prior_std^2)
        """
        return torch.randn(num_samples, self.dim) * self.prior_std