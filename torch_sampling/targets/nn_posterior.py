import torch
import torch.nn as nn
import torch.func

from sampling.targets import TargetDistribution

class NNPosterior(TargetDistribution):
    """
    Represents the posterior distribution of a neural network's weights.
    
    The log_prob(x) computes the unnormalized log-posterior:
    log p(x | Data) = log p(Data | x) + log p(x)
    
    where:
    - x: A flattened vector (or batch of vectors) of network weights (theta).
    - log p(Data | x): The log-likelihood of the *entire* dataset given weights x.
    - log p(x): The log-prior of the weights x.
    """
    
    def __init__(self, model: nn.Module, data_X: torch.Tensor, data_Y: torch.Tensor, 
                 task_type: str = 'classification', prior_std: float = 1.0, 
                 likelihood_std: float = 0.1):
        """
        Initializes the posterior distribution.
        
        Args:
            model: The neural network module (e.g., an MLP).
            data_X: The training features (fixed).
            data_Y: The training labels (fixed).
            task_type: 'classification' or 'regression'.
            prior_std: The standard deviation of the Gaussian prior N(0, prior_std^2)
                       applied to all weights.
            likelihood_std: The standard deviation of the Gaussian likelihood
                            (for regression task only).
        """
        super().__init__()
        self.model = model
        self.data_X = data_X
        self.data_Y = data_Y
        self.task_type = task_type
        
        # --- Store parameter/buffer names and shapes ---
        # This is needed to map a flat vector back to the model's state dict
        
        # We only need the *names* and *shapes* to reconstruct the dict
        # from a flat vector. We detach them to avoid memory leaks.
        self.param_shapes = {
            name: param.shape 
            for name, param in model.named_parameters()
        }
        self.param_names = list(self.param_shapes.keys())
        
        # Buffers are non-parameter tensors (e.g., batchnorm running means)
        # They are fixed (not part of 'x') but needed for functional_call
        self.buffer_dict = {
            name: buffer.detach() 
            for name, buffer in model.named_buffers()
        }
        
        # --- Define Prior ---
        # Simple N(0, prior_std^2) prior for all weights
        self.prior = torch.distributions.Normal(0.0, prior_std)
        
        # --- Define Likelihood (for regression) ---
        if self.task_type == 'regression':
            self.likelihood_std = likelihood_std
        elif self.task_type != 'classification':
            raise ValueError("task_type must be 'classification' or 'regression'")

        # --- Calculate dimension ---
        self._dim = sum(torch.prod(torch.tensor(s)).item() for s in self.param_shapes.values())

    @property
    def dim(self) -> int:
        return self._dim

    def _vector_to_param_dict(self, vector: torch.Tensor) -> dict:
        """
        Converts a flat 1D vector into a dictionary of parameters
        matching the model's structure.
        """
        params_dict = {}
        current_idx = 0
        for name in self.param_names:
            shape = self.param_shapes[name]
            numel = torch.prod(torch.tensor(shape)).item()
            # Slice the vector and reshape
            params_dict[name] = vector[current_idx : current_idx + numel].view(shape)
            current_idx += numel
        return params_dict

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the unnormalized log posterior for a batch of weight vectors.
        
        Args:
            x: (batch_size, dim) - A batch of flattened weight vectors.
        
        Returns:
            log_prob: (batch_size, 1)
        """
        
        # 1. Calculate Log-Prior: log p(x)
        # We assume an independent N(0, prior_std^2) prior for all weights.
        # log_prob(x) gives per-weight log-prob, so we sum them up.
        log_prior = self.prior.log_prob(x).sum(dim=1, keepdim=True)
        
        # 2. Calculate Log-Likelihood: log p(Data | x)
        # We need to do this for each weight vector in the batch.
        log_likelihoods = []
        
        for i in range(x.shape[0]):
            # Get the i-th weight vector
            flat_params_i = x[i]
            
            # Reconstruct the model's parameter dictionary
            param_dict_i = self._vector_to_param_dict(flat_params_i)
            
            # --- Perform a functional forward pass ---
            # This computes self.model(self.data_X) but uses
            # param_dict_i as its parameters instead of self.model.parameters().
            # This whole operation is differentiable w.r.t. flat_params_i.
            preds = torch.func.functional_call(
                self.model,
                (self.buffer_dict, param_dict_i), # Buffers and Parameters
                args=(self.data_X,)                 # Model inputs
            )
            
            # --- Compute log-likelihood for the *entire* dataset ---
            if self.task_type == 'classification':
                # Use cross_entropy, which is a *negative* log-likelihood.
                # We use reduction='sum' to get the LL for the whole dataset.
                log_lik_i = -torch.nn.functional.cross_entropy(
                    preds, self.data_Y, reduction='sum'
                )
            else: # regression
                # Gaussian likelihood: N(y | f(x), likelihood_std^2)
                dist = torch.distributions.Normal(preds, self.likelihood_std)
                log_lik_i = dist.log_prob(self.data_Y).sum()
                
            log_likelihoods.append(log_lik_i)
            
        log_likelihood = torch.stack(log_likelihoods).unsqueeze(-1)
        
        # 3. Return Log-Posterior = Log-Likelihood + Log-Prior
        return log_likelihood + log_prior