import torch
from tqdm import tqdm

from ..targets import TargetDistribution
from ..utils.kernel_state import KernelState


class Sampler(torch.nn.Module):
    """
    General Sampler class for MCMC methods.    
    """
    def __init__(
        self,
        target : TargetDistribution,
        verbose : bool = False
    ) -> None:
        super().__init__()
        self.target = target
        self.verbose = verbose
        self_current_state: KernelState = None

    @property
    def dim(self) -> int:
        """
        Dimension of the sampler data.
        """
        return self.target.dim

    def _init_state(
        self,
        x: torch.Tensor
    ) -> KernelState:
        """
        Initialize the sampler state.
        
        Arguments:
            x: (batch_size, dim) tensor of points
        
        Returns:
            state: KernelState
        """
        grad, log_prob = self.target.grad_log_prob(x, return_log_prob=True)
        self._current_state = KernelState(x=x, log_prob=log_prob, grad=grad)

    def build_initial_point(
        self,
        n_samples : int = 1,
        device : torch.device = torch.device("cpu"),
        dtype : torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Build an initial point for the sampler. Used in the method `sample` and `sample_trajectory`.
        
        Arguments:
            n_samples: number of samples to generate
            device: device to place the tensor on
            dtype: data type of the tensor
        
        Returns:
            x0: (n_samples, dim) tensor of initial points
        """
        return torch.randn(n_samples, self.dim, device=device, dtype=dtype)
    
    def forward(
        self, 
        x0: torch.Tensor,
        n_steps: int = 1,
        *args, 
        return_trajectory : bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the sampler. This is a wrapper around the step function.
        
        Arguments:
            x0: starting point (batch_size, dim)
            n_steps: number of steps to take
            return_trajectory: if True, returns the trajectory of points, otherwise returns only the last point.
        
        Returns:
            xs: (n_steps + 1, batch_size, dim) if return_trajectory is True,
            x: (batch_size, dim) if return_trajectory is False
        """

        if self.verbose:
            pbar = tqdm(total=n_steps)
        
        self._init_state(x0)

        x = x0.clone()

        if return_trajectory:
            xs = [x.clone().to("cpu")]
            for _ in range(n_steps):
                if self.verbose:
                    pbar.update(1)
                x = self.step(x, *args, **kwargs)
                xs.append(x.clone().to("cpu"))
            result = torch.stack(xs)
        else:
            for _ in range(n_steps):
                if self.verbose:
                    pbar.update(1)
                x = self.step(*args, **kwargs)
            result = x
        
        if self.verbose:
            pbar.close()

        return result
    
    def step(
        self, 
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        Kernel step function. Given a point x_k, returns a new point x_{k+1}.
        
        Arguments:
            x: (batch_size, dim)
        
        Returns:
            x_new: (batch_size, dim)
        """
        raise NotImplementedError("Kernel step function must be implemented in the subclass.")

    def sample_trajectory(
        self, 
        n_samples : int = 1,
        n_steps : int = 1,
        *args,
        device : torch.device = torch.device("cpu"),
        **kwargs
    ) -> torch.Tensor:
        """
        Sample a trajectory of points starting from x0.

        Arguments:
            n_samples: number of samples to generate
            n_steps: number of steps to take
            device: device to place the initial tensor on
        
        Returns:
            xs: (n_steps + 1, batch_size, dim)
        """
        x0 = self.build_initial_point(
            n_samples=n_samples,
            device=device,
            dtype=torch.float32
        )
        return self.forward(x0, n_steps, *args, return_trajectory=True, **kwargs)

    def sample(
        self, 
        n_samples : int = 1,
        n_steps : int = 1,
        *args,
        device : torch.device = torch.device("cpu"),
        **kwargs
    ) -> torch.Tensor:
        """
        Sample a single point starting from x0.
        
        Arguments:
            n_samples: number of samples to generate
            n_steps: number of steps to take
            device: device to place the initial tensor on
        
        Returns:
            x: (batch_size, dim)
        """
        x0 = self.build_initial_point(
            n_samples=n_samples,
            device=device,
            dtype=torch.float32
        )
        return self.forward(x0, n_steps, *args, return_trajectory=False, **kwargs)
