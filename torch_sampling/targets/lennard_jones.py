import numpy as np
import torch
from functools import partial
from scipy.interpolate import CubicSpline

# Assuming TargetDistribution is in a file named target_distribution.py
# If not, you'll need to provide this base class.
from .target_distribution import TargetDistribution 

# --- Optimized Helper Functions ---

def lennard_jones_energy(r, eps=1.0, rm=1.0):
    """
    Numerically stable Lennard-Jones 12-6 potential.
    """
    # Add a small epsilon for stability, though distances_from_vectors also adds one.
    r_safe = r + 1e-8 
    
    # Calculate the (rm / r)^6 term
    r_inv_6 = (rm / r_safe).pow(6)
    
    # Square the (rm / r)^6 term to get (rm / r)^12
    r_inv_12 = r_inv_6.pow(2)
    
    lj = eps * (r_inv_12 - 2.0 * r_inv_6)
    return lj

def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.
    
    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.
    
    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()

def cubic_spline(x_new, x, c):
    """
    Evaluates the cubic spline.
    
    Note: Assumes x and c are already on the same device as x_new.
    This is handled by registering them as buffers in the class.
    """
    # code from https://github.com/cambridge-mlg/Progressive-Tempering-Sampler-with-Diffusion/blob/main/ptsd/targets/lennard_jones.py
    intervals = torch.bucketize(x_new, x) - 1
    intervals = torch.clamp(intervals, 0, len(x) - 2)  # Ensure valid intervals
    
    # Calculate the difference from the left breakpoint of the interval
    dx = x_new - x[intervals]
    
    # Evaluate the cubic spline at x new
    y_new = (
        c[0, intervals] * dx**3
        + c[1, intervals] * dx**2
        + c[2, intervals] * dx
        + c[3, intervals]
    )
    return y_new

# --- Optimized LennardJones Class ---

class LennardJones(TargetDistribution):
    def __init__(
        self, 
        n_particles: int,
        eps: float = 1.0,
        rm: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 0.5,
        energy_factor: float = 1.0,
        interpolate: bool = True,
        interp_min: float = 0.65,
        interp_max: float = 2.0,
        n_interp_points: int = 1000,
        data_path: str = None
    ) -> None:
        """
        Lennard-Jones 12-6 potential target distribution.

        Arguments:
            n_particles (int): The number of particles in the system.
            eps (float): Depth of the potential well.
            rm (float): Distance at which the potential is zero.
            oscillator (bool): Whether to add an additional harmonic oscillator term.
            oscillator_scale (float): Scale for the harmonic oscillator term.
            energy_factor (float): Factor to scale the energy.
            interpolate (bool): Whether to use cubic spline interpolation for short distances.
            interp_min (float): Minimum distance for interpolation.
            interp_max (float): Maximum distance for interpolation.
            n_interp_points (int): Number of points to use for interpolation.
            data_path (str): Path to data file to load precomputed samples. 
        """
        super().__init__()


        self._n_particles = n_particles
        self._particle_dim = 3

        self._eps = eps
        self._rm = rm
        self._oscillator = oscillator
        self._oscillator_scale = oscillator_scale
        self._energy_factor = energy_factor
        self._interpolate = interpolate
        self._interp_min = interp_min
        self._interp_max = interp_max
        self._data_path = data_path

        if interpolate:
            interpolate_points = torch.linspace(interp_min, interp_max, n_interp_points)
            lj_energy = lennard_jones_energy(interpolate_points, eps, rm)
            coeffs = CubicSpline(
                interpolate_points.numpy(), lj_energy.numpy()
            ).c
            coeffs = torch.tensor(coeffs, dtype=torch.float32)
            
            # **OPTIMIZATION**: Register points and coefficients as buffers
            # This ensures they are moved to the correct device (e.g., GPU)
            # when .to(device) is called on the LennardJones object.
            self.register_buffer("_interp_points", interpolate_points)
            self.register_buffer("_interp_coeffs", coeffs)

            # self._spline_fn = partial(cubic_spline, x=self._interp_points, c=self._interp_coeffs)
            self._spline_fn = lambda x_new: cubic_spline(x_new, self._interp_points, self._interp_coeffs)
                 
    # def to(self, device):
    #     """
    #     Moves the distribution to the specified device.
    #     Arguments:
    #         device: torch.device
    #     Returns:
    #         self
    #     """
    #     super().to(device)
    #     # for some reason partial functions are not moved to the correct device
    #     self._spline_fn = partial(cubic_spline, x=self._interp_points, c=self._interp_coeffs)
    #     return self

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the distribution.
        Returns:
            dim: int
        """
        return self._n_particles * self._particle_dim

    @property
    def n_particles(self) -> int:
        """
        Returns the number of particles.
        Returns:
            n_particles: int
        """
        return self._n_particles

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._particle_dim)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _distance_vectors(self, x):
        """
        **OPTIMIZED**: Computes the matrix R of all distance vectors using broadcasting.
        
        Arguments:
            x: Tensor of shape `[n_batch, dim]`
        
        Returns:
            R: Tensor of shape `[n_batch, n_particles, n_particles, particle_dim]`
        """
        x_particles = x.view(-1, self._n_particles, self._particle_dim)
        
        # Use broadcasting to compute all pairs of differences
        r1 = x_particles.unsqueeze(2)  # Shape: [B, N, 1, D]
        r2 = x_particles.unsqueeze(1)  # Shape: [B, 1, N, D]
        
        # r1 - r2 results in shape [B, N, N, D]
        return r1 - r2

    def _energy(self, x):
        """
        **OPTIMIZED**: Computes the Lennard-Jones energy using diagonal masking.

        Arguments:
            x: tensor of samples of size (batch_size, dim) or (batch_size, n_particles, particle_dim)
        
        Returns:
            energy: tensor of energies of size (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # 1. Get all distance vectors (including diagonal)
        r_vecs = self._distance_vectors(x)  # Shape: [B, N, N, D]
        
        # 2. Get all distances (including diagonal)
        dists = distances_from_vectors(r_vecs)  # Shape: [B, N, N]

        # 3. Calculate LJ energies for all pairs
        lj_energies = lennard_jones_energy(dists, self._eps, self._rm)  # Shape: [B, N, N]
        
        if self._interpolate:
            # Create the mask for distances less than the interpolation minimum
            mask = (dists < self._interp_min)
            
            lj_energies = torch.where(
                mask,
                self._spline_fn(dists),
                lj_energies
            )
        
        # 4. Create a diagonal mask
        # We unsqueeze(0) to make it broadcastable with the batch dim [1, N, N]
        diag_mask = torch.eye(
            self._n_particles, 
            device=x.device, 
            dtype=torch.bool
        ).unsqueeze(0)
        
        # 5. Set diagonal (self-interaction) energies to 0.0
        # This is more compile-friendly than trying to index/remove them.
        lj_energies = torch.where(diag_mask, 0.0, lj_energies)
        
        # 6. Sum all energies. 
        # We divide by 2.0 because we counted both (i, j) and (j, i)
        lj_energies = lj_energies.sum(dim=(-2, -1)) * self._energy_factor

        if self._oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(batch_size)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies.unsqueeze(-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log probability of the input x

        Arguments:
            x: tensor of samples of size (batch_size, dim) or (batch_size, n_particles, particle_dim)
        Returns:
            log_prob: (batch_size, 1)
        """
        return -self._energy(x)

    def sample(self, num_samples):
        samples_array = np.load(self._data_path)
        idx = np.random.choice(samples_array.shape[0], num_samples, replace=False)
        samples = torch.tensor(samples_array[idx], dtype=torch.float32)
        return samples