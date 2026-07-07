import torch
import torch.nn as nn
import numpy as np
import h5py
import warnings

from .target_distribution import TargetDistribution

# OpenMM Imports
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit

warnings.filterwarnings("ignore")


class ALDP(TargetDistribution):
    def __init__(
        self,
        temperature : float = 300.0,
        data_path : str = None,
        regularize_energy: bool = False,
        check_chirality: bool = False,
        energy_cut: float = 1e8,
        energy_max: float = 1e20
    ) -> None:
        """
        Wraps the AmberEnergyModule into a TargetDistribution representing Alanine Dipeptide in vacuum.
        
        Args:
            prmtop_path (str): Path to the AMBER topology file.
            temperature (float): Temperature in Kelvin (default: 300K).
        """
        super().__init__()
        
        # 2. Physics Constants
        # Boltzmann constant in kJ / (mol * K)
        self.k_b = 0.008314462618 
        self.temperature = temperature
        self.beta = 1.0 / (self.k_b * self.temperature)
        
        # 3. Determine Dimensions
        # The energy module stores per-atom parameters (like charges)
        # We can infer N_atoms from there.
        self.n_atoms = 22  # Alanine Dipeptide has 22 atoms
        self._dim = self.n_atoms * 3

        self._data_path = data_path
        self.regularize_energy = regularize_energy
        self.energy_cut = energy_cut
        self.energy_max = energy_max

        self.check_chirality = check_chirality
        
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def n_particles(self) -> int:
        return self.n_atoms
    
    def _potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the potential energy of the system.
        
        Arguments:
            x: (batch_size, n_atoms, 3) coordinates
        
        Returns:
            energy: (batch_size,) potential energies
        """
        raise NotImplementedError("Potential energy calculation not implemented.")
    
    def _check_chirality(self, x):
        """
        Computes the signed volume of the chiral center.
        Positive/Negative determines L vs D form.
        
        Arguments:
            x: (batch_size, n_atoms, 3) coordinates
        
        Returns:
            volume: (batch_size,) signed volumes
        """
        # You need to find these indices for your specific topology
        # For standard Alanine Dipeptide (ACE-ALA-NME), usually:
        # Check your pdb/topology to confirm these indices!
        # These are illustrative indices for the ALA residue:
        idx_ca = 8   # CA
        idx_n  = 6   # N
        idx_c  = 14  # C
        idx_cb = 10  # CB
        
        # Vectors from CA to neighbors
        ca = x[:, idx_ca]
        n  = x[:, idx_n]
        c  = x[:, idx_c]
        cb = x[:, idx_cb]
        
        v1 = n - ca
        v2 = c - ca
        v3 = cb - ca
        
        # Scalar Triple Product
        # volume = dot(v1, cross(v2, v3))
        cross_prod = torch.linalg.cross(v2, v3, dim=1)
        volume = torch.sum(v1 * cross_prod, dim=1)
        
        return volume

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates unnormalized log probability: log_p = -E(x) / (kB * T)
        
        Args:
            x: (batch_size, dim) flattened coordinates in Nanometers.
        Returns:
            log_prob: (batch_size, 1)
        """
        # 1. Reshape flat input (B, N*3) -> (B, N, 3)
        batch_size = x.shape[0]
        
        # Validation to ensure dimensions match
        if x.shape[1] != self.dim:
            raise ValueError(f"Input dimension {x.shape[1]} does not match system dimension {self.dim}")

        coords = x.view(batch_size, self.n_atoms, 3)
        
        # 2. Calculate Potential Energy (kJ/mol)
        # The energy module is differentiable, so autograd will flow through this.
        potential_energy = self._potential_energy(coords)  # (batch_size,)

        if self.regularize_energy:
            potential_energy = self._regularize(potential_energy)
        
        tempered_energy = potential_energy * self.beta

        # 3. Convert to Log Probability
        # log p(x) = -E(x) * beta
        log_p = - tempered_energy

        if self.check_chirality:
            volumes = self._check_chirality(coords)
            # if volume < 0, penalize heavily
            penalty = torch.relu(-volumes) * 1e8
            log_p = log_p - penalty
        
        # Ensure output shape is (batch_size, 1)
        return log_p.view(batch_size, 1)
    
    def _regularize(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Regularizes extremely high energy values to avoid numerical instability.
        
        Args:
            energy: (batch_size,) tensor of potential energies.
        
        Returns:
            regularized_energy: (batch_size,) tensor of regularized energies.
        """
        # Cap the energy at energy_max
        energy = torch.where(energy < self.energy_max, energy, self.energy_max)
        # Make it logarithmic above energy cut and linear below
        energy = torch.where(
            energy < self.energy_cut, energy, torch.log(energy - self.energy_cut + 1) + self.energy_cut
        )
        energy = torch.where(torch.isfinite(energy), energy, torch.tensor(self.energy_max, device=energy.device))
        return energy

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Loads and returns precomputed samples from the file specified
        in `data_path` during initialization.

        Arguments:
            num_samples (int): The number of samples to return.

        Returns:
            samples (torch.Tensor): A tensor of samples.
        """
        if self._data_path is None:
            raise ValueError(
                "Cannot sample: `data_path` was not provided during initialization."
            )
        
        # samples_array = np.load(self._data_path)
        if self._data_path.endswith('.npy'):
            samples_array = np.load(self._data_path)
        elif self._data_path.endswith('.pt'):
            samples_array = torch.load(self._data_path).numpy()
        elif self._data_path.endswith('.h5'):
            samples_array = h5py.File(self._data_path, 'r')['coordinates'][:]
        else:
            raise NotImplementedError(
                f"Loading samples from {self._data_path} is not implemented. Use .npy, .pt, or .h5"
            )
            
        if samples_array.shape[0] < num_samples:
            raise ValueError(
                f"Requested {num_samples} samples, but only {samples_array.shape[0]} are available in {self._data_path}."
            )
            
        idx = np.random.choice(samples_array.shape[0], num_samples, replace=False)
        samples = torch.tensor(samples_array[idx], dtype=torch.float32)
        samples = samples.view(num_samples, -1)
        return samples