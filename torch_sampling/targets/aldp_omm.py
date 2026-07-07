import numpy as np
import torch

# Import base class
from .aldp import ALDP

# Imports for ALDP energy calculation
from simtk import (
    unit,
    openmm as mm,
)
from openmmtools import testsystems

class OpenMMInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context):
        """
        input: (Batch, 66) tensor on GPU
        """
        device = input.device
        n_batch = input.shape[0]
        input_np = input.detach().cpu().numpy().reshape(n_batch, -1, 3) 
        
        energies = np.zeros(n_batch)
        forces = np.zeros_like(input_np)

        for i in range(n_batch):
            openmm_context.setPositions(input_np[i])
            state = openmm_context.getState(getForces=True, getEnergy=True)
            
            energies[i] = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forces[i] = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

        # Convert back to torch once
        energies_torch = torch.from_numpy(energies).to(device).to(input.dtype)
        forces_torch = torch.from_numpy(-forces).to(device).to(input.dtype)
        
        ctx.save_for_backward(forces_torch.view(n_batch, -1))
        return energies_torch

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output.unsqueeze(-1), None, None


class ALDP_OMM(ALDP):
    def __init__(
        self,
        temperature : float = 300.0,
        data_path : str = None,
        regularize_energy: bool = False,
        check_chirality: bool = False,
        energy_cut: float = 1e8,
        energy_max: float = 1e20,
        env: str = 'implicit',
        platform_name: str = 'CUDA'
    ) -> None:
        """
        Boltzmann distribution of Alanine dipeptide (ALDP).
        """
        super(ALDP_OMM, self).__init__(
            temperature=temperature,
            data_path=data_path,
            regularize_energy=regularize_energy,
            check_chirality=check_chirality,
            energy_cut=energy_cut,
            energy_max=energy_max
        )

        # Path for loading samples,
        self.data_path = data_path
        self.temperature = temperature
        self.check_chirality = check_chirality
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.env = env
        self.platform_name = platform_name

        # System setup
        if env == 'vacuum':
            self.system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            self.system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise ValueError(
                f"Environment '{env}' not recognized. Use 'vacuum' or 'implicit'."
            )
        
        platform = mm.Platform.getPlatformByName(platform_name)
        properties = {}
        if platform_name == 'CUDA':
            properties = {'CudaPrecision': 'single'}

        self.sim_context = mm.Context(
            self.system.system,
            mm.LangevinIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                1.0 * unit.femtosecond
            ),
            platform,
            properties
        )
        
        # self.openmm_energy = OpenMMEnergyInterface.apply
        self.openmm_energy = OpenMMInterface.apply

    def _potential_energy(self, coords):
        coords_flattened = coords.view(coords.shape[0], -1)
        return self.openmm_energy(coords_flattened, self.sim_context)