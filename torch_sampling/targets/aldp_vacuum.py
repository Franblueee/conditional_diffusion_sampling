import torch
import torch.nn as nn
import warnings

from .aldp import ALDP

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from openmmtools import testsystems

warnings.filterwarnings("ignore")

class ALDPVacuumEnergyModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # # Load via OpenMM to get exact parameters
        # prmtop = app.AmberPrmtopFile(prmtop_path)
        # system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
        system = testsystem.system
        
        # --- Bonds ---
        bonds_idx, bonds_k, bonds_length = [], [], []
        bond_forces = [f for f in system.getForces() if isinstance(f, mm.HarmonicBondForce)]
        if bond_forces:
            bf = bond_forces[0]
            for i in range(bf.getNumBonds()):
                p1, p2, length, k = bf.getBondParameters(i)
                bonds_idx.append([p1, p2])
                bonds_k.append(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
                bonds_length.append(length.value_in_unit(unit.nanometer))
        
        self.register_buffer('bonds_idx', torch.tensor(bonds_idx, dtype=torch.long))
        self.register_buffer('bonds_k', torch.tensor(bonds_k, dtype=torch.float32))
        self.register_buffer('bonds_length', torch.tensor(bonds_length, dtype=torch.float32))

        # --- Angles ---
        angles_idx, angles_k, angles_theta = [], [], []
        angle_forces = [f for f in system.getForces() if isinstance(f, mm.HarmonicAngleForce)]
        if angle_forces:
            af = angle_forces[0]
            for i in range(af.getNumAngles()):
                p1, p2, p3, angle, k = af.getAngleParameters(i)
                angles_idx.append([p1, p2, p3])
                angles_k.append(k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))
                angles_theta.append(angle.value_in_unit(unit.radian))

        self.register_buffer('angles_idx', torch.tensor(angles_idx, dtype=torch.long))
        self.register_buffer('angles_k', torch.tensor(angles_k, dtype=torch.float32))
        self.register_buffer('angles_theta', torch.tensor(angles_theta, dtype=torch.float32))

        # --- Torsions ---
        torsions_idx, torsions_k, torsions_phase, torsions_period = [], [], [], []
        tor_forces = [f for f in system.getForces() if isinstance(f, mm.PeriodicTorsionForce)]
        if tor_forces:
            tf = tor_forces[0]
            for i in range(tf.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = tf.getTorsionParameters(i)
                torsions_idx.append([p1, p2, p3, p4])
                torsions_k.append(k.value_in_unit(unit.kilojoule_per_mole))
                torsions_phase.append(phase.value_in_unit(unit.radian))
                torsions_period.append(periodicity)
        
        self.register_buffer('torsions_idx', torch.tensor(torsions_idx, dtype=torch.long))
        self.register_buffer('torsions_k', torch.tensor(torsions_k, dtype=torch.float32))
        self.register_buffer('torsions_phase', torch.tensor(torsions_phase, dtype=torch.float32))
        self.register_buffer('torsions_period', torch.tensor(torsions_period, dtype=torch.float32))

        # --- Non-Bonded ---
        nb_forces = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
        if nb_forces:
            nf = nb_forces[0]
            n_particles = nf.getNumParticles()
            charges, sigmas, epsilons = [], [], []
            for i in range(n_particles):
                q, sig, eps = nf.getParticleParameters(i)
                charges.append(q.value_in_unit(unit.elementary_charge))
                sigmas.append(sig.value_in_unit(unit.nanometer))
                epsilons.append(eps.value_in_unit(unit.kilojoule_per_mole))
            
            self.register_buffer('charges', torch.tensor(charges, dtype=torch.float32))
            self.register_buffer('sigmas', torch.tensor(sigmas, dtype=torch.float32))
            self.register_buffer('epsilons', torch.tensor(epsilons, dtype=torch.float32))
            
            mask_matrix = torch.ones((n_particles, n_particles))
            mask_matrix.fill_diagonal_(0)
            
            exc_idx, exc_q, exc_sig, exc_eps = [], [], [], []
            for i in range(nf.getNumExceptions()):
                p1, p2, q_prod, sig, eps = nf.getExceptionParameters(i)
                mask_matrix[p1, p2] = 0
                mask_matrix[p2, p1] = 0
                
                epsv = eps.value_in_unit(unit.kilojoule_per_mole)
                qv = q_prod.value_in_unit(unit.elementary_charge**2)
                if abs(epsv) > 1e-9 or abs(qv) > 1e-9:
                    exc_idx.append([p1, p2])
                    exc_q.append(qv)
                    exc_sig.append(sig.value_in_unit(unit.nanometer))
                    exc_eps.append(epsv)

            self.register_buffer('exclusion_mask', mask_matrix)
            self.register_buffer('exc_idx', torch.tensor(exc_idx, dtype=torch.long) if exc_idx else torch.empty((0,2), dtype=torch.long))
            self.register_buffer('exc_q', torch.tensor(exc_q, dtype=torch.float32))
            self.register_buffer('exc_sig', torch.tensor(exc_sig, dtype=torch.float32))
            self.register_buffer('exc_eps', torch.tensor(exc_eps, dtype=torch.float32))
        
        self.coulomb_const = 138.935456

    def forward(self, coords):
        if coords.ndim == 2: coords = coords.unsqueeze(0)
        
        # Bonds
        b1, b2 = coords[:, self.bonds_idx[:, 0]], coords[:, self.bonds_idx[:, 1]]
        e_bond = torch.sum(0.5 * self.bonds_k * (torch.norm(b1-b2, dim=2) - self.bonds_length)**2, dim=1)

        # Angles
        a1, a2, a3 = coords[:, self.angles_idx[:,0]], coords[:, self.angles_idx[:,1]], coords[:, self.angles_idx[:,2]]
        v1, v2 = a1-a2, a3-a2
        v1n = v1 / (torch.norm(v1, dim=2, keepdim=True) + 1e-12)
        v2n = v2 / (torch.norm(v2, dim=2, keepdim=True) + 1e-12)
        theta = torch.acos(torch.clamp(torch.sum(v1n*v2n, dim=2), -0.999999, 0.999999))
        e_angle = torch.sum(0.5 * self.angles_k * (theta - self.angles_theta)**2, dim=1)

        # Torsions
        t1, t2, t3, t4 = coords[:, self.torsions_idx[:,0]], coords[:, self.torsions_idx[:,1]], coords[:, self.torsions_idx[:,2]], coords[:, self.torsions_idx[:,3]]
        b0, b1, b2 = t2-t1, t3-t2, t4-t3
        
        v = b1 / (torch.norm(b1, dim=2, keepdim=True) + 1e-12)
        w = torch.linalg.cross(b0, b1, dim=2)
        x = torch.linalg.cross(b1, b2, dim=2)
        x_u = x / (torch.norm(x, dim=2, keepdim=True) + 1e-12)
        y = torch.linalg.cross(v, x_u, dim=2)
        u = w / (torch.norm(w, dim=2, keepdim=True) + 1e-12)
        
        phi = torch.atan2(torch.sum(u*y, dim=2), torch.sum(u*x_u, dim=2))
        e_tor = torch.sum(self.torsions_k * (1 + torch.cos(self.torsions_period * phi - self.torsions_phase)), dim=1)

        # Non-Bonded
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        r = torch.norm(diff, dim=3)
        diag_mask = torch.eye(r.shape[1], device=r.device, dtype=torch.bool)
        r_safe = r.clone(); r_safe[:, diag_mask] = 1.0
        inv_r = 1.0 / r_safe
        
        sig_ij = 0.5 * (self.sigmas.unsqueeze(0) + self.sigmas.unsqueeze(1))
        eps_ij = torch.sqrt(self.epsilons.unsqueeze(0) * self.epsilons.unsqueeze(1))
        
        term6 = (sig_ij * inv_r)**6
        term12 = term6**2
        e_lj = 4 * eps_ij * (term12 - term6)
        e_coul = self.coulomb_const * (self.charges.unsqueeze(0) * self.charges.unsqueeze(1)) * inv_r
        
        mask = self.exclusion_mask.unsqueeze(0) * torch.triu(torch.ones_like(self.exclusion_mask), diagonal=1)
        e_nb = torch.sum((e_lj + e_coul) * mask, dim=(1,2))

        # Exceptions
        e_exc = 0.0
        if self.exc_idx.shape[0] > 0:
            p1, p2 = coords[:, self.exc_idx[:,0]], coords[:, self.exc_idx[:,1]]
            r_exc = torch.norm(p1-p2, dim=2)
            term6_x = (self.exc_sig / r_exc)**6
            e_lj_x = torch.sum(4 * self.exc_eps * (term6_x**2 - term6_x), dim=1)
            e_coul_x = torch.sum(self.coulomb_const * self.exc_q / r_exc, dim=1)
            e_exc = e_lj_x + e_coul_x

        return e_bond + e_angle + e_tor + e_nb + e_exc

class ALDPVacuum(ALDP):
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
        Alanine dipeptide (ALDP) in vacuum using Amber force field.        
        """
        super(ALDPVacuum, self).__init__(
            temperature=temperature,
            data_path=data_path,
            regularize_energy=regularize_energy,
            check_chirality=check_chirality,
            energy_cut=energy_cut,
            energy_max=energy_max
        )
        
        self.energy_module = ALDPVacuumEnergyModule()

    def _potential_energy(self, coords):
        """
        Arguments:
            coords: (batch_size, n_atoms, 3) coordinates
        
        Returns:
            energy: (batch_size,) potential energies
        """
        return self.energy_module(coords) # (batch_size,)