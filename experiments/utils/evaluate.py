import torch
import mdtraj
import numpy as np

from torch_sampling.targets import TargetDistribution, ALDPVacuum, ALDP_OMM, MLPPosterior
from omegaconf import DictConfig, OmegaConf

from openmmtools.testsystems import AlanineDipeptideVacuum, AlanineDipeptideImplicit

from torch_sampling.utils.metrics import (
    maximum_mean_discrepancy, 
    wasserstein2_distance, 
    wasserstein2_distance_equivariant,
    total_variation,
    relative_mae,
    relative_mae_equivariant,
    kl_div_ramachandran,
    fes_rmse_ramachandran
)

N_SAMPLES_LIMIT = 20000

def evaluate_samples(
    gt_samples: torch.Tensor,
    pred_samples: torch.Tensor,
    target: TargetDistribution,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Evaluate the predicted samples against ground truth samples.
    
    Arguments:
        gt_samples: Ground truth samples (shape: [n_samples, dim]).
        pred_samples: Predicted samples (shape: [n_samples, dim]).
        
    Returns:
        A dictionary containing evaluation metrics.
    """

    print(f"Evaluating with {gt_samples.shape[0]} ground truth samples and {pred_samples.shape[0]} predicted samples.")

    n_samples = max(gt_samples.shape[0], pred_samples.shape[0])

    if n_samples > N_SAMPLES_LIMIT:
        device = torch.device("cpu")

    gt_samples = gt_samples.to(device)
    pred_samples = pred_samples.to(device)
    target = target.to(device)

    metrics = {}

    if isinstance(target, MLPPosterior):
        test_nll = - target.log_likelihood(target.X_test, target.y_test, pred_samples).mean() / target.X_test.shape[0]
        metrics['test_nll'] = test_nll.item()
        return metrics
        
    gt_energies = -target.log_prob(gt_samples)
    pred_energies = -target.log_prob(pred_samples)

    try:
        tvd_energy = total_variation(gt_energies, pred_energies)
    except Exception as e:
        print(f"Error computing TVD: {e}")
        tvd_energy = torch.tensor(float('nan'), device=device)
    metrics['tvd_energy'] = tvd_energy.item()

    try:
        rel_mae = relative_mae(gt_samples, pred_samples)
    except Exception as e:
        print(f"Error computing Relative MAE: {e}")
        rel_mae = torch.tensor(float('nan'), device=device)
    metrics['rel_mae'] = rel_mae.item()    

    if n_samples < N_SAMPLES_LIMIT:

        try:
            mmd_energy = maximum_mean_discrepancy(gt_energies, pred_energies)
        except Exception as e:
            print(f"Error computing MMD: {e}")
            mmd_energy = torch.tensor(float('nan'), device=device)
        
        try:
            w2_energy = wasserstein2_distance(gt_energies, pred_energies)
        except Exception as e:
            print(f"Error computing W2 (energy): {e}")
            w2_energy = torch.tensor(float('nan'), device=device)
        
        try:
            w2_data = wasserstein2_distance(gt_samples, pred_samples)
        except Exception as e:
            print(f"Error computing W2 (data): {e}")
            w2_data = torch.tensor(float('nan'), device=device)

        metrics['mmd_energy'] = mmd_energy.item()
        metrics['w2_energy'] = w2_energy.item()
        metrics['w2_data'] = w2_data.item()
    
    # if target has n_particles as an attribute
    if hasattr(target, 'n_particles'):
        
        n_particles = target.n_particles
        dim = target.dim

        gt_samples = gt_samples.view(-1, n_particles, dim // n_particles)
        pred_samples = pred_samples.view(-1, n_particles, dim // n_particles)

        if n_samples < N_SAMPLES_LIMIT:
            try:
                w2_data_equivariant = wasserstein2_distance_equivariant(
                    gt_samples, pred_samples
                )
                metrics['w2_data_equivariant'] = w2_data_equivariant.item()
            except Exception as e:
                print(f"Error computing W2 (particles): {e}")
                metrics['w2_data_equivariant'] = float('nan')
        
        try:
            rel_mae_equivariant = relative_mae_equivariant(
                gt_samples, pred_samples
            )
            metrics['rel_mae_equivariant'] = rel_mae_equivariant.item()
        except Exception as e:
            print(f"Error computing Relative MAE (particles): {e}")
            metrics['rel_mae_equivariant'] = float('nan')
    
    if isinstance(target, (ALDPVacuum, ALDP_OMM)):

        n_particles = target.n_particles
        dim = target.dim

        env = "vacuum"
        if isinstance(target, ALDP_OMM):
            if target.env == "implicit":
                env = "implicit"

        if env == "vacuum":
            aldp_system = AlanineDipeptideVacuum(constraints=None)
        elif env == "implicit":
            aldp_system = AlanineDipeptideImplicit(constraints=None)
        
        topology = mdtraj.Topology.from_openmm(aldp_system.topology)

        gt_samples_np = gt_samples.view(-1, n_particles, dim // n_particles).cpu().numpy()
        pred_samples_np = pred_samples.view(-1, n_particles, dim // n_particles).cpu().numpy()

        traj_gt = mdtraj.Trajectory(gt_samples_np, topology)
        phi_gt = mdtraj.compute_phi(traj_gt)[1].reshape(-1)
        psi_gt = mdtraj.compute_psi(traj_gt)[1].reshape(-1)
        not_nan_idx = ~np.isnan(phi_gt) & ~np.isnan(psi_gt)
        phi_gt = phi_gt[not_nan_idx]
        psi_gt = psi_gt[not_nan_idx]

        traj_pred = mdtraj.Trajectory(pred_samples_np, topology)
        phi_pred = mdtraj.compute_phi(traj_pred)[1].reshape(-1)
        psi_pred = mdtraj.compute_psi(traj_pred)[1].reshape(-1)
        not_nan_idx = ~np.isnan(phi_pred) & ~np.isnan(psi_pred)
        phi_pred = phi_pred[not_nan_idx]
        psi_pred = psi_pred[not_nan_idx]

        for num_bins in [36, 50, 100]:
            try:
                kl_div_ram = kl_div_ramachandran(
                    phi_gt=torch.tensor(phi_gt, device=device),
                    psi_gt=torch.tensor(psi_gt, device=device),
                    phi_pred=torch.tensor(phi_pred, device=device),
                    psi_pred=torch.tensor(psi_pred, device=device),
                    num_bins=num_bins,
                    eps=1e-10
                )
            except Exception as e:
                print(f"Error computing KL divergence (Ramachandran): {e}")
                kl_div_ram = torch.tensor(float('nan'), device=device)
            metrics[f'kl_div_ramachandran_{num_bins}'] = kl_div_ram.item()

            try:
                fes_rmse_ram = fes_rmse_ramachandran(
                    phi_gt=torch.tensor(phi_gt, device=device),
                    psi_gt=torch.tensor(psi_gt, device=device),
                    phi_pred=torch.tensor(phi_pred, device=device),
                    psi_pred=torch.tensor(psi_pred, device=device),
                    num_bins=num_bins,
                    eps=1e-10
                )
            except Exception as e:
                print(f"Error computing FES RMSE (Ramachandran): {e}")
                fes_rmse_ram = torch.tensor(float('nan'), device=device)
            metrics[f'fes_rmse_ramachandran_{num_bins}'] = fes_rmse_ram.item()

    return metrics