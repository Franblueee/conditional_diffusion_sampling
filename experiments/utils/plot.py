import os
import torch
import wandb
import numpy as np

from matplotlib import pyplot as plt

from torch_sampling.targets import GaussianMixture, ALDP_OMM, ALDPVacuum

from omegaconf import DictConfig

import mdtraj
from openmmtools.testsystems import AlanineDipeptideVacuum, AlanineDipeptideImplicit

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def imshow_density(
    log_prob,
    bins: int,
    scale: float,
    ax=None,
    device: torch.device = torch.device('cpu'),
    **kwargs
):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = log_prob(xy).reshape(bins, bins).T
    # print("Density:", density.min().item(), density.max().item())
    im = ax.imshow(
        density.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)

def contour_density(
    log_prob,
    bins: int,
    scale: float,
    ax=None,
    device: torch.device = torch.device('cpu'),
    **kwargs
):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = log_prob(xy).reshape(bins, bins).T
    im = ax.contour(
        density.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)


def scatter_samples(
    x,
    scale: float, 
    ax=None, 
    **kwargs
):
    x = torch.clamp(x, -scale*1.5, scale*1.5)
    im = ax.scatter(x[:, 0].cpu(), x[:, 1].cpu(), **kwargs)

def contourf_density(
    log_prob,
    bins: int,
    scale: float,
    ax=None,
    device: torch.device = torch.device('cpu'),
    **kwargs
):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = log_prob(xy).reshape(bins, bins).T
    im = ax.contourf(
        density.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)

def plot_density_and_samples(
    target_log_prob, 
    samples: torch.Tensor,
    bins: int = 200,
    scale: float = 10.0,
    contour_levels: int = 20,
    ax = None,
    device: torch.device = torch.device('cpu')
):
    if ax is None:
        ax = plt.gca()
    
    scale_plot = scale * 1.2

    im = imshow_density(
        target_log_prob, 
        bins, 
        scale_plot, 
        ax=ax, 
        vmin=-scale,
        cmap="Blues",
        device=device,
        zorder=1,
        alpha=0.5
    )

    im = contour_density(
        target_log_prob, 
        bins, 
        scale_plot, 
        ax=ax, 
        colors='grey',
        linestyles='solid',
        levels=contour_levels,
        device=device,
        zorder=1,
        alpha=0.25
    )

    im = scatter_samples(samples, scale, ax=ax, color='black', alpha=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-scale_plot, scale_plot)
    ax.set_ylim(-scale_plot, scale_plot)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_contour_and_samples(
    target_log_prob, 
    samples: torch.Tensor,
    bins: int = 100,
    scale: float = 10.0,
    contour_levels: int = 10,
    ax = None,
    device: torch.device = torch.device('cpu')
):
    if ax is None:
        ax = plt.gca()

    im = contour_density(
        target_log_prob, 
        bins, 
        scale, 
        ax=ax, 
        cmap=LinearSegmentedColormap.from_list("", ["navy", "aquamarine"]),
        levels=contour_levels,
        device=device,
        zorder=1
    )

    im = scatter_samples(samples, scale, ax=ax, color='black', alpha=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_results(
    target, 
    gt_samples: torch.Tensor, 
    pred_samples: torch.Tensor, 
    output_dir: str,
    config : DictConfig,
    device: torch.device = torch.device('cpu')
):
    if isinstance(target, GaussianMixture) and target.dim == 2:
        plot_gmm_results(
            target,
            gt_samples.detach(),
            pred_samples.detach(),
            output_dir=output_dir,
            device=device,
            bins=config.task.plot.bins if hasattr(config.task.plot, 'bins') else 100,
            scale=config.task.scale if hasattr(config.task, 'scale') else 40.0,
            contour_levels=config.task.plot.contour_levels if hasattr(config.task.plot, 'contour_levels') else 20
        )
    elif isinstance(target, (ALDP_OMM, ALDPVacuum)):
        env = "vacuum"
        if isinstance(target, ALDP_OMM):
            if target.env == 'implicit':
                env = "implicit"
        plot_aldp_results(
            gt_samples.detach(),
            pred_samples.detach(),
            output_dir=output_dir,
            bins=config.task.plot.bins if hasattr(config.task.plot, 'bins') else 100,
            vmin=config.task.plot.vmin if hasattr(config.task.plot, 'vmin') else 0.001,
            env=env
        )

def plot_smc_diagnostics(
    schedules_list: list,
    Lambda_list: list,
    sqrt_D_list: list,
    output_dir: str,
    name = "smc_diagnostics"
):
    fig, ax = plt.subplots(len(Lambda_list), 2, figsize=(6, 3 * len(Lambda_list)))
    if len(Lambda_list) == 1:
        ax = ax.reshape(1, -1)

    for i in range(len(Lambda_list)):
        schedule = schedules_list[i].detach().cpu().numpy()
        Lambda = Lambda_list[i].detach().cpu().numpy()
        sqrt_D = sqrt_D_list[i].detach().cpu().numpy()
        ax[i, 0].plot(schedule, Lambda, marker='o')
        ax[i, 0].set_xlabel(r"$\beta$")
        ax[i, 0].set_ylabel(r"$\Lambda$")
        ax[i, 0].set_title(f"Round {i+1}")
        ax[i, 0].grid()

        ax[i, 1].plot(schedule, sqrt_D, marker='o')
        ax[i, 1].set_xlabel(r"$\beta$")
        ax[i, 1].set_ylabel(r"$\sqrt{D}$")
        ax[i, 1].set_title(f"Round {i+1}")
        ax[i, 1].grid()

    plt.tight_layout()

    if wandb.run is not None:
        wandb.log({f"{name}": wandb.Image(fig)})
        print(f"Logged {name} plot to WandB")
    else:
        fig.savefig(os.path.join(output_dir, f'{name}.png'))
        print(f"Saved {name} plot to {os.path.join(output_dir, f'{name}.png')}")
    plt.close(fig)

def plot_pt_diagnostics(
    schedules_list: list,
    Lambda_list: list,
    rejection_rates_list: list,
    output_dir: str,
    name = "pt_diagnostics"
):
    fig, ax = plt.subplots(len(Lambda_list), 2, figsize=(6, 3 * len(Lambda_list)))
    if len(Lambda_list) == 1:
        ax = ax.reshape(1, -1)
    for i in range(len(Lambda_list)):
        schedule = schedules_list[i].detach().cpu().numpy()
        Lambda = Lambda_list[i].detach().cpu().numpy()
        rejection_rates = rejection_rates_list[i].detach().cpu().numpy()
        ax[i, 0].plot(schedule, Lambda, marker='o')
        ax[i, 0].set_xlabel(r"$\beta$")
        ax[i, 0].set_ylabel(r"$\Lambda$")
        ax[i, 0].set_title(f"Round {i+1}")
        ax[i, 0].grid()

        ax[i, 1].plot(rejection_rates, marker='o')
        ax[i, 1].set_ylabel(r"$r$")
        ax[i, 1].set_xticks(np.arange(len(rejection_rates)))
        ax[i, 1].set_xticklabels([fr"${i+1}\leftrightarrow{i+2}$" for i in range(len(rejection_rates))])
        ax[i, 1].set_title(f"Round {i+1}")
        ax[i, 1].set_ylim(0, 1.05)
        ax[i, 1].grid()

    plt.tight_layout()

    if wandb.run is not None:
        wandb.log({f"{name}": wandb.Image(fig)})
        print(f"Logged {name} plot to WandB")
    else:
        fig.savefig(os.path.join(output_dir, f'{name}.png'))
        print(f"Saved {name} plot to {os.path.join(output_dir, f'{name}.png')}")
    plt.close(fig)
    
def plot_gmm_results(
    target, 
    gt_samples: torch.Tensor, 
    pred_samples: torch.Tensor, 
    output_dir: str,
    bins: int = 100,
    scale: float = 40.0,
    contour_levels: int = 20,
    device: torch.device = torch.device('cpu')
):
    samples_list = [("Ground truth samples", "gt_samples", gt_samples), ("Predicted samples", "pred_samples", pred_samples)]

    for show_name, short_name, samples in samples_list:

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_density_and_samples(
            target.log_prob,
            samples, 
            device=device,
            bins=bins,
            scale=scale,
            contour_levels=contour_levels,
        )
        fig.suptitle(f"{show_name}", fontsize=16)
        plt.tight_layout()

        if wandb.run is not None:
            wandb.log({f"{short_name}": wandb.Image(fig)})
            print(f"Logged {short_name} plot to WandB")
        else:
            fig.savefig(os.path.join(output_dir, f'{short_name}.png'))
            print(f"Saved {short_name} plot to {os.path.join(output_dir, f'{short_name}.png')}")
        plt.close(fig)

def plot_aldp_results(
    gt_samples: torch.Tensor, 
    pred_samples: torch.Tensor, 
    output_dir: str,
    bins: int = 100,
    vmin: float = 0.001,
    env: str = "vacuum"
):
    if env == "implicit":
        aldp_system = AlanineDipeptideImplicit(constraints=None)
    else:
        aldp_system = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp_system.topology)

    gt_samples = gt_samples.cpu().numpy().reshape(-1, 22, 3)
    traj_gt = mdtraj.Trajectory(gt_samples, topology)
    phi_gt = mdtraj.compute_phi(traj_gt)[1].reshape(-1)
    psi_gt = mdtraj.compute_psi(traj_gt)[1].reshape(-1)
    not_nan_idx = ~np.isnan(phi_gt) & ~np.isnan(psi_gt)
    phi_gt = phi_gt[not_nan_idx]
    psi_gt = psi_gt[not_nan_idx]

    pred_samples = pred_samples.cpu().numpy().reshape(-1, 22, 3)
    traj_pred = mdtraj.Trajectory(pred_samples, topology)
    phi_pred = mdtraj.compute_phi(traj_pred)[1].reshape(-1)
    psi_pred = mdtraj.compute_psi(traj_pred)[1].reshape(-1)
    not_nan_idx = ~np.isnan(phi_pred) & ~np.isnan(psi_pred)
    phi_pred = phi_pred[not_nan_idx]
    psi_pred = psi_pred[not_nan_idx]
    
    samples_list = [("Ground truth samples", phi_gt, psi_gt), ("Predicted samples", phi_pred, psi_pred)]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(2):
        show_name, phi, psi = samples_list[i]

        ax[i].hist2d(
            phi, 
            psi, 
            bins=bins, 
            range=[[-np.pi, np.pi], [-np.pi, np.pi]], 
            density=True, 
            cmap=LinearSegmentedColormap.from_list("", ["navy", "aquamarine"]), 
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=1.0)
        )
        ax[i].set_xlabel(r"$\phi$")
        ax[i].set_ylabel(r"$\psi$")
        ax[i].set_title(show_name)
        ax[i].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax[i].set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        ax[i].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax[i].set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])

    plt.suptitle('Ramachandran Plots', fontsize=16)
    plt.tight_layout()

    if wandb.run is not None:
        wandb.log({f"ramachandran_plots": wandb.Image(fig)})
        print(f"Logged ramachandran_plots plot to WandB")
    else:
        fig.savefig(os.path.join(output_dir, f'ramachandran_plots.png'))
        print(f"Saved ramachandran_plots plot to {os.path.join(output_dir, f'ramachandran_plots.png')}")
    plt.close(fig)