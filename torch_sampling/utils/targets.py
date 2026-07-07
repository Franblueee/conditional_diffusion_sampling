import torch
import numpy as np

from sampling.targets import GaussianMixture

def random_2D_gaussian_mixture(n_modes: int, std: float, scale: float = 10.0, seed: float = 0.0) -> GaussianMixture:
    """
    """
    torch.manual_seed(seed)
    means = (torch.rand(n_modes, 2) - 0.5) * scale
    covs = torch.diag_embed(torch.ones(n_modes, 2)) * std ** 2
    weights = torch.rand(n_modes)
    weights = weights / weights.sum()
    return GaussianMixture(means, covs, weights)

def symmetric_2D_gaussian_mixture(n_modes: int, std: float, scale: float = 10.0) -> GaussianMixture:
    """
    """
    angles = torch.linspace(0, 2 * np.pi, n_modes + 1)[:n_modes]
    means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
    covs = torch.diag_embed(torch.ones(n_modes, 2) * std ** 2)
    weights = torch.ones(n_modes) / n_modes
    return GaussianMixture(means, covs, weights)

def star_2D_gaussian_mixture(scale: float = 10.0) -> GaussianMixture:
    """
    """
    means = torch.tensor([[0.0, 0.0], [scale, 0.0], [-scale, 0.0], [0.0, scale], [0.0, -scale]])
    covs = torch.stack([torch.eye(2) * 0.5] * means.shape[0])
    weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    return GaussianMixture(means, covs, weights)

def uniform_gaussian_mixture(
    scale: float = 40.0, 
    dim: int = 2,
    log_var_scaling: float = 1.0,
    n_mixes: int = 40,
    seed: int = 0
) -> GaussianMixture:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mean = (torch.rand((n_mixes, dim), generator=generator) - 0.5) * 2 * scale
    log_var = torch.ones((n_mixes, dim)) * log_var_scaling
    covs = torch.diag_embed(torch.nn.functional.softplus(log_var))
    weights = torch.ones((n_mixes,))
    return GaussianMixture(mean, covs, weights)

def nonuniform_gaussian_mixture(
    scale: float = 40.0, 
    dim: int = 2,
    log_var_scaling: float = 1.0,
    n_mixes: int = 40,
    seed: int = 0,
    perc_big_mixes: float = 0.2,
    weight_big_mixes: float = 100.0
) -> GaussianMixture:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mean = (torch.rand((n_mixes, dim), generator=generator) - 0.5) * 2 * scale
    log_var = torch.ones((n_mixes, dim)) * log_var_scaling
    covs = torch.diag_embed(torch.nn.functional.softplus(log_var))
    weights = torch.ones((n_mixes,))
    n_big_mixes = int(n_mixes * perc_big_mixes)
    weights[:n_big_mixes] *= weight_big_mixes
    return GaussianMixture(mean, covs, weights)