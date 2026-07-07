import torch
import ot as pot
import numpy as np

from .geometry import kabsch_rmsd_matrix_chunked

def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float = None,
) -> torch.Tensor:
    """
    Computes the Gaussian kernel between two sets of samples.
    Arguments:
        x: (num_samples, dim) tensor of samples from the first distribution
        y: (num_samples, dim) tensor of samples from the second distribution
        lengthscale: the lengthscale of the Gaussian kernel
    
    Returns:
        kernel: (num_samples, num_samples) tensor of Gaussian kernel values
    """

    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    # Calculate the squared Euclidean distance between all pairs of points
    x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # (num_samples, 1)
    y_norm_sq = torch.sum(y ** 2, dim=1, keepdim=True)  # (num_samples, 1)
    pairwaise_sq_dist = x_norm_sq + y_norm_sq.t() - 2.0 * torch.matmul(x, y.t())  # (num_samples, num_samples)
    pairwaise_sq_dist = pairwaise_sq_dist.clamp(min=0.0)  # Ensure non-negative distances

    if lengthscale is None:
        lengthscale = x.shape[1]  # Default lengthscale based on dimensionality
    kernel = torch.exp(-pairwaise_sq_dist / lengthscale)  # (num_samples, num_samples)
    return kernel

def maximum_mean_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float = None
) -> torch.Tensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two distributions.

    Arguments:
        x: (num_samples, dim) tensor of samples from the first distribution
        y: (num_samples, dim) tensor of samples from the second distribution
        lengthscale_list: lengthscale for the Gaussian kernel
    
    Returns:
        mmd: torch.Tensor, the MMD value between the two distributions
    """

    K_xx = gaussian_kernel(x, x, lengthscale)  # (num_samples, num_samples)
    K_yy = gaussian_kernel(y, y, lengthscale)  # (num_samples, num_samples)
    K_xy = gaussian_kernel(x, y, lengthscale)  # (num_samples, num_samples)

    mmd = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
    mmd = mmd.clamp(min=0.0).sqrt()  # Ensure non-negative MMD
    return mmd

def wasserstein2_distance(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Wasserstein distance between two distributions using samples from each distribution.

    Arguments:
        x: (num_samples, dim) tensor of samples from the first distribution
        y: (num_samples, dim) tensor of samples from the second distribution
    
    Returns:
        w_dist: torch.Tensor, the Wasserstein distance between the two distributions
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"

    dist_matrix = pot.dist(x.detach().cpu().numpy(), y.detach().cpu().numpy(), metric='euclidean')
    x_ = np.ones(len(x)) / len(x)
    y_ = np.ones(len(y)) / len(y)
    w2_dist = torch.tensor(pot.emd2(x_, y_, dist_matrix, numItermax=100000), device=x.device)
    return w2_dist

def wasserstein2_distance_equivariant(
    X: torch.Tensor,
    Y: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Wasserstein distance between two sets of particles using RMSD as the distance metric.

    Arguments:
        X: (batch_size_X, n_particles, dim) tensor of positions of the first batch of molecules
        Y: (batch_size_Y, n_particles, dim) tensor of positions of the second batch of molecules
    
    Returns:
        w_dist: torch.Tensor, the Wasserstein distance between the two sets of molecules
    
    """

    assert X.shape == Y.shape, "Input tensors must have the same shape"
    assert len(X.shape) == 3, "Input tensors must be 3D"

    dist_matrix = kabsch_rmsd_matrix_chunked(X, Y, chunk_size=256).cpu().numpy()
    x_ = np.ones(len(X)) / len(X)
    y_ = np.ones(len(Y)) / len(Y)
    w2_dist = torch.tensor(pot.emd2(x_, y_, dist_matrix, numItermax=100000), device=X.device)
    # G = pot.emd(x_, y_, dist_matrix)
    # w2_dist = np.sum(G * dist_matrix) / G.sum()
    # w2_dist = torch.tensor(w2_dist, device=X.device)
    return w2_dist

def total_variation(
    x: torch.Tensor,
    y: torch.Tensor,
    num_bins: int = 200
) -> torch.Tensor:
    """
    Computes the total variation distance between two distributions using samples from each distribution.

    Arguments:
        x: (num_samples, dim) tensor of samples from the first distribution
        y: (num_samples, dim) tensor of samples from the second distribution
        num_bins: number of bins to use for histogram computation
    
    Returns:
        tv_dist: torch.Tensor, the total variation distance between the two distributions    
    """

    # assert x.shape == y.shape, "Input tensors must have the same shape"
    assert len(x.shape) == 2, "Input tensors must be 2D"
    
    dim = x.shape[1]
    bins = (num_bins,) * dim
    all_data = torch.cat([x, y], dim=0) # (num_samples * 2, dim)
    min_vals, _ = all_data.min(dim=0) # (dim,)
    max_vals, _ = all_data.max(dim=0) # (dim,)
    ranges = tuple(
        (min_vals[i].item(), max_vals[i].item()) for i in range(dim)
    )
    ranges = tuple(item for subtuple in ranges for item in subtuple) # flatten the tuple
    hist_x, _ = torch.histogramdd(x.cpu(), bins=bins, range=ranges)
    hist_y, _ = torch.histogramdd(y.cpu(), bins=bins, range=ranges)

    hist_x_norm = hist_x / hist_x.sum()
    hist_y_norm = hist_y / hist_y.sum()

    total_var = 0.5 * torch.abs(hist_x_norm - hist_y_norm).sum()
    
    return total_var

def setup_quadratic_function(dim: int, seed: int = 0):
    # Useful for porting this problem to non torch libraries.
    torch.random.manual_seed(seed)
    # example function that we may want to calculate expectations over
    x_shift = 2 * torch.randn(dim)
    A = 2 * torch.rand((dim, dim))
    b = torch.rand(dim)
    torch.seed()  # set back to random number
    return x_shift, A, b
    # if x.dtype == torch.float64:
    #     return x_shift.double(), A.double(), b.double()
    # else:
    #     assert x.dtype == torch.float32
    #     return x_shift, A, b

def setup_equivariant_function(n_particles: int, seed: int = 0):
    torch.random.manual_seed(seed)
    a = torch.randn(1, n_particles)
    b = torch.randn(1, n_particles)
    torch.seed()  # set back to random number
    return a, b

def quadratic_function(x: torch.Tensor, seed: int = 0):
    x_shift, A, b = setup_quadratic_function(x.shape[-1], seed)
    x_shift = x_shift.to(x.device).type(x.dtype)
    A = A.to(x.device).type(x.dtype)
    b = b.to(x.device).type(x.dtype)
    x = x + x_shift
    return torch.einsum("bi,ij,bj->b", x, A, x) + torch.einsum("i,bi->b", b, x)

def equivariant_function(x: torch.Tensor, seed: int = 0):
    n_particles = x.shape[1]
    a, b = setup_equivariant_function(n_particles, seed)
    a = a.to(x.device).type(x.dtype)
    b = b.to(x.device).type(x.dtype)
    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (batch, n_particles, n_particles, dim)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (batch, n_particles, n_particles)
    K_xx = torch.exp(-dist_sq / n_particles)  # (batch, n_particles, n_particles)
    res = K_xx @ a.t() # (batch_size, n_particles, 1)
    res = b @ res # (batch_size, 1, 1)
    return res.squeeze(-1)  # (batch_size,)

def relative_mae(gt_samples, pred_samples):

    true_f = quadratic_function(gt_samples)
    pred_f = quadratic_function(pred_samples)

    true_expectation = torch.mean(true_f)
    est_expectation = torch.mean(pred_f)

    return torch.abs((est_expectation - true_expectation) / true_expectation)

def relative_mae_equivariant(gt_samples, pred_samples):

    true_f = equivariant_function(gt_samples)
    pred_f = equivariant_function(pred_samples)

    true_expectation = torch.mean(true_f)
    est_expectation = torch.mean(pred_f)

    return torch.abs((est_expectation - true_expectation) / true_expectation)

def kl_div_ramachandran(
    phi_gt, psi_gt, phi_pred, psi_pred, num_bins=50, eps=1e-10
) -> torch.Tensor:
    """
    Compute the KL divergence between the Ramachandran plots of ground truth and predicted samples.
    """
    # 1. Stack inputs to shape (N, 2)
    # torch.histogramdd expects a single tensor of shape (N, D)
    samples_gt = torch.stack([phi_gt, psi_gt], dim=1).cpu()
    samples_pred = torch.stack([phi_pred, psi_pred], dim=1).cpu()

    # Define range for torsion angles [-pi, pi]
    # range needs to be a flattened sequence [min_dim1, max_dim1, min_dim2, max_dim2]
    bounds = [-np.pi, np.pi, -np.pi, np.pi]

    # 2. Compute Histograms
    hist_gt, _ = torch.histogramdd(
        samples_gt, bins=num_bins, range=bounds
    )
    hist_pred, _ = torch.histogramdd(
        samples_pred, bins=num_bins, range=bounds
    )

    prob_gt = hist_gt + eps
    prob_gt = prob_gt / prob_gt.sum()

    prob_pred = hist_pred + eps
    prob_pred = prob_pred / prob_pred.sum()

    # 5. Compute KL Divergence: sum(P * log(P / Q))
    kl_div = (prob_gt * (prob_gt.log() - prob_pred.log())).sum()

    return kl_div

def fes_rmse_ramachandran(
    phi_gt, psi_gt, phi_pred, psi_pred, num_bins=36, eps=1e-10, prob_cutoff=1e-4
) -> torch.Tensor:
    """
    Compute the RMSE between the free energy surface of ground truth and predicted samples in Ramachandran space.
    """
    # Define range for torsion angles [-pi, pi]
    bounds = [-np.pi, np.pi, -np.pi, np.pi]

    # Stack inputs to shape (N, 2)
    samples_gt = torch.stack([phi_gt, psi_gt], dim=1).cpu()
    samples_pred = torch.stack([phi_pred, psi_pred], dim=1).cpu()    

    # Compute Histograms
    hist_gt, _ = torch.histogramdd(
        samples_gt, bins=num_bins, range=bounds
    )
    hist_pred, _ = torch.histogramdd(
        samples_pred, bins=num_bins, range=bounds
    )

    prob_gt = hist_gt / hist_gt.sum()
    prob_pred = hist_pred / hist_pred.sum()

    mask = prob_gt > prob_cutoff

    f_gt = - torch.log(prob_gt[mask])
    f_pred = - torch.log(prob_pred[mask] + eps)

    mse = ((f_gt - f_pred) ** 2).mean()
    rmse = torch.sqrt(mse)
    
    return rmse