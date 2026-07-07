import torch

def kabsch(x, y):
    """
    Computes the optimal rotation and translation to align two sets of points (x -> y).

    Arguments:
        x: A tensor of shape (n_particles, dim) representing N points in 3D space.
        y: A tensor of shape (n_particles, dim) representing N points in 3D space.
    
    Returns:
        R: A tensor of shape (dim, dim) representing the optimal rotation matrix.
        t: A tensor of shape (dim,) representing the optimal translation vector.
    """

    n_points, dim = x.shape
    assert x.shape == y.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_x = torch.mean(x, dim=0, keepdims=True) # (1, 3)
    centroid_y = torch.mean(y, dim=0, keepdims=True) # (1, 3)

    # Center the points
    p = x - centroid_x # (n_particles, dim)
    q = y - centroid_y # (n_particles, dim)

    # Compute the covariance matrix
    H = torch.matmul(p.T, q) # (dim, dim)

    # SVD
    U, S, Vt = torch.linalg.svd(H) # U: (dim, dim), S: (dim,), Vt: (dim, dim)

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.T, U.T)) # scalar
    if d < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.T, U.T) # (dim, dim)

    # Optimal translation
    t = centroid_y.squeeze(0) - torch.matmul(centroid_x, R.T).squeeze(0) # (dim,)

    return R, t

def kabsch_rmsd(x, y):
    """
    Computes the optimal rotation and translation to align two sets of points (x -> y), and their RMSD.

    Arguments:
        x: A tensor of shape (n_particles, dim) representing N points in 3D space.
        y: A tensor of shape (n_particles, dim) representing N points in 3D space.
    
    Returns:
        R: A tensor of shape (dim, dim) representing the optimal rotation matrix.
        t: A tensor of shape (dim,) representing the optimal translation vector.
        rmsd: A scalar tensor representing the RMSD between the aligned point sets.
    """

    R, t = kabsch(x, y)
    x0 = torch.matmul(x, R.T) + t
    dist = torch.linalg.norm(x0 - y, dim=-1).mean()
    return R, t, dist

def kabsch_rmsd_matrix(x, y):
    """
    Computes the Kabsch RMSD distance matrix between two sets of point clouds in a batched manner.

    This function calculates the optimal rotation and translation for each pair of point 
    clouds from the input batches (one from x, one from y) and then computes their RMSD.

    Arguments:
        x: A tensor of shape (batch_x, n_particles, dim) representing the first set of point clouds.
        y: A tensor of shape (batch_y, n_particles, dim) representing the second set of point clouds.
    
    Returns:
        A tensor of shape (batch_x, batch_y) representing the RMSD distance matrix.
    """
    # Ensure point clouds have the same number of particles and dimensions
    assert x.shape[1:] == y.shape[1:], "Point clouds must have the same n_particles and dim"
    n_points, dim = x.shape[1:]

    # --- Step 1: Reshape for Broadcasting and Center Point Clouds ---
    # Add dimensions to x and y to allow broadcasting over all pairs.
    # x reshapes to (batch_x, 1, n_points, dim)
    # y reshapes to (1, batch_y, n_points, dim)
    x_b = x.unsqueeze(1)
    y_b = y.unsqueeze(0)

    # Compute centroids for each point cloud in each batch.
    # centroid_x shape: (batch_x, 1, 1, dim)
    # centroid_y shape: (1, batch_y, 1, dim)
    centroid_x = torch.mean(x_b, dim=-2, keepdim=True)
    centroid_y = torch.mean(y_b, dim=-2, keepdim=True)

    # Center the point clouds by subtracting their respective centroids.
    p = x_b - centroid_x
    q = y_b - centroid_y

    # --- Step 2: Batched SVD on the Covariance Matrix ---
    # Compute the covariance matrix H for all pairs.
    # The matmul broadcasts p and q to (batch_x, batch_y, n_points, dim)
    # and performs the multiplication, resulting in H of shape (batch_x, batch_y, dim, dim).
    H = torch.matmul(p.transpose(-2, -1), q)

    # Perform Singular Value Decomposition (SVD) on the batched covariance matrices.
    # U and Vt will have shape (batch_x, batch_y, dim, dim).
    U, S, Vt = torch.linalg.svd(H)

    # --- Step 3: Compute Optimal Rotation and Translation ---
    # Correct for reflections to ensure a right-handed coordinate system.
    det = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
    # sign_d = torch.sign(det).unsqueeze(-1).unsqueeze(-1) # Shape: (batch_x, batch_y, 1, 1)
    sign_d = torch.sign(det).unsqueeze(-1) # Shape: (batch_x, batch_y, 1)
    
    # Flip the last row of Vt if the determinant is negative
    Vt_corrected = Vt.clone()
    Vt_corrected[..., -1, :] *= sign_d

    # Compute the optimal rotation matrix for each pair.
    # R shape: (batch_x, batch_y, dim, dim)
    R = torch.matmul(Vt_corrected.transpose(-2, -1), U.transpose(-2, -1))

    # Compute the optimal translation vector for each pair.
    # t shape: (batch_x, batch_y, 1, dim)
    rotated_centroid_x = torch.matmul(centroid_x, R.transpose(-2, -1))
    t = centroid_y - rotated_centroid_x

    # --- Step 4: Apply Transformation and Compute RMSD ---
    # Align the point clouds from batch x using the calculated rotations and translations.
    # x_aligned shape: (batch_x, batch_y, n_points, dim)
    x_aligned = torch.matmul(x_b, R.transpose(-2, -1)) + t
    
    # Calculate the difference between aligned clouds and target clouds.
    # diff shape: (batch_x, batch_y, n_points, dim)
    diff = x_aligned - y_b

    # Compute the norm along the dimension axis, then the mean over the particle axis.
    # This results in the final distance matrix.
    # dist_matrix shape: (batch_x, batch_y)
    dist_matrix = torch.linalg.norm(diff, dim=-1).mean(dim=-1)

    return dist_matrix

def kabsch_rmsd_matrix_chunked(x, y, chunk_size=128):
    """
    Computes the Kabsch RMSD distance matrix in chunks to conserve memory.

    This is a wrapper around `kabsch_rmsd_matrix` that processes the `x` batch
    in smaller chunks to prevent out-of-memory errors with large batch sizes.

    Arguments:
        x: A tensor of shape (batch_x, n_particles, dim).
        y: A tensor of shape (batch_y, n_particles, dim).
        chunk_size: The number of point clouds from x to process at a time.
                    Lower this value if you are still running out of memory.
    
    Returns:
        A tensor of shape (batch_x, batch_y) representing the RMSD distance matrix.
    """
    batch_size_x = x.shape[0]
    results = []    
    for i in range(0, batch_size_x, chunk_size):
        # Select a chunk from the x batch
        x_chunk = x[i:i + chunk_size]
        
        # Compute the distance matrix for the chunk against the entire y batch
        # This computes a (chunk_size, batch_y) matrix, which is much smaller.
        dist_chunk = kabsch_rmsd_matrix(x_chunk, y)
        
        results.append(dist_chunk)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate the results from all chunks into the final matrix
    return torch.cat(results, dim=0)

def fix_chirality(coords, target_sign='negative'):
    """
    Detects D-isoforms and reflects them to become L-isoforms.
    
    Args:
        coords: (Batch, N_atoms, 3) Tensor
        target_sign: 'negative' or 'positive'. 
                     For standard PDB atom ordering of L-Alanine, 
                     the chiral volume is usually negative.
    
    Returns:
        fixed_coords: (Batch, N_atoms, 3) with consistent chirality
        num_flipped: int, number of samples that were corrected
    """
    # Create a copy to avoid modifying input in-place
    new_coords = coords.clone()
    
    # --- 1. Define Indices for Chiral Center (ACE-ALA-NME) ---
    # These indices correspond to the 'alanine-dipeptide.prmtop'
    # N=6, CA=8, CB=10, C=14
    idx_n  = 6
    idx_ca = 8
    idx_cb = 10
    idx_c  = 14
    
    # --- 2. Calculate Signed Volume ---
    # Center everything on CA for calculation
    ca = new_coords[:, idx_ca]
    n  = new_coords[:, idx_n]
    c  = new_coords[:, idx_c]
    cb = new_coords[:, idx_cb]
    
    # Vectors radiating from CA
    v_n  = n - ca
    v_c  = c - ca
    v_cb = cb - ca
    
    # Volume = v_n . (v_c x v_cb)
    # Note: The order of cross product determines the sign convention.
    cross_prod = torch.linalg.cross(v_c, v_cb, dim=1)
    volume = torch.sum(v_n * cross_prod, dim=1) # Shape (Batch,)

    # --- 3. Identify "Wrong" Samples ---
    if target_sign == 'negative':
        # We want Volume < 0. If Volume > 0, it's wrong.
        mask = volume > 0
    else:
        # We want Volume > 0. If Volume < 0, it's wrong.
        mask = volume < 0
        
    # --- 4. Reflect "Wrong" Samples ---
    # We reflect across the X-axis: (x, y, z) -> (-x, y, z)
    # This inverts chirality.
    if mask.any():
        new_coords[mask, :, 0] *= -1
        
    return new_coords, mask.sum().item()