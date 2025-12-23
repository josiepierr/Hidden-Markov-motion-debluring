import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def optimal_bandwidth_ess(particles, weights):
    """
    Calculate optimal bandwidth using effective sample size
    """
    # Weighted mean
    weighted_mean = np.sum(weights * particles)
    # Weighted variance
    weighted_var = np.sum(weights * (particles - weighted_mean)**2)
    weighted_std = np.sqrt(weighted_var)
    
    # Effective sample size
    n_eff = 1.0 / np.sum(weights**2)
    
    # Silverman's rule of thumb
    bandwidth = 1.06 * weighted_std * n_eff**(-0.2)
    return bandwidth


@njit(parallel=True, fastmath=True)
def _gaussian_kde_numba(eval_points, x_particles, y_particles, weights, bw_x, bw_y):
    # Manually implement weighted KDE with anisotropic bandwidth (accelerated + parallel)

    n_eval = eval_points.shape[0]
    n_particles = x_particles.shape[0]
    density = np.zeros(n_eval, dtype=np.float64)

    # Normalization factor for 2D Gaussian
    norm_factor = 1.0 / (2.0 * np.pi * bw_x * bw_y)

    inv_bw_x = 1.0 / bw_x
    inv_bw_y = 1.0 / bw_y

    # Parallelize across evaluation points
    for i in prange(n_eval):
        x_eval = eval_points[i, 0]
        y_eval = eval_points[i, 1]

        acc = 0.0
        for j in range(n_particles):
            dx = (x_particles[j] - x_eval) * inv_bw_x
            dy = (y_particles[j] - y_eval) * inv_bw_y
            acc += weights[j] * (norm_factor * np.exp(-0.5 * (dx * dx + dy * dy)))

        density[i] = acc

    return density


def gaussian_kde(eval_points, x_particles, y_particles, weights, bw_x, bw_y):
    """
    Implement weighted KDE with anisotropic bandwidth
    """
    eval_points = np.ascontiguousarray(eval_points, dtype=np.float64)
    x_particles = np.ascontiguousarray(x_particles, dtype=np.float64)
    y_particles = np.ascontiguousarray(y_particles, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    density = _gaussian_kde_numba(eval_points, x_particles, y_particles, weights, bw_x, bw_y)

    print(f"Density range: [{density.min():.6e}, {density.max():.6e}]")
    return density


def reconstruct_image_from_particles(x_particles, y_particles, weights, image_shape, epsilon):
    """
    Reconstruct image from particles using KDE - matches MATLAB exactly
    
    Parameters:
    -----------
    x_particles : ndarray
        Particle x-coordinates (should be from final iteration)
    y_particles : ndarray
        Particle y-coordinates (should be from final iteration)
    weights : ndarray
        Particle weights (should be from final iteration)
    image_shape : tuple
        Shape of the output image (height, width)
    epsilon : float
        Smoothing parameter from SMC
    
    Returns:
    --------
    reconstructed_image : ndarray
        Reconstructed image normalized to [0, 1]
    """
    height, width = image_shape

    eval_x = np.linspace(-1 + 1 / width, 1 - 1 / width, width)
    eval_y = np.linspace(0.5 - 1 / height, -0.5 + 1 / height, height)

    grid_x, grid_y = np.meshgrid(eval_x, eval_y)

    eval_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    bw_x = np.sqrt(epsilon ** 2 + optimal_bandwidth_ess(x_particles, weights) ** 2)
    bw_y = np.sqrt(epsilon ** 2 + optimal_bandwidth_ess(y_particles, weights) ** 2)

    print(f"Bandwidths: bw_x={bw_x:.6f}, bw_y={bw_y:.6f}")

    density = gaussian_kde(eval_points, x_particles, y_particles, weights, bw_x, bw_y)

    reconstructed = density.reshape((height, width))
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-10)
    return reconstructed
