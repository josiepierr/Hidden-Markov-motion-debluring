import numpy as np

def optimal_bandwidth_ess(particles, weights):
    """
    Calculate optimal bandwidth using effective sample size
    Matches MATLAB's optimal_bandwidthESS function
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

def gaussian_kde(eval_points, x_particles, y_particles, weights, bw_x, bw_y):
    # Manually implement weighted KDE with anisotropic bandwidth
    # MATLAB: ksdensity([x(Niter, :)' y(Niter, :)'], eval, 'Weight', W(Niter, :), ...)
    
    n_eval = len(eval_points)
    density = np.zeros(n_eval)
    
    # Normalization factor for 2D Gaussian
    norm_factor = 1.0 / (2 * np.pi * bw_x * bw_y)
    
    # For each evaluation point, sum weighted Gaussians from all particles
    for i in range(n_eval):
        x_eval, y_eval = eval_points[i]
        
        # Distance to each particle (scaled by bandwidth)
        dx = (x_particles - x_eval) / bw_x
        dy = (y_particles - y_eval) / bw_y
        
        # Gaussian kernel values
        kernel_vals = norm_factor * np.exp(-0.5 * (dx**2 + dy**2))
        
        # Weighted sum
        density[i] = np.sum(weights * kernel_vals)
    
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
    
    # Create coordinate grid matching MATLAB
    # x is in [-1, 1]
    eval_x = np.linspace(-1 + 1/width, 1 - 1/width, width)
    # y is in [-0.5, 0.5]
    eval_y = np.linspace(0.5 - 1/height, -0.5 + 1/height, height)
    
    # Build evaluation grid
    grid_x, grid_y = np.meshgrid(eval_x, eval_y)
    
    # Flatten for KDE evaluation - order matters!
    # MATLAB uses: eval =[RevalX(:) repmat(evalY, 1, pixels(2))']
    # This creates [x, y] pairs
    eval_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Calculate optimal bandwidths
    bw_x = np.sqrt(epsilon**2 + optimal_bandwidth_ess(x_particles, weights)**2)
    bw_y = np.sqrt(epsilon**2 + optimal_bandwidth_ess(y_particles, weights)**2)
    
    print(f"Bandwidths: bw_x={bw_x:.6f}, bw_y={bw_y:.6f}")
    
    density = gaussian_kde(eval_points, x_particles, y_particles, weights, bw_x, bw_y)
    
    # Reshape to image
    reconstructed = density.reshape((height, width))
    
    # Normalize to [0, 1] range like MATLAB's mat2gray
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-10)
    
    return reconstructed