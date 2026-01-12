"""
SMC-EMS FOR PET RECONSTRUCTION - FREDHOLM CORRECT (NUMBA FIXED)
Based on: Crucinio et al., "A Particle Method for Solving Fredholm Equations" (2021)

FIX: np.mean with axis parameter is NOT supported in numba nopython mode
Solution: Manually compute the ensemble average h_n
"""

import numpy as np
from numba import jit, prange
import time

print("=" * 80)
print("CORRECT SMC-EMS ALGORITHM FOR PET (NUMBA FIXED)")
print("=" * 80)


@jit(nopython=True, parallel=True)
def compute_likelihood_fredholm_correct(x1, x2, R_noisy, angles_rad, xi_array, sigma_forward):
    """
    CORRECT FREDHOLM EQUATION APPROACH
    
    Weight Update (Algorithm 1, line 3 from paper):
        G_n(x_n, y_n) = g(y_n | x_n) / h_n(y_n)
        
    where:
        g(y | x) = forward model kernel (Gaussian Radon)
        h_n(y) = (1/N) Σ_i g(y | X_i_n)  [particle ensemble average]
    
    This is the KEY DIFFERENCE from standard likelihood!
    The denominator normalizes by what the average particle predicts.
    """
    
    N = x1.shape[0]
    n_angles = angles_rad.shape[0]
    n_xi = xi_array.shape[0]
    
    # ===========================================================================
    # STEP 1: Compute forward model for all particles
    # ===========================================================================
    # g_pred[i, xi_idx, angle_idx] = g(Y_i | X_i) for particle i
    g_pred = np.zeros((N, n_xi, n_angles))
    
    for i in prange(N):
        for angle_idx in range(n_angles):
            cos_angle = np.cos(angles_rad[angle_idx])
            sin_angle = np.sin(angles_rad[angle_idx])
            proj = x1[i] * cos_angle + x2[i] * sin_angle
            
            for xi_idx in range(n_xi):
                xi_val = xi_array[xi_idx]
                distance = proj - xi_val
                g_pred[i, xi_idx, angle_idx] = np.exp(-0.5 * (distance / sigma_forward) ** 2)
    
    # ===========================================================================
    # STEP 2: Compute h_n(y) = ensemble average = (1/N) Σ_i g(y | X_i)
    # ===========================================================================
    # FIX: Can't use np.mean(g_pred, axis=0) in numba nopython mode
    # Manually compute the average
    h_n = np.zeros((n_xi, n_angles))
    
    for xi_idx in range(n_xi):
        for angle_idx in range(n_angles):
            # Sum over all particles
            total = 0.0
            for i in range(N):
                total += g_pred[i, xi_idx, angle_idx]
            # Average
            h_n[xi_idx, angle_idx] = total / N
    
    # Avoid division by zero (very small ensemble predictions)
    for xi_idx in range(n_xi):
        for angle_idx in range(n_angles):
            h_n[xi_idx, angle_idx] = max(h_n[xi_idx, angle_idx], 1e-10)
    
    # ===========================================================================
    # STEP 3: Compute weights = g(y|x) / h_n(y) for each particle
    # ===========================================================================
    log_weights = np.zeros(N)
    
    for i in prange(N):
        # For each observation (xi, angle), weight contribution is:
        # log W_i += log(g(y|x_i)) - log(h_n(y))
        log_likelihood = 0.0
        
        for angle_idx in range(n_angles):
            for xi_idx in range(n_xi):
                g_val = g_pred[i, xi_idx, angle_idx]
                h_val = h_n[xi_idx, angle_idx]
                
                # log(g/h_n) = log(g) - log(h_n)
                log_likelihood += np.log(g_val + 1e-10) - np.log(h_val)
        
        log_weights[i] = log_likelihood
    
    return log_weights


@jit(nopython=True, parallel=True)
def markov_kernel_move(x1, x2, sigma=0.02):
    """Random walk smoothing kernel (EMS smoothing)"""
    N = x1.shape[0]
    new_x1 = np.zeros_like(x1)
    new_x2 = np.zeros_like(x2)
    
    for i in prange(N):
        new_x1[i] = x1[i] + np.random.normal(0, sigma)
        new_x2[i] = x2[i] + np.random.normal(0, sigma)
    
    return new_x1, new_x2


@jit(nopython=True)
def resample_particles(x1, x2, weights, N):
    """Systematic resampling"""
    W_norm = weights / np.sum(weights)
    cumsum = np.cumsum(W_norm)
    
    x1_resampled = np.zeros(N)
    x2_resampled = np.zeros(N)
    
    u = np.random.uniform(0, 1.0 / N)
    j = 0
    
    for i in range(N):
        threshold = u + i / N
        
        while cumsum[j] < threshold and j < N - 1:
            j += 1
        
        x1_resampled[i] = x1[j]
        x2_resampled[i] = x2[j]
    
    return x1_resampled, x2_resampled


def smc_ems_pet_correct(N, Niter, epsilon, angles, R_noisy, sigma_forward,
                        tolerance_window=15, return_history=True):
    """
    SMC-EMS for PET Reconstruction - CORRECT FREDHOLM VERSION
    
    Based on: Crucinio et al., 2021
    Algorithm 1: Particle Method for Fredholm Equations of the First Kind
    """
    
    n_xi, n_angles = R_noisy.shape
    
    # Initialize particles uniformly
    x1 = np.random.uniform(-0.75, 0.75, N)
    x2 = np.random.uniform(-0.75, 0.75, N)
    W = np.ones(N) / N
    
    # Offset positions
    xi = np.linspace(-1.0, 1.0, n_xi)
    
    # History storage
    if return_history:
        x1_history = np.zeros((Niter, N))
        x2_history = np.zeros((Niter, N))
        W_history = np.zeros((Niter, N))
    else:
        x1_history = None
        x2_history = None
        W_history = None
    
    ess_hist = []
    l2_hist = []
    
    print(f"\n{'Iter':<5} {'ESS':<10} {'L2-norm':<12} {'Max W':<12} {'Status':<20}")
    print("-" * 60)
    
    for iteration in range(Niter):
        
        # STEP 1: MARKOV KERNEL MOVE (EMS smoothing)
        x1_moved, x2_moved = markov_kernel_move(x1, x2, sigma=epsilon)
        
        # STEP 2: LIKELIHOOD EVALUATION (FREDHOLM CORRECT VERSION!)
        log_weights = compute_likelihood_fredholm_correct(
            x1_moved, x2_moved, R_noisy, angles, xi, 
            sigma_forward=sigma_forward
        )
        
        # Numerical stability
        log_weights = log_weights - np.max(log_weights)
        weights_unnorm = np.exp(log_weights)
        
        # SMC weight update
        W = W * weights_unnorm
        W = W / (np.sum(W) + 1e-10)
        
        # STEP 3: DIAGNOSTICS
        ess = 1.0 / np.sum(W ** 2)
        ess_hist.append(ess)
        
        l2_norm = np.sqrt(np.mean((x1_moved - x1) ** 2 + (x2_moved - x2) ** 2))
        l2_hist.append(l2_norm)
        
        max_w = np.max(W)
        
        # STEP 4: STORE HISTORY
        if return_history:
            x1_history[iteration, :] = x1_moved
            x2_history[iteration, :] = x2_moved
            W_history[iteration, :] = W
        
        # STEP 5: PRINT PROGRESS
        status = "Running"
        if ess < N / 2:
            status = "Resampling..."
        
        print(f"{iteration + 1:<5} {ess:<10.0f} {l2_norm:<12.6f} {max_w:<12.8f} {status:<20}")
        
        # STEP 6: RESAMPLE IF ESS LOW
        if ess < N / 2:
            x1, x2 = resample_particles(x1_moved, x2_moved, W, N)
            W = np.ones(N) / N
        else:
            x1, x2 = x1_moved, x2_moved
        
        # STEP 7: CONVERGENCE CHECK
        if iteration > tolerance_window:
            recent = l2_hist[-tolerance_window:]
            if np.std(recent) < 0.01 * np.mean(recent) and np.mean(recent) < 0.1:
                print(f"\n✓ Converged at iteration {iteration + 1}")
                iterations_actual = iteration + 1
                
                if return_history:
                    x1_history = x1_history[:iterations_actual, :]
                    x2_history = x2_history[:iterations_actual, :]
                    W_history = W_history[:iterations_actual, :]
                
                break
        else:
            iterations_actual = Niter
    else:
        iterations_actual = Niter
    
    ess_hist = np.array(ess_hist)
    l2_hist = np.array(l2_hist)
    
    print(f"\n✓ SMC-EMS completed after {iterations_actual} iterations")
    print(f" Final ESS: {ess:.0f}/{N}")
    print(f" Final max weight: {max_w:.8f}")
    
    return x1_history, x2_history, W_history, iterations_actual, ess_hist, l2_hist


# ===========================================================================
# QUICK TEST
# ===========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity
    import warnings
    warnings.filterwarnings('ignore')

    # Import your data
    try:
        from pet_exact_corrected import phantom, noisyR, phi_rad, pixels, Niter, N, epsilon

        print("\n" + "="*80)
        print("RUNNING SMC-EMS (CORRECT FREDHOLM ALGORITHM)")
        print("="*80)

        start = time.time()

        x1_history, x2_history, W_history, iterations_actual, ess_hist, l2_hist = smc_ems_pet_correct(
            N=N, Niter=Niter, epsilon=epsilon, angles=phi_rad, R_noisy=noisyR,
            sigma_forward=0.020,  # ← VALEUR À TUNER (important!)
            tolerance_window=15, return_history=True
        )

        smc_time = time.time() - start

        print(f"\n✓ SMC-EMS done in {smc_time:.2f}s")

        # =======================================================================
        # RECONSTRUCT & SAVE FINAL PARTICLES (FIXED KDE)
        # =======================================================================

        print("\nReconstructing...")

        def reconstruct_kde_safe(x1, x2, W, bandwidth=0.03, grid_size=128):
            """Reconstruct image using KDE - SAFE VERSION (handles log(0))"""
            eval_x = np.linspace(-0.75, 0.75, grid_size)
            eval_y = np.linspace(-0.75, 0.75, grid_size)
            eval_X, eval_Y = np.meshgrid(eval_x, eval_y)
            eval_points = np.column_stack([eval_X.ravel(), eval_Y.ravel()])
            
            particles = np.column_stack([x1, x2])
            
            # Normalize weights
            W_norm = W / np.sum(W)
            
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', atol=1e-10, rtol=1e-10)
            
            # Suppress KDE warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kde.fit(particles, sample_weight=W_norm)
                
                # Evaluate density safely
                try:
                    log_density = kde.score_samples(eval_points)
                    # Clamp to avoid -inf from log(0)
                    log_density = np.maximum(log_density, -100)
                    density = np.exp(log_density)
                except:
                    print("⚠️ KDE numerical issue, using direct computation...")
                    density = np.zeros(len(eval_points))
                    for i, point in enumerate(eval_points):
                        # Manual Gaussian kernel density
                        distances = np.sqrt((particles[:, 0] - point[0])**2 + 
                                          (particles[:, 1] - point[1])**2)
                        kernel_vals = np.exp(-0.5 * (distances / bandwidth)**2)
                        density[i] = np.sum(W_norm * kernel_vals) / (2 * np.pi * bandwidth**2 + 1e-10)
            
            # Reshape and normalize
            image = density.reshape((grid_size, grid_size))
            image = np.maximum(image, 0)  # Ensure non-negative
            max_val = np.max(image) + 1e-10
            image = image / max_val
            image = np.flipud(image)
            
            return image

        x1_final = x1_history[-1, :]
        x2_final = x2_history[-1, :]
        W_final = W_history[-1, :]

        image_recon = reconstruct_kde_safe(x1_final, x2_final, W_final, bandwidth=0.03)

        # Metrics
        mse = np.mean((phantom - image_recon) ** 2)
        psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-10))

        print(f"✓ MSE={mse:.6f}, PSNR={psnr:.2f}dB")

        # =======================================================================
        # SAVE FINAL PARTICLES FOR OPTIMIZATION
        # =======================================================================

        print("\nSaving particle data for optimization...")
        np.save('x1_final.npy', x1_final)
        np.save('x2_final.npy', x2_final)
        np.save('W_final.npy', W_final)
        print("✓ Saved x1_final.npy, x2_final.npy, W_final.npy")

        # =======================================================================
        # VISUALIZATION
        # =======================================================================

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(phantom, cmap='hot')
        axes[0].set_title('Original Phantom', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(image_recon, cmap='hot')
        axes[1].set_title(f'Fredholm EMS\nMSE={mse:.6f}, PSNR={psnr:.2f}dB',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')

        error = np.abs(phantom - image_recon)
        axes[2].imshow(error, cmap='viridis')
        axes[2].set_title(f'Error (max={np.max(error):.4f})', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('test_fredholm_ems.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: test_fredholm_ems.png")
        plt.show()

        # =======================================================================
        # RESULTS
        # =======================================================================

        print("\n" + "="*80)
        print("RESULTS - SMC-EMS (CORRECT FREDHOLM ALGORITHM)")
        print("="*80)
        print(f"\nMSE: {mse:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"\n✅ Particle data saved. You can now run:")
        print(f"   python PHASE1_BandwidthTuning_FIXED.py")

    except ImportError as e:
        print(f"\n⚠️ Import error: {e}")
        print("\nTo run full test, make sure:")
        print(" 1. pet_exact_corrected.py has been executed")
        print(" 2. phantom.npy exists")
