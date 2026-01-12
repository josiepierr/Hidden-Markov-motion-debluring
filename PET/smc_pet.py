# PET SMC-EMS (Crucinio-Doucet-Johansen) — Python implementation
# Fast version: O(N*M) per iteration (M << N), robust Poisson scaling, plots phantom + reconstructions.
#
# pip install numpy matplotlib scikit-image

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize


# ---------------------------
# Utils
# ---------------------------
def normpdf(z, mu=0.0, sigma=1.0):
    z = (z - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma)


def effective_sample_size(w):
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)
    return 1.0 / np.sum(w**2)


def weighted_mean(x, w):
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)
    return np.sum(w * np.asarray(x, float))


def weighted_std(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)
    m = np.sum(w * x)
    v = np.sum(w * (x - m) ** 2)
    return np.sqrt(max(v, 1e-300))


def optimal_bandwidthESS(x, w):
    """
    Silverman rule with ESS (paper-style):
      bw = 1.06 * std * ESS^{-1/5}
    """
    ess = effective_sample_size(w)
    s = weighted_std(x, w)
    return 1.06 * s * (ess ** (-1.0 / 5.0))


def multinomial_resample(w, rng):
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)
    N = w.size
    return rng.choice(np.arange(N), size=N, replace=True, p=w)


def sample_from_sinogram(xi, phi, Rprime, M, rng):
    """
    Draw M samples y=(xi,phi) from sinogram intensity Rprime (shape: n_phi x n_xi).
    Returns array (M,2): [xi_sample, phi_sample]
    """
    flat = np.asarray(Rprime, float).reshape(-1)
    s = flat.sum()
    if s <= 0:
        idx = rng.integers(0, flat.size, size=M)
    else:
        p = flat / s
        idx = rng.choice(np.arange(flat.size), size=M, replace=True, p=p)

    n_phi = len(phi)
    n_xi = len(xi)
    phi_idx = idx // n_xi
    xi_idx = idx % n_xi
    return np.column_stack([xi[xi_idx], phi[phi_idx]])


def weighted_kde_grid(x1, x2, w, grid_x, grid_y, bw1, bw2):
    """
    2D product Gaussian KDE on a tensor grid (grid_x x grid_y).
    Returns array shape (len(grid_x), len(grid_y)).

    dens(xg, yg) = sum_i w_i N(xg - x_i; 0, bw1) N(yg - y_i; 0, bw2)
    """
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)

    grid_x = np.asarray(grid_x, float)
    grid_y = np.asarray(grid_y, float)

    Kx = normpdf(grid_x[:, None] - x1[None, :], 0.0, bw1)  # (Gx,N)
    Ky = normpdf(grid_y[:, None] - x2[None, :], 0.0, bw2)  # (Gy,N)

    return (Kx * w[None, :]) @ Ky.T  # (Gx,Gy)


def Hreconstruction_pet(phi, xi, sigma, KDEx, KDEy):
    """
    Convolution of estimated f with PET kernel (as in MATLAB Hreconstruction_pet.m):
      hatH(xi_i, phi_j) = dx^2 * sum_p N(x_p cos(phi_j) + y_p sin(phi_j) - xi_i; 0, sigma) * f_hat(p)
    KDEx: (P,2) grid points, KDEy: (P,) density at those points
    """
    KDEx = np.asarray(KDEx, float)
    KDEy = np.asarray(KDEy, float)

    xs = np.unique(KDEx[:, 0])
    if len(xs) < 2:
        raise ValueError("KDEx grid too small for dx.")
    dx = xs[1] - xs[0]

    x = KDEx[:, 0]
    y = KDEx[:, 1]
    xi = np.asarray(xi, float)
    phi = np.asarray(phi, float)

    hatH = np.zeros((len(xi), len(phi)), float)

    for j, ph in enumerate(phi):
        s = x * np.cos(ph) + y * np.sin(ph)         # (P,)
        res = s[None, :] - xi[:, None]              # (n_xi, P)
        g = normpdf(res, 0.0, sigma)                # (n_xi, P)
        hatH[:, j] = (dx**2) * (g @ KDEy)

    return hatH


# ---------------------------
# SMC-EMS PET (FAST O(N*M))
# ---------------------------
def smc_pet_fast(
    N,
    maxIter,
    epsilon,
    phi,
    xi,
    R,         # sinogram, shape (n_det, n_phi)
    sigma,
    m=15,
    M=512,     # Monte Carlo samples per iteration from sinogram (M << N)
    seed=0
):
    """
    Vectorized version of MATLAB smc_pet.m but computationally feasible.
    Key: use a batch of M sinogram samples y=(xi,phi) per iteration (shared across particles).
    """
    rng = np.random.default_rng(seed)

    pixels = len(phi)

    # particle trajectories and weights (store all, like MATLAB)
    x1 = np.zeros((maxIter, N), float)
    x2 = np.zeros((maxIter, N), float)
    W  = np.zeros((maxIter, N), float)

    # init particles in [-0.75,0.75]^2
    x1[0, :] = 1.5 * rng.random(N) - 0.75
    x2[0, :] = 1.5 * rng.random(N) - 0.75
    W[0, :]  = 1.0 / N

    # KDE eval grid
    evalX1 = np.linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels)
    evalX2 = np.linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels)
    RevalX = np.tile(evalX1, (pixels, 1))
    eval_grid = np.column_stack([RevalX.reshape(-1), np.tile(evalX2, pixels)])

    delta1 = phi[1] - phi[0]
    delta2 = xi[1] - xi[0]

    # initial KDE and hatH
    bw1 = np.sqrt(epsilon**2 + optimal_bandwidthESS(x1[0, :], W[0, :])**2)
    bw2 = np.sqrt(epsilon**2 + optimal_bandwidthESS(x2[0, :], W[0, :])**2)
    kde_grid = weighted_kde_grid(x1[0, :], x2[0, :], W[0, :], evalX1, evalX2, bw1, bw2)
    KDE_vec = kde_grid.reshape(-1)
    hatHNew = Hreconstruction_pet(phi, xi, sigma, eval_grid, KDE_vec)

    zeta = np.zeros(maxIter, float)
    zeta[0] = (bw1**2 - weighted_mean(x1[0, :], W[0, :])**2) + (bw2**2 - weighted_mean(x2[0, :], W[0, :])**2)

    L2norm = np.zeros(maxIter, float)
    moving_var = np.zeros(maxIter, float)
    L2norm[0] = delta1 * delta2 * np.sum(hatHNew**2)

    iter_stop = maxIter

    # sampling uses R' in MATLAB -> (phi, xi)
    Rprime = R.T  # shape: (n_phi, n_xi)

    print("Start SMC (FAST)")

    for n in range(1, maxIter):
        hatHOld = hatHNew

        # draw M samples from sinogram
        y_samples = sample_from_sinogram(xi, phi, Rprime, M, rng)  # (M,2)
        xi_s = y_samples[:, 0]                                     # (M,)
        ph_s = y_samples[:, 1]                                     # (M,)
        cos_ph = np.cos(ph_s)[None, :]                             # (1,M)
        sin_ph = np.sin(ph_s)[None, :]                             # (1,M)

        # ESS + resampling
        ess = effective_sample_size(W[n-1, :])
        if ess < N / 2.0:
            idx = multinomial_resample(W[n-1, :], rng)
            x1[n, :] = x1[n-1, idx]
            x2[n, :] = x2[n-1, idx]
            W[n, :]  = 1.0 / N
        else:
            x1[n, :] = x1[n-1, :]
            x2[n, :] = x2[n-1, :]
            W[n, :]  = W[n-1, :]

        # Markov kernel (smoothing)
        x1[n, :] = x1[n, :] + epsilon * rng.standard_normal(N)
        x2[n, :] = x2[n, :] + epsilon * rng.standard_normal(N)

        # ----- Compute hN for the M sinogram samples using previous particles (as in MATLAB) -----
        # hN[m] = mean_i W(n,i) * N(x_prev_i cos(phi_m) + y_prev_i sin(phi_m) - xi_m; 0, sigma)
        x_prev = x1[n-1, :][:, None]  # (N,1)
        y_prev = x2[n-1, :][:, None]  # (N,1)
        w_curr = W[n, :][:, None]     # (N,1)

        res_prev = x_prev * cos_ph + y_prev * sin_ph - xi_s[None, :]  # (N,M)
        g_prev = normpdf(res_prev, 0.0, sigma)                         # (N,M)
        hN = np.mean(w_curr * g_prev, axis=0) + 1e-300                 # (M,)

        # ----- Update weights for all particles (vectorized) -----
        x_cur = x1[n, :][:, None]                                      # (N,1)
        y_cur = x2[n, :][:, None]                                      # (N,1)
        res_cur = x_cur * cos_ph + y_cur * sin_ph - xi_s[None, :]      # (N,M)
        g_cur = normpdf(res_cur, 0.0, sigma)                           # (N,M)

        potential = np.mean(g_cur / hN[None, :], axis=1)               # (N,)
        W[n, :] = W[n, :] * potential
        W[n, :] = W[n, :] / (W[n, :].sum() + 1e-300)

        # KDE bandwidth + zeta
        bw1 = np.sqrt(epsilon**2 + optimal_bandwidthESS(x1[n, :], W[n, :])**2)
        bw2 = np.sqrt(epsilon**2 + optimal_bandwidthESS(x2[n, :], W[n, :])**2)
        zeta[n] = (bw1**2 - weighted_mean(x1[n, :], W[n, :])**2) + (bw2**2 - weighted_mean(x2[n, :], W[n, :])**2)

        # reconstruct h (for stopping)
        kde_grid = weighted_kde_grid(x1[n, :], x2[n, :], W[n, :], evalX1, evalX2, bw1, bw2)
        KDE_vec = kde_grid.reshape(-1)
        hatHNew = Hreconstruction_pet(phi, xi, sigma, eval_grid, KDE_vec)

        L2norm[n] = delta1 * delta2 * np.sum((hatHNew - hatHOld) ** 2)

        if n + 1 >= m:
            moving_var[n] = np.var(zeta[(n - m + 1):(n + 1)])
            if L2norm[n] <= moving_var[n] and iter_stop == maxIter:
                iter_stop = n + 1
                print(f"Stop at iteration {iter_stop}")

        if (n + 1) % 5 == 0 or (n + 1) == 2:
            print(f"iter {n+1:3d}/{maxIter} | ESS={ess:,.0f}")

    print("SMC Finished")
    return x1, x2, W, iter_stop, moving_var, L2norm


# ---------------------------
# Main (like pet.m)
# ---------------------------
def main():
    rng = np.random.default_rng(0)

    pixels = 128

    # Phantom
    P = shepp_logan_phantom()
    P = resize(P, (pixels, pixels), anti_aliasing=True, mode="reflect")
    P = np.clip(P, 0, None)
    P = P / (P.max() + 1e-300)

    plt.figure(figsize=(4, 4))
    plt.imshow(P, cmap="gray")
    plt.title("Shepp–Logan phantom (ground truth)")
    plt.axis("off")
    plt.show()

    # Radon projections
    phi_deg = np.linspace(0, 360, pixels, endpoint=True)
    R = radon(P, theta=phi_deg, circle=False)  # shape: (n_det, n_angles)

    # Build xi (offsets) and normalize like MATLAB
    n_det = R.shape[0]
    diag = np.sqrt(2.0) * (pixels // 2)
    xi = np.linspace(-diag, diag, n_det)
    xi = xi / (np.max(np.abs(xi)) + 1e-300)

    # Convert phi to radians
    phi = np.deg2rad(phi_deg)

    # Poisson noise (IMPORTANT: do NOT use 1e-12 in Python; it makes almost all zeros)
    # Choose a scale so the Poisson counts are meaningful, then normalize like MATLAB.
    scale = 5e4  # try 1e4, 5e4, 1e5 if needed
    noisyR = rng.poisson(np.clip(R, 0, None) * scale).astype(float)
    noisyR = noisyR / (noisyR.max() + 1e-300)

    # Parameters (paper-like)
    Niter = 100       # for quick tests; then increase to 100 if needed
    N = 20000            # paper uses 20000
    epsilon = 1e-3       # smoothing parameter
    sigma = 0.02         # alignment std (paper PET uses small variance; MATLAB uses 0.02)
    m = 15               # moving window for stopping
    M = 512              # sinogram samples per iteration (speed/quality tradeoff)

    x, y, W, iter_stop, mv, L2 = smc_pet_fast(
        N=N, maxIter=Niter, epsilon=epsilon, phi=phi, xi=xi, R=noisyR,
        sigma=sigma, m=m, M=M, seed=1
    )

    # Plot selected iterations (keep those <= Niter)
    #showIter = [1, 5, Niter]
    showIter = [1, 5, 10, 15, 20, 50, 70, Niter]
    showIter = [it for it in showIter if it <= Niter]

    evalX = np.linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels)
    evalY = np.linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels)

    fig1, axs1 = plt.subplots(2, 4, figsize=(12, 6))
    fig2, axs2 = plt.subplots(2, 4, figsize=(12, 6))
    axs1 = axs1.ravel()
    axs2 = axs2.ravel()

    Pn = P - P.min()
    Pn = Pn / (Pn.max() + 1e-300)

    mse_list = []
    for k, it in enumerate(showIter):
        n = it - 1
        bw1 = np.sqrt(epsilon**2 + optimal_bandwidthESS(x[n, :], W[n, :])**2)
        bw2 = np.sqrt(epsilon**2 + optimal_bandwidthESS(y[n, :], W[n, :])**2)
        kde_grid = weighted_kde_grid(x[n, :], y[n, :], W[n, :], evalX, evalY, bw1, bw2)

        KDEn = kde_grid
        KDEn = KDEn - KDEn.min()
        KDEn = KDEn / (KDEn.max() + 1e-300)
        KDEn = np.flipud(KDEn)
        KDEn = np.rot90(KDEn, k=1)


        axs1[k].imshow(KDEn, cmap="hot")
        axs1[k].set_title(f"Iteration {it}", fontsize=10)
        axs1[k].axis("off")

        mse_list.append(float(np.mean((Pn - KDEn) ** 2)))

        err = np.abs(KDEn - Pn)
        positive = (Pn > 0)
        err_rel = err.copy()
        err_rel[positive] = err[positive] / (Pn[positive] + 1e-300)

        axs2[k].imshow(err_rel, cmap="hot")
        axs2[k].set_title(f"Iteration {it}", fontsize=10)
        axs2[k].axis("off")

    # Hide unused subplots
    for kk in range(len(showIter), 8):
        axs1[kk].axis("off")
        axs2[kk].axis("off")

    plt.tight_layout()
    plt.show()

    print("MSE:", mse_list)
    print("iter_stop:", iter_stop if iter_stop != Niter else f"(not triggered <= {Niter})")
    if iter_stop <= Niter:
        ess_stop = effective_sample_size(W[iter_stop - 1, :])
        print("ESS at iter_stop:", ess_stop)


if __name__ == "__main__":
    main()
