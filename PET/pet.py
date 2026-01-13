#!/usr/bin/env python3

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from skimage.data import shepp_logan_phantom
from skimage.transform import radon

from smc_pet import smc_pet, _kde_2d_numba
from optimal_bandwidthESS import optimal_bandwidthESS


def _to_unit_range(A):
    mn = float(np.min(A))
    mx = float(np.max(A))
    if mx > mn:
        return (A - mn) / (mx - mn)
    return A * 0.0


def _save_image(path, A, title=None, cmap='hot', vmin=0.0, vmax=1.0):
    plt.figure(figsize=(4, 4))
    plt.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _kde_to_image(kde_flat, pixels):
    img = np.reshape(kde_flat, (pixels, pixels), order='F')
    img = _to_unit_range(img)
    img = np.flipud(img)
    return img


def reconstruct_image(x_row, y_row, w_row, pixels, epsilon):
    evalX = np.linspace(-0.75 + 1.0 / pixels, 0.75 - 1.0 / pixels, pixels)
    evalY = np.linspace(-0.75 + 1.0 / pixels, 0.75 - 1.0 / pixels, pixels)
    eval_x = np.repeat(evalX, pixels)
    eval_y = np.tile(evalY, pixels)

    bw1 = np.sqrt(epsilon**2 + optimal_bandwidthESS(x_row, w_row) ** 2)
    bw2 = np.sqrt(epsilon**2 + optimal_bandwidthESS(y_row, w_row) ** 2)

    kde_flat = _kde_2d_numba(x_row, y_row, w_row, eval_x, eval_y, bw1, bw2)
    return _kde_to_image(kde_flat, pixels)


def ise(P, K, pixels):
    dx = 1.5 / pixels
    return float(np.sum((P - K) ** 2) * (dx * dx))


def main():
    np.random.seed(0)

    pixels = 128
    Niter = 100
    N = 20000
    epsilon = 1e-3
    sigma = 0.02
    m = 15

    showIter = [1, 5, 10, 15, 20, 50, 70, Niter]
    progress_every = 1

    out_dir = 'outputs'
    recon_dir = os.path.join(out_dir, 'reconstructions')
    os.makedirs(recon_dir, exist_ok=True)

    P = shepp_logan_phantom()
    if P.shape[0] != pixels:
        P = zoom(P, pixels / P.shape[0], order=1)
    P[P < 0] = 0.0
    P = _to_unit_range(P)

    _save_image(os.path.join(out_dir, '00_phantom.png'), P, title='Phantom', cmap='hot')
    _save_image(os.path.join(out_dir, '00_phantom_gray.png'), P, title='Phantom', cmap='gray')

    phi_deg = np.linspace(0, 360, pixels, endpoint=True)
    R = radon(P, theta=phi_deg, circle=False)

    n_xi = R.shape[0]
    xi_raw = np.arange(n_xi, dtype=np.float64) - (n_xi - 1) / 2.0
    xi = xi_raw / (np.max(np.abs(xi_raw)) + 1e-300)

    phi = np.deg2rad(phi_deg)

    poisson_scale = 5e4
    Rpos = np.clip(R, 0.0, None)
    R01 = _to_unit_range(Rpos)
    counts = R01 * poisson_scale
    noisy_counts = np.random.poisson(counts).astype(np.float64)
    R_noisy = noisy_counts / (poisson_scale + 1e-300)
    R_noisy = R_noisy / (R_noisy.max() + 1e-300)

    _save_image(os.path.join(out_dir, '01_radon_clean.png'), _to_unit_range(Rpos), title='Radon data (clean)', cmap='hot')
    _save_image(os.path.join(out_dir, '02_radon_noisy.png'), R_noisy, title='Radon data (noisy)', cmap='hot')

    t0 = time.time()
    x, y, W, iter_stop, mv, L2 = smc_pet(
        N, Niter, epsilon, phi, xi, R_noisy, sigma, m,
        progress=True, progress_every=progress_every
    )
    runtime = time.time() - t0
    print(f"SMC runtime: {runtime/60:.2f} min, iter_stop={iter_stop+1}")

    ESS = 1.0 / (np.sum(W**2, axis=1) + 1e-300)
    metrics = np.column_stack([np.arange(1, Niter + 1), ESS, L2, mv])
    np.savetxt(
        os.path.join(out_dir, 'metrics.csv'),
        metrics,
        delimiter=',',
        header='iter,ESS,L2norm,moving_var',
        comments=''
    )

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, Niter + 1), ESS)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('ESS')
    plt.title('Effective Sample Size')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ess_vs_iteration.png'), dpi=150, bbox_inches='tight')
    plt.close()

    showIter = [it for it in showIter if 1 <= it <= Niter]

    ise_vals = []
    mse_vals = []

    fig1, axes1 = plt.subplots(2, 4, figsize=(14, 6))
    fig2, axes2 = plt.subplots(2, 4, figsize=(14, 6))
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()

    for k, it in enumerate(showIter):
        n = it - 1
        K = reconstruct_image(x[n, :], y[n, :], W[n, :], pixels, epsilon)

        _save_image(os.path.join(recon_dir, f'recon_iter_{it:03d}.png'), K, title=f'Iteration {it}', cmap='hot')

        err = np.abs(K - P)
        _save_image(os.path.join(recon_dir, f'error_abs_iter_{it:03d}.png'), _to_unit_range(err), title=f'Iteration {it}', cmap='hot')

        positive = (P > 0)
        err_rel = err.copy()
        err_rel[positive] = err[positive] / (P[positive] + 1e-10)
        _save_image(os.path.join(recon_dir, f'error_rel_iter_{it:03d}.png'), _to_unit_range(err_rel), title=f'Iteration {it}', cmap='hot')

        axes1[k].imshow(K, cmap='hot', vmin=0.0, vmax=1.0)
        axes1[k].set_title(f'Iteration {it}', fontsize=10)
        axes1[k].axis('off')

        axes2[k].imshow(err_rel, cmap='hot')
        axes2[k].set_title(f'Iteration {it}', fontsize=10)
        axes2[k].axis('off')

        mse_vals.append(float(np.mean((K - P) ** 2)))
        ise_vals.append(ise(P, K, pixels))

    for kk in range(len(showIter), 8):
        axes1[kk].axis('off')
        axes2[kk].axis('off')

    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'pet_reconstructions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'pet_relative_errors.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    K_final = reconstruct_image(x[Niter - 1, :], y[Niter - 1, :], W[Niter - 1, :], pixels, epsilon)
    np.save(os.path.join(out_dir, 'pet_reconstruction_final.npy'), K_final)
    _save_image(os.path.join(out_dir, 'pet_reconstruction_final.png'), K_final, title=f'Iteration {Niter}', cmap='hot')

    ise_table = np.column_stack([
        np.array(showIter, dtype=int),
        np.array(mse_vals, dtype=float),
        np.array(ise_vals, dtype=float),
    ])
    np.savetxt(os.path.join(out_dir, 'ise_selected.csv'), ise_table, delimiter=',', header='iter,MSE,ISE', comments='')

    plt.figure(figsize=(6, 4))
    plt.plot(showIter, ise_vals, '-o')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('ISE')
    plt.title('Integrated Squared Error')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ise_vs_iteration.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
