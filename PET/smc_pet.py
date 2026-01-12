import time
import numpy as np
import numba

from hreconstruction_pet import hreconstruction_pet
from mult_resample import mult_resample
from optimal_bandwidthESS import optimal_bandwidthESS
from pinky import pinky

@numba.jit(nopython=True)
def _normpdf(x, sigma):
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)

@numba.jit(nopython=True)
def _kde_2d_numba(x1, x2, W, eval_x, eval_y, bw1, bw2):
    N = len(x1)
    n_eval = len(eval_x)
    kde = np.zeros(n_eval)

    W_sum = np.sum(W)
    if W_sum <= 0.0:
        W_sum = 1.0
    Wn = W / W_sum

    inv = 1.0 / (2.0 * np.pi * bw1 * bw2)
    for i in range(n_eval):
        s = 0.0
        ex = eval_x[i]
        ey = eval_y[i]
        for j in range(N):
            dx = (ex - x1[j]) / bw1
            dy = (ey - x2[j]) / bw2
            s += Wn[j] * np.exp(-0.5 * (dx * dx + dy * dy)) * inv
        kde[i] = s
    return kde

@numba.jit(nopython=True)
def _compute_hN(y, x1_old, x2_old, Wn, sigma):
    N = y.shape[0]
    hN = np.zeros(N)
    for j in range(N):
        c = np.cos(y[j, 1])
        s = np.sin(y[j, 1])
        xij = y[j, 0]
        acc = 0.0
        for k in range(N):
            proj = x1_old[k] * c + x2_old[k] * s - xij
            acc += Wn[k] * _normpdf(proj, sigma)
        hN[j] = acc / N
    return hN

@numba.jit(nopython=True)
def _update_weights(y, x1_new, x2_new, Wn, hN, sigma):
    N = y.shape[0]
    Wnew = Wn.copy()
    for i in range(N):
        acc = 0.0
        for j in range(N):
            c = np.cos(y[j, 1])
            s = np.sin(y[j, 1])
            xij = y[j, 0]
            proj = x1_new[i] * c + x2_new[i] * s - xij
            g = _normpdf(proj, sigma)
            denom = hN[j]
            if denom > 1e-300:
                acc += g / denom
        Wnew[i] = Wn[i] * (acc / N)
    return Wnew

def _to_unit_range(A):
    mn = float(np.min(A))
    mx = float(np.max(A))
    if mx > mn:
        return (A - mn) / (mx - mn)
    return A.copy()

def _kde_vec_for_h(kde_flat, pixels):
    img = np.reshape(kde_flat, (pixels, pixels), order='F')
    img = _to_unit_range(img)
    img = np.flipud(img)
    return np.reshape(img, (pixels * pixels,), order='F')

def smc_pet(N, maxIter, epsilon, phi, xi, R, sigma, m, *, progress=True, progress_every=1):
    pixels = len(phi)

    x1 = np.zeros((maxIter, N), dtype=np.float64)
    x2 = np.zeros((maxIter, N), dtype=np.float64)
    W = np.zeros((maxIter, N), dtype=np.float64)

    zeta = np.zeros(maxIter, dtype=np.float64)
    L2norm = np.zeros(maxIter, dtype=np.float64)
    moving_var = np.zeros(maxIter, dtype=np.float64)

    iter_stop = maxIter - 1

    np.random.seed(0)
    x1[0, :] = 1.5 * np.random.rand(N) - 0.75
    x2[0, :] = 1.5 * np.random.rand(N) - 0.75
    W[0, :] = 1.0 / N

    evalX1 = np.linspace(-0.75 + 1.0 / pixels, 0.75 - 1.0 / pixels, pixels)
    evalX2 = np.linspace(-0.75 + 1.0 / pixels, 0.75 - 1.0 / pixels, pixels)
    dx = evalX1[1] - evalX1[0]

    eval_x_flat = np.repeat(evalX1, pixels)
    eval_y_flat = np.tile(evalX2, pixels)
    eval_grid = np.column_stack([eval_x_flat, eval_y_flat])

    delta1 = phi[1] - phi[0] if len(phi) > 1 else 1.0
    delta2 = xi[1] - xi[0] if len(xi) > 1 else 1.0

    t0 = time.time()

    bw1 = np.sqrt(epsilon ** 2 + optimal_bandwidthESS(x1[0, :], W[0, :]) ** 2)
    bw2 = np.sqrt(epsilon ** 2 + optimal_bandwidthESS(x2[0, :], W[0, :]) ** 2)
    kde_flat = _kde_2d_numba(x1[0, :], x2[0, :], W[0, :], eval_x_flat, eval_y_flat, bw1, bw2)

    zeta[0] = np.sum(kde_flat ** 2) * dx ** 2
    kde_vec = _kde_vec_for_h(kde_flat, pixels)

    hatHNew = hreconstruction_pet(phi, xi, sigma, eval_grid, kde_vec)
    L2norm[0] = delta1 * delta2 * np.sum(hatHNew ** 2)

    if progress:
        print(f"Start SMC: N={N}, maxIter={maxIter}, pixels={pixels}", flush=True)

    RT = R.T

    for n in range(1, maxIter):
        it_start = time.time()

        y = pinky(xi, phi, RT, N)
        hatHOld = hatHNew

        ESS = 1.0 / (np.sum(W[n - 1, :] ** 2) + 1e-300)

        if ESS < N / 2:
            idx = mult_resample(W[n - 1, :], N)
            x1[n, :] = x1[n - 1, idx]
            x2[n, :] = x2[n - 1, idx]
            W[n, :] = 1.0 / N
        else:
            x1[n, :] = x1[n - 1, :]
            x2[n, :] = x2[n - 1, :]
            W[n, :] = W[n - 1, :]

        x1[n, :] = x1[n, :] + epsilon * np.random.randn(N)
        x2[n, :] = x2[n, :] + epsilon * np.random.randn(N)

        hN = _compute_hN(y, x1[n - 1, :], x2[n - 1, :], W[n, :], sigma)
        Wn = _update_weights(y, x1[n, :], x2[n, :], W[n, :], hN, sigma)

        sW = np.sum(Wn)
        if sW > 0:
            W[n, :] = Wn / sW
        else:
            W[n, :] = 1.0 / N

        bw1 = np.sqrt(epsilon ** 2 + optimal_bandwidthESS(x1[n, :], W[n, :]) ** 2)
        bw2 = np.sqrt(epsilon ** 2 + optimal_bandwidthESS(x2[n, :], W[n, :]) ** 2)
        kde_flat = _kde_2d_numba(x1[n, :], x2[n, :], W[n, :], eval_x_flat, eval_y_flat, bw1, bw2)

        zeta[n] = (bw1 ** 2 - (np.sum(W[n, :] * x1[n, :]) ** 2)) + (bw2 ** 2 - (np.sum(W[n, :] * x2[n, :]) ** 2))

        kde_vec = _kde_vec_for_h(kde_flat, pixels)
        hatHNew = hreconstruction_pet(phi, xi, sigma, eval_grid, kde_vec)

        L2norm[n] = delta1 * delta2 * np.sum((hatHNew - hatHOld) ** 2)

        if n >= m:
            moving_var[n] = np.var(zeta[(n - m + 1):(n + 1)])
            if L2norm[n] <= moving_var[n]:
                iter_stop = n
                if progress:
                    print(f"Stop criterion met at iter={n+1}", flush=True)
                break

        if progress and (n % progress_every == 0 or n == maxIter - 1):
            elapsed = time.time() - t0
            it_time = time.time() - it_start
            mv_now = moving_var[n] if n >= m else float('nan')
            print(f"iter {n+1:4d}/{maxIter} | ESS={ESS:,.1f} | L2={L2norm[n]:.3e} | mv={mv_now:.3e} | dt={it_time:.2f}s | elapsed={elapsed/60:.1f}min", flush=True)

    if progress:
        total = time.time() - t0
        print(f"SMC finished in {total/60:.2f} min; iter_stop={iter_stop+1}", flush=True)

    return x1, x2, W, iter_stop, moving_var, L2norm
