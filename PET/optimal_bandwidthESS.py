import numpy as np
import numba

@numba.jit(nopython=True)
def _weighted_std(x, W):
    m = np.sum(x * W)
    v = np.sum(W * (x - m) ** 2)
    return np.sqrt(v / np.sum(W))

@numba.jit(nopython=True)
def optimal_bandwidthESS(x, W):
    ESS = np.sum(W ** 2)
    s = _weighted_std(x, W)
    return 1.06 * s * (ESS ** 0.2)
