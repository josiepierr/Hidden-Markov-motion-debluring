import numpy as np

def pinky(xi, phi, dist_t, N):
    dist_t = np.asarray(dist_t, dtype=np.float64)
    dist_t = np.clip(dist_t, 0.0, None)

    col = np.sum(dist_t, axis=0)
    col_sum = np.sum(col)
    if col_sum <= 0:
        col = np.ones_like(col)
        col_sum = np.sum(col)
    col = col / col_sum

    cdf_col = np.cumsum(col)
    u1 = np.random.rand(N)
    idx_x = np.searchsorted(cdf_col, u1, side='right')
    idx_x = np.clip(idx_x, 0, len(xi) - 1)

    denom = np.sum(dist_t[:, idx_x], axis=0)
    denom = np.where(denom > 0, denom, 1.0)
    cond = dist_t[:, idx_x] / denom
    cdf_row = np.cumsum(cond, axis=0)

    u2 = np.random.rand(N)
    row_idx = (cdf_row >= u2).argmax(axis=0)

    y = np.empty((N, 2), dtype=np.float64)
    y[:, 0] = np.asarray(xi, dtype=np.float64)[idx_x]
    y[:, 1] = np.asarray(phi, dtype=np.float64)[row_idx]
    return y
