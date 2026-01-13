import numpy as np
import numba

@numba.jit(nopython=True)
def mult_resample(W, N):
    W = W.ravel()
    indices = np.zeros(N, dtype=np.int64)
    s = W[0]
    u = np.sort(np.random.rand(N))
    j = 0
    for i in range(N):
        while s < u[i]:
            j += 1
            s += W[j]
        indices[i] = j
    return indices
