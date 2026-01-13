import numpy as np

def hreconstruction_pet(phi, xi, sigma, KDEx, KDEy):
    KDEx = np.asarray(KDEx, dtype=np.float64)
    KDEy = np.asarray(KDEy, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    xi = np.asarray(xi, dtype=np.float64).ravel()

    dx = KDEx[0, 1] - KDEx[1, 1]
    inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    hatH = np.zeros((xi.size, phi.size), dtype=np.float64)
    for i in range(xi.size):
        xi_val = xi[i]
        for j in range(phi.size):
            ph = phi[j]
            proj = KDEx[:, 0] * np.cos(ph) + KDEx[:, 1] * np.sin(ph) - xi_val
            pdf = inv * np.exp(-0.5 * (proj / sigma) ** 2)
            hatH[i, j] = (dx * dx) * np.sum(pdf * KDEy)
    return hatH
