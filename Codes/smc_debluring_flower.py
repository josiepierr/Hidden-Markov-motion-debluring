"""
SMC-EMS deblurring simplifié sur une image (par ex. ta fleur).

Dépendances :
    pip install numpy matplotlib pillow scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve

# ============================================================
# 1. Chargement et préparation de l'image nette
# ============================================================

# Chemin vers ton image (adapte selon ton arborescence)
# Exemple si ton image est dans ../Images/image1.webp :
INPUT_IMAGE_PATH = r"..\Images\image1.webp"

# Taille de travail (plus petit = plus rapide)
H_img, W_img = 128, 128


def load_grayscale_image(path, size=(H_img, W_img)):
    img = Image.open(path).convert("L")      # niveaux de gris
    img = img.resize(size, Image.BICUBIC)    # redimensionner
    img_np = np.array(img).astype(np.float32)
    img_np /= 255.0                          # normaliser dans [0,1]
    return img_np


f_true = load_grayscale_image(INPUT_IMAGE_PATH, size=(H_img, W_img))

# ============================================================
# 2. PSF de motion blur + génération de l'image floutée
# ============================================================

def motion_blur_psf(length, angle_deg, size):
    """
    PSF simple de flou de mouvement :
    - length : longueur du flou (en pixels)
    - angle_deg : angle du flou (0 = horizontal)
    - size : taille de la fenêtre PSF (impair, ex: 31)
    """
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    angle = np.deg2rad(angle_deg)
    dx = np.cos(angle)
    dy = np.sin(angle)

    # On dessine un segment discret centré
    for i in range(length):
        t = i - length / 2.0
        x = center + int(round(dx * t))
        y = center + int(round(dy * t))
        if 0 <= y < size and 0 <= x < size:
            psf[y, x] = 1.0

    # Normalisation
    psf /= psf.sum() + 1e-12
    return psf


def blur_image(img, psf):
    return fftconvolve(img, psf, mode="same")


# Paramètres du flou
BLUR_LENGTH = 15     # longueur du flou en pixels
BLUR_ANGLE = 0       # 0 = horizontal
PSF_SIZE = 31        # taille de la fenêtre PSF

psf = motion_blur_psf(BLUR_LENGTH, BLUR_ANGLE, PSF_SIZE)
h_clean = blur_image(f_true, psf)

# Ajout d'un léger bruit
NOISE_STD = 0.01
noise = np.random.normal(0, NOISE_STD, size=h_clean.shape).astype(np.float32)
h_obs = np.clip(h_clean + noise, 0.0, 1.0)

# ============================================================
# 3. Algorithme SMC–EMS simplifié
# ============================================================

# Paramètres SMC
N_PARTICLES = 5000
N_ITER = 30
K_SIGMA = 1.0   # écart-type de la mutation gaussienne (kernel K)
G_SIGMA = 2.0   # écart-type de g(y|x) utilisé dans les poids


def init_particles(N, H, W):
    """
    Initialise les particules X_0 de façon uniforme sur l'image.
    X : tableau (N, 2) avec (row, col).
    """
    ys = np.random.uniform(0, H - 1, size=N)
    xs = np.random.uniform(0, W - 1, size=N)
    return np.stack([ys, xs], axis=1).astype(np.float32)


X = init_particles(N_PARTICLES, H_img, W_img)

# ------------------------------------------------------------
# 3.1 Distribution empirique de h_obs pour tirer les Y
# ------------------------------------------------------------

coords_y = np.array([(i, j) for i in range(H_img) for j in range(W_img)], dtype=np.int32)
values_y = h_obs.flatten()
values_y_sum = values_y.sum() + 1e-12
prob_y = values_y / values_y_sum


def sample_Y(N):
    """Tire N pixels Y selon la densité empirique h_obs."""
    idx = np.random.choice(len(coords_y), size=N, p=prob_y)
    return coords_y[idx].astype(np.float32)  # (N, 2)


# ------------------------------------------------------------
# 3.2 Fonction g(y|x) simplifiée (gaussienne en distance)
# ------------------------------------------------------------

def g_y_given_x(Y, X, sigma=G_SIGMA):
    """
    g(Y|X) ~ densité gaussienne en fonction de la distance entre X et Y.
    Y, X : tableaux (N, 2)
    Retourne un vecteur (N,)
    """
    diff = Y - X
    dist2 = np.sum(diff * diff, axis=1)
    return np.exp(-0.5 * dist2 / (sigma**2))


# ------------------------------------------------------------
# 3.3 Estimation de h_n(Y_i) par les particules
# ------------------------------------------------------------

def estimate_h_n_at_Y(Y, X, sigma=G_SIGMA):
    """
    h_n(Y_i) ≈ (1/N) ∑_j g(Y_i | X_j)
    Y : (N, 2)
    X : (N, 2)
    Retourne un vecteur (N,)
    NB: O(N^2), à garder N raisonnable
    """
    Np = X.shape[0]
    h_vals = np.zeros(Np, dtype=np.float32)
    for i in range(Np):
        yi = Y[i]                # (2,)
        diff = X - yi            # (N, 2)
        dist2 = np.sum(diff * diff, axis=1)
        gi = np.exp(-0.5 * dist2 / (sigma**2))  # (N,)
        h_vals[i] = gi.mean()
    return h_vals


# ------------------------------------------------------------
# 3.4 Mutation des particules (kernel K)
# ------------------------------------------------------------

def mutate_particles(X, sigma=K_SIGMA):
    noise = np.random.normal(0, sigma, size=X.shape).astype(np.float32)
    X_new = X + noise
    # On reste dans l'image
    X_new[:, 0] = np.clip(X_new[:, 0], 0, H_img - 1)
    X_new[:, 1] = np.clip(X_new[:, 1], 0, W_img - 1)
    return X_new


# ------------------------------------------------------------
# 3.5 Resampling multinomial
# ------------------------------------------------------------

def resample_particles(X, weights):
    """
    Resampling multinomial à partir des poids 'weights'.
    """
    Np = X.shape[0]
    indices = np.random.choice(Np, size=Np, p=weights)
    return X[indices]


# ------------------------------------------------------------
# 3.6 Reconstruction d'image à partir des particules
# ------------------------------------------------------------

def reconstruct_image_from_particles(X, H, W, smooth=False):
    """
    Reconstruit une image (H, W) en histogrammant les particules X.
    """
    H = int(H)
    W = int(W)
    img = np.zeros((H, W), dtype=np.float32)

    ys = np.clip(np.round(X[:, 0]).astype(int), 0, H - 1)
    xs = np.clip(np.round(X[:, 1]).astype(int), 0, W - 1)

    for y, x in zip(ys, xs):
        img[y, x] += 1.0

    img /= img.max() + 1e-12
    return img


# ============================================================
# 3.7 Boucle principale SMC
# ============================================================

history_f = []

for n in range(N_ITER):
    print(f"Iteration {n+1}/{N_ITER}")

    # 1) Tirage des Y_n^i depuis l'image floutée (h_obs)
    Y = sample_Y(N_PARTICLES)  # (N, 2)

    # 2) Estimation h_n(Y_i) via les particules X
    h_n_Y = estimate_h_n_at_Y(Y, X, sigma=G_SIGMA)

    # 3) Calcul g(Y_i | X_i)
    g_vals = g_y_given_x(Y, X, sigma=G_SIGMA)

    # 4) Potentiels G_n^i = g / h_n
    G = g_vals / (h_n_Y + 1e-12)

    # 5) Poids normalisés
    weights = G / (G.sum() + 1e-12)

    # 6) Resampling
    X = resample_particles(X, weights)

    # 7) Mutation (kernel K)
    X = mutate_particles(X, sigma=K_SIGMA)

    # 8) Reconstruction de f_n
    f_est = reconstruct_image_from_particles(X, H_img, W_img)
    history_f.append(f_est)

# Dernière estimation
f_smc = history_f[-1]

# ============================================================
# 4. Affichage des résultats
# ============================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(f_true, cmap="gray")
plt.title("Image nette (originale)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(h_obs, cmap="gray")
plt.title("Image floutée (motion blur)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(f_smc, cmap="gray")
plt.title("Reconstruction SMC (approx)")
plt.axis("off")

plt.tight_layout()
plt.show()
