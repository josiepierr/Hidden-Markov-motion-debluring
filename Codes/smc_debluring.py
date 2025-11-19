import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import os

# ==============================
# 1. Paramètres généraux
# ==============================

# Chemin de ton image (300 x 600 déjà redimensionnée)
INPUT_IMAGE_PATH =  r"..\Images\yes.png"   

# Taille cible (comme dans l'article : 300 x 600)
H_img, W_img =128, 128

# Paramètres du motion blur et du bruit
MOTION_LENGTH = 31       # longueur de la traînée horizontale
NOISE_STD = 0.02         # écart-type du bruit gaussien ajouté

# Paramètres SMC
N_PARTICLES = 5000       # comme l'article
N_ITER = 100             # comme Figure 9 (tu peux réduire pour tester)
SMOOTH_SIGMA = 3.0       # sigma du noyau Gaussien pour K (déplacement des particules)

# Pour reproductibilité
np.random.seed(42)

# ==============================
# 2. Utilitaires image
# ==============================

def load_grayscale_image(path, size=(H_img, W_img)):
    """Charge une image, convertit en niveaux de gris, redimensionne et normalise dans [0,1]."""
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def save_image(arr, path):
    """Enregistre un tableau 2D [0,1] en image PNG."""
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)

# ==============================
# 3. PSF : motion blur horizontal
# ==============================

def make_motion_psf(length=31):
    """
    Noyau de flou de mouvement horizontal.
    PSF de taille (length x length), ligne horizontale au centre.
    """
    psf = np.zeros((length, length), dtype=np.float32)
    psf[length // 2, :] = 1.0 / length
    return psf

PSF_SIZE = MOTION_LENGTH
PSF = make_motion_psf(PSF_SIZE)

def forward_model(f_true, psf):
    """
    Applique la convolution 2D (flou) avec le PSF.
    Utilise une FFT pour être O(HW log HW).
    """
    blurred = fftconvolve(f_true, psf, mode="same")
    return blurred

# ==============================
# 4. Simulation de l'observation h (image floue + bruit)
# ==============================

def simulate_observation(f_true, psf, noise_std=0.02):
    """Construit l'image floue + bruit gaussien (comme h dans l'article)."""
    h_clean = forward_model(f_true, psf)
    noise = noise_std * np.random.randn(*h_clean.shape).astype(np.float32)
    h_noisy = h_clean + noise
    h_noisy = np.clip(h_noisy, 0.0, 1.0)
    return h_noisy

# ==============================
# 5. SMC utils : g(y|x), ESS, resampling, reconstruction
# ==============================

def g_y_given_x_psf(Y_int, X, psf):
    """
    Calcule g(Y|X) = PSF(Y - X) pour un vecteur de N particules.
    - X : array (N, 2)   positions "nettes"
    - Y_int : array (N, 2) positions observées (pixels)
    - psf : noyau local (PSF_SIZE x PSF_SIZE)
    Retour : array (N,)
    """
    N = X.shape[0]
    # On approxime les positions de X sur la grille par arrondi
    X_int = np.rint(X).astype(int)

    dy = Y_int[:, 0] - X_int[:, 0]
    dx = Y_int[:, 1] - X_int[:, 1]

    center = psf.shape[0] // 2  # on suppose PSF carré
    py = dy + center
    px = dx + center

    # Masque des indices valides dans le support du PSF
    valid = (
        (py >= 0) & (py < psf.shape[0]) &
        (px >= 0) & (px < psf.shape[1])
    )

    vals = np.zeros(N, dtype=np.float32)
    vals[valid] = psf[py[valid], px[valid]]

    # Pour éviter les zéros stricts
    eps = 1e-12
    return vals + eps

def compute_ESS(weights):
    """Effective Sample Size : ESS = 1 / sum(w^2)."""
    return 1.0 / np.sum(weights ** 2)

def systematic_resampling(weights):
    """Resampling systématique (variance plus faible que multinomial)."""
    N = len(weights)
    positions = (np.arange(N) + np.random.rand()) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # sécurité
    idx = np.searchsorted(cumsum, positions)
    return idx

def reconstruct_f_from_particles(X, H, W, smooth_sigma=0.0):
    """
    Reconstruit f(x) (image nette) à partir des positions de particules.
    - X : (N, 2)  positions particules
    - H, W : dimensions de l'image
    - smooth_sigma : si > 0, lissage par noyau gaussien.
    """
    img = np.zeros((H, W), dtype=np.float32)
    y_int = np.clip(np.rint(X[:, 0]).astype(int), 0, H - 1)
    x_int = np.clip(np.rint(X[:, 1]).astype(int), 0, W - 1)

    # On ajoute 1 à chaque pixel touché par une particule
    np.add.at(img, (y_int, x_int), 1.0)
    if img.sum() > 0:
        img /= img.sum()

    if smooth_sigma > 0:
        # Noyau gaussien séparé simple
        radius = int(3 * smooth_sigma)
        ax = np.arange(-radius, radius + 1)
        gauss = np.exp(-0.5 * (ax / smooth_sigma) ** 2)
        gauss /= gauss.sum()
        kernel = np.outer(gauss, gauss)
        img = fftconvolve(img, kernel, mode="same")
        img = np.clip(img, 0.0, None)
        if img.sum() > 0:
            img /= img.sum()

    return img

# ==============================
# 6. Algorithme SMC principal
# ==============================

def smc_motion_deblurring(h_obs, psf, n_particles=5000, n_iter=100, smooth_sigma=3.0):
    """
    Implémente l'Algorithme 1 de l'article dans le cadre du motion deblurring.
    h_obs : image floue observée (H x W)
    psf   : noyau de flou (PSF_SIZE x PSF_SIZE)
    """
    H, W = h_obs.shape
    N = n_particles

    # 1) Domaines pour X (image nette) et Y (image observée)
    #    -> ici X et Y sont tous les pixels de l'image 2D.
    # Liste de toutes les coordonnées de pixels
    coords_y = np.array([(i, j) for i in range(H) for j in range(W)], dtype=int)
    h_flat = h_obs.flatten().astype(np.float64)
    h_flat = np.clip(h_flat, 1e-12, None)
    h_flat /= h_flat.sum()

    # 2) Initialisation f_1 : uniforme sur X = [0, H-1] x [0, W-1]
    X = np.zeros((N, 2), dtype=np.float32)
    X[:, 0] = np.random.uniform(0, H, size=N)
    X[:, 1] = np.random.uniform(0, W, size=N)

    # Kernel de déplacement K (Gaussian)
    def move_particles(X_prev, sigma=SMOOTH_SIGMA):
        noise = sigma * np.random.randn(*X_prev.shape)
        X_new = X_prev + noise
        X_new[:, 0] = np.clip(X_new[:, 0], 0, H - 1)
        X_new[:, 1] = np.clip(X_new[:, 1], 0, W - 1)
        return X_new

    f_est = reconstruct_f_from_particles(X, H, W, smooth_sigma=smooth_sigma)

    for it in range(1, n_iter + 1):
        print(f"Iteration {it}/{n_iter}")

        # 3) (Ligne 2 de l'algorithme) :
        #    - X_n ~ K(X_{n-1}, .)
        #    - Y_n échantillonné depuis l'empirique de h (ici, h_obs)
        X = move_particles(X, sigma=SMOOTH_SIGMA)

        # Y_n : on tire N pixels selon la distribution h
        idx_y = np.random.choice(len(coords_y), size=N, p=h_flat)
        Y_int = coords_y[idx_y]

        # 4) approx h_n(y) = (f_n * g)(y) via convolution
        h_approx = forward_model(f_est, psf)
        h_approx = np.clip(h_approx, 1e-12, None)
        h_approx /= h_approx.sum()
        h_approx_flat = h_approx.flatten()

        # 5) g(Y|X) et h_n(Y) pour calculer G_n
        g_vals = g_y_given_x_psf(Y_int, X, psf)
        h_vals = h_approx_flat[idx_y]

        G_tilde = g_vals / h_vals
        G_tilde = np.clip(G_tilde, 1e-12, None)

        # Poids normalisés
        w = G_tilde.astype(np.float64)
        w /= w.sum()

        # 6) ESS + resampling adaptatif (comme dans le tuto "particles")
        ESS = compute_ESS(w)
        # print(f"  ESS = {ESS:.1f} / {N}")
        if ESS < N / 2:
            idx = systematic_resampling(w)
            X = X[idx]
            Y_int = Y_int[idx]
            w = np.full(N, 1.0 / N, dtype=np.float64)

        # 7) Reconstruction de f_{n+1} (densité sur X, donc image)
        f_est = reconstruct_f_from_particles(X, H, W, smooth_sigma=smooth_sigma)

    return f_est

# ==============================
# 7. Main : tout lancer et sauvegarder les images
# ==============================

if __name__ == "__main__":

    # --- Charger image nette ---
    f_true = load_grayscale_image(INPUT_IMAGE_PATH, size=(H_img, W_img))

    # --- Simuler observation floue + bruit ---
    h_obs = simulate_observation(f_true, PSF, noise_std=NOISE_STD)

    # --- Lancer SMC ---
    f_rec = smc_motion_deblurring(
        h_obs,
        PSF,
        n_particles=N_PARTICLES,
        n_iter=N_ITER,
        smooth_sigma=SMOOTH_SIGMA,
    )

    # --- Sauvegarder les trois images ---
    os.makedirs("Results", exist_ok=True)
    save_image(f_true, "Results/true_image.png")
    save_image(h_obs, "Results/blurred_noisy.png")
    save_image(f_rec, "Results/smc_reconstruction.png")

    # Petit affichage rapide
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Référence (f)")
    plt.imshow(f_true, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Floue + bruit (h)")
    plt.imshow(h_obs, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Reconstruction SMC")
    plt.imshow(f_rec, cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.show()
