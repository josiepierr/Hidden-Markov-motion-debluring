import numpy as np
from pathlib import Path
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt

try:
    # optional, for fast/separable 1D filtering
    from scipy.ndimage import gaussian_filter1d
except Exception as e:
    gaussian_filter1d = None


def _box_filter1d_reflect(img: np.ndarray, axis: int, radius: int) -> np.ndarray:
    """
    Simple O(HW*radius) box filter fallback (reflect padding) if SciPy is unavailable.
    radius=0 => identity.
    """
    if radius <= 0:
        return img.copy()

    pad = radius
    if axis == 1:
        padded = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
        out = np.empty_like(img, dtype=np.float64)
        win = 2 * radius + 1
        for j in range(img.shape[1]):
            out[:, j] = np.sum(padded[:, j:j+win], axis=1) / win
        return out
    elif axis == 0:
        padded = np.pad(img, ((pad, pad), (0, 0)), mode="reflect")
        out = np.empty_like(img, dtype=np.float64)
        win = 2 * radius + 1
        for i in range(img.shape[0]):
            out[i, :] = np.sum(padded[i:i+win, :], axis=0) / win
        return out
    else:
        raise ValueError("axis must be 0 or 1")


def _gaussian_filter1d_reflect(img: np.ndarray, axis: int, sigma: float) -> np.ndarray:
    """
    Uses scipy.ndimage.gaussian_filter1d if available, otherwise a crude fallback
    via repeated box filters (approximation).
    """
    if sigma <= 0:
        return img.copy()

    if gaussian_filter1d is not None:
        return gaussian_filter1d(img, sigma=sigma, axis=axis, mode="reflect")

    # Fallback: approximate Gaussian by 3 box filters (very rough)
    # Choose radius ~ sigma
    r = max(1, int(round(sigma)))
    out = img.astype(np.float64, copy=True)
    for _ in range(3):
        out = _box_filter1d_reflect(out, axis=axis, radius=r)
    return out


def motion_blur_forward(f: np.ndarray, b: float, sigma: float) -> np.ndarray:
    """
    Forward operator A f: blur sharp image f into hhat.

    Model: horizontal uniform motion blur of length b (pixels),
           then vertical Gaussian blur with std sigma (pixels).
    """
    if b < 0:
        raise ValueError("b must be >= 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    # horizontal box half-length
    radius = int(np.floor(b / 2.0))
    tmp = _box_filter1d_reflect(f.astype(np.float64), axis=1, radius=radius)

    # vertical gaussian
    out = _gaussian_filter1d_reflect(tmp, axis=0, sigma=sigma)
    return out


def motion_blur_adjoint(r: np.ndarray, b: float, sigma: float) -> np.ndarray:
    """
    Adjoint operator A^T r.

    For this symmetric blur (box + Gaussian, both symmetric with reflect padding),
    A^T is well-approximated by applying the same blur again.
    """
    return motion_blur_forward(r, b=b, sigma=sigma)


def ems_deblur(
    h: np.ndarray,
    b: float,
    sigma: float,
    epsilon: float,
    n_iter: int,
    f0: np.ndarray = None,
    tiny: float = 1e-10,
    renormalize: bool = True,
):
    """
    Deterministic EMS deblurring.

    EM/RL step:
      f <- f * (A^T ( h / (A f + tiny) )) / (A^T 1)
    Smoothing step:
      f <- S_epsilon(f)  (Gaussian smoothing, std=epsilon in pixels)

    Parameters
    ----------
    h : observed blurred image (nonnegative)
    b, sigma : blur parameters in pixel units (match your blurred_image)
    epsilon : smoothing strength (pixels). Set 0 for pure RL/EM.
    n_iter : number of iterations
    f0 : initialization (default: uniform / normalized h)
    tiny : numerical stabilizer
    renormalize : if True, keep sum(f)=sum(h) (or sum=1 if you normalize h)

    Returns
    -------
    f : reconstructed sharp image
    history : dict with per-iter diagnostics
    """
    h = np.maximum(h.astype(np.float64), 0.0)

    if f0 is None:
        # sensible default: start from h (or uniform); ensure strictly positive
        f = np.maximum(h, 0.0) + tiny
    else:
        f = np.maximum(f0.astype(np.float64), 0.0) + tiny

    # Optional: keep the same total mass as h
    target_mass = float(np.sum(h)) if renormalize else None
    if renormalize and target_mass > 0:
        f *= target_mass / float(np.sum(f))

    # Precompute A^T 1 (normalization term)
    ones = np.ones_like(h, dtype=np.float64)
    At1 = motion_blur_adjoint(ones, b=b, sigma=sigma)
    At1 = np.maximum(At1, tiny)

    history = {
        "fix_point": [],
        "f": []
    }

    prev_f = f.copy()

    for it in range(1, n_iter + 1):
        Af = motion_blur_forward(f, b=b, sigma=sigma)
        ratio = h / np.maximum(Af, tiny)                 # h / (A f)
        backproj = motion_blur_adjoint(ratio, b=b, sigma=sigma)  # A^T( h/(Af) )

        # EM / RL multiplicative update
        f = f * (backproj / At1)

        # Enforce nonnegativity
        f = np.maximum(f, 0.0)

        # Smoothing step (EMS)
        if epsilon > 0:
            # isotropic smoothing: Gaussian in x and y with std=epsilon
            f = _gaussian_filter1d_reflect(f, axis=0, sigma=epsilon)
            f = _gaussian_filter1d_reflect(f, axis=1, sigma=epsilon)
            f = np.maximum(f, 0.0)

        # Renormalize mass if desired
        if renormalize and target_mass and target_mass > 0:
            s = float(np.sum(f))
            if s > 0:
                f *= target_mass / s

        # Diagnostics
        hat_h = motion_blur_forward(f, b=b, sigma=sigma)
        data_mse = float(np.mean((hat_h - h) ** 2))
        delta_f = float(np.linalg.norm(f - prev_f) / (np.linalg.norm(prev_f) + tiny))
        history["fix_point"].append(data_mse)
        history["f"].append(f.copy())

        prev_f = f.copy()

    return f, history

def main():
    current_dir = Path(__file__).parent
    orig_filename = "cactus.png"
    blur_filename = "blurred_cactus.png"
    orig_path = current_dir / "original_image" / orig_filename
    blur_path = current_dir / "blurred_image" / blur_filename
    
    b=128
    sigma=0.02
    epsilon = 1e-3
    n_iter = 50

    # Load image
    image = io.imread(orig_path)
    image = img_as_float(image)
    if image.ndim == 3:
        image = image[:, :, 0]  # Convert to grayscale

    print(f"Original image shape: {image.shape}")
    # Import blurred image
    image_h = io.imread(blur_path)
    image_h = img_as_float(image_h)
    if image_h.ndim == 3:
        image_h = image_h[:, :, 0]  # Convert to grayscale 

    # Run EMS deblurring
    f_rec, history = ems_deblur(
        h=image_h,
        b=b,
        sigma=sigma,
        epsilon=epsilon,
        n_iter=n_iter
    )
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Blurred Image")
    plt.imshow(image_h, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Image (EMS)")
    plt.imshow(f_rec, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
    
