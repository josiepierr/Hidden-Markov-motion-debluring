import numpy as np
from scipy.stats import norm
from skimage.util import img_as_float
import argparse
import os
from skimage.io import imread, imsave
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d
from skimage import img_as_ubyte
from pathlib import Path


def blurred_image(image_f, b, sigma):
    """
    Fast blur: separable convolution (uniform horizontal + Gaussian vertical)
    """

    # Convert to grayscale and float
    if image_f.ndim == 3:
        image_f = image_f[:, :, 0]
    image_f = img_as_float(image_f)

    H, W = image_f.shape

    # --- 1) Horizontal uniform blur kernel ---
    # b is in pixels now (you were normalizing, but that is unnecessary)
    half = int(b // 2)
    if half < 1:
        half = 1
    uniform_kernel = np.ones(2 * half + 1)
    uniform_kernel /= uniform_kernel.sum()

    # --- 2) Vertical Gaussian blur ---
    # sigma is already given in “pixel units”
    # We apply Gaussian only vertically (axis=0)
    vert_gauss = gaussian_filter1d(image_f, sigma=sigma, axis=0, mode='nearest')

    # --- 3) Horizontal convolution with uniform kernel ---
    blurred = convolve2d(
        vert_gauss,
        uniform_kernel[np.newaxis, :],   # horizontal 1D kernel
        mode='same',
        boundary='symm'
    )

    # Normalize to [0,1]
    blurred = blurred - blurred.min()
    blurred = blurred / blurred.max()

    return blurred


def uniform_pdf(x, a, b):
    """
    Uniform probability density function
    
    Parameters:
    -----------
    x : ndarray
        Input values
    a : float
        Lower bound
    b : float
        Upper bound
    
    Returns:
    --------
    pdf : ndarray
        PDF values
    """
    pdf = np.zeros_like(x)
    mask = (x >= a) & (x <= b)
    pdf[mask] = 1.0 / (b - a)
    return pdf


def multiplicative_noise(image, alpha, beta):
    """
    Multiplicative noise for perfectly observed images
    Based on Lee and Vardi 1994
    
    Parameters:
    -----------
    image : ndarray
        Noise-free image
    alpha : float
        Amount of noise
    beta : float
        Probability of adding noise
    
    Returns:
    --------
    noisy_image : ndarray
        Noisy image
    """
    pixels = image.shape
    unif = np.random.rand(*pixels)
    
    mult_matrix = np.ones(pixels)
    mult_matrix[unif <= beta/2] = 1 - alpha
    mult_matrix[(unif > beta/2) & (unif <= 1 - beta/2)] = 1
    mult_matrix[unif > 1 - beta/2] = 1 + alpha
    
    return image * mult_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur an image and add multiplicative noise.")
    parser.add_argument("--image_name", type=str, required=True,
                        help="Name of the input image file (inside original_image folder).")
    parser.add_argument("--b", type=float, required=True,
                        help="Horizontal velocity for blur.")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Vertical blur width (Gaussian sigma).")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Noise magnitude.")
    parser.add_argument("--beta", type=float, required=True,
                        help="Noise probability.")

    args = parser.parse_args()
    
    current_dir = Path(__file__).parent
    orig_folder = current_dir / "original_image"
    input_path = orig_folder / args.image_name
    print(f"Loading image from {input_path}")
    image = imread(input_path)
    print(f"Image shape: {image.shape}")

    # Apply blur
    blurred = blurred_image(image, args.b, args.sigma)
    print("Image blurred.")

    # Apply multiplicative noise
    noisy = multiplicative_noise(blurred, args.alpha, args.beta)
    print("Multiplicative noise added.")

    # Normalize to [0,1]
    noisy = noisy - noisy.min()
    noisy = noisy / noisy.max()

    # Convert float64 → uint8 so PNG accepts it
    noisy_uint8 = img_as_ubyte(noisy)

    # Save output in same folder
    blur_folder = current_dir / "blurred_image"
    output_path = blur_folder / f"blurred_{args.image_name}"
    imsave(output_path, noisy_uint8)

    print(f"Saved result to {output_path}")
