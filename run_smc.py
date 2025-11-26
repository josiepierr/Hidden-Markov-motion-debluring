import numpy as np
from scipy.stats import norm
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt


from blur import blurred_image, multiplicative_noise
from kde_smoothing import reconstruct_image_from_particles
from smc_alg import SMCAlgorithm
import argparse
from pathlib import Path

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="SMC Motion Deblurring")

    parser.add_argument("--b", type=float, required=True,
                        help="Horizontal motion speed for the blur model.")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Gaussian blur standard deviation.")
    parser.add_argument("--n_iter", type=int, required=True,
                        help="Number of SMC iterations.")
    parser.add_argument("--n_particles", type=int, required=True,
                        help="Number of particles used in SMC.")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="Smoothing parameter for reconstruction.")
    parser.add_argument("--orig_file", type=str, required=True,
                        help="Name of the original (sharp) image file.")
    parser.add_argument("--blur_file", type=str, required=True,
                        help="Name of the blurred image file.")
    
    args = parser.parse_args()

    # Parameters
    b = args.b  # speed of constant motion
    sigma = args.sigma  # standard deviation of normal
    n_iter = args.n_iter  # number of time steps
    n_particles = args.n_particles  # number of particles
    epsilon = args.epsilon  # smoothing parameter
    orig_filename = args.orig_file
    blur_filename = args.blur_file

    current_dir = Path(__file__).parent
    orig_path = current_dir / "original_image" / orig_filename
    blur_path = current_dir / "blurred_image" / blur_filename
    
    print("SMC Motion Deblurring - Python Implementation")
    print(f"Parameters: b={b}, sigma={sigma}, N={n_particles}, iterations={n_iter}")
    print("\nThis is a translation of the MATLAB code from the Fredholm equation paper.")
    print("To use: load an image, blur it, add noise, then run SMC deblurring.")

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

    # Run SMC deblurring
    smc_algo = SMCAlgorithm(
        n_particles=n_particles,
        image=image_h,
        original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=10
    )
    x_particles, y_particles, weights, ess_history, reconstruction_errors = smc_algo.run()
    print("SMC deblurring completed.")
    
    # Plot ESS history
    plt.figure()
    plt.plot(ess_history)
    plt.title('Effective Sample Size over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('ESS')
    plt.grid()
    plt.show()

    plt.figure()
    iterations, errors = zip(*reconstruction_errors)
    plt.plot(iterations, errors, marker='o')
    plt.title('Reconstruction Error over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error (L2 Norm)')
    plt.grid()
    plt.show()

    # Reconstruct image from final particles
    reconstructed_image = reconstruct_image_from_particles(
        x_particles[-1, :],  # Last iteration x coordinates
        y_particles[-1, :],  # Last iteration y coordinates  
        weights[-1, :],      # Last iteration weights
        image.shape,
        epsilon
    )
    # Display original, blurred, and reconstructed images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(image_h, cmap='gray', vmin=0, vmax=1)
    plt.title('Blurred Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Reconstructed Image (SMC)')
    plt.axis('off')
    plt.show()