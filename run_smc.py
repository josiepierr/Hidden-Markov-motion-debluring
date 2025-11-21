import numpy as np
from scipy.stats import norm
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt


from blur import blurred_image, multiplicative_noise
from kde_smoothing import reconstruct_image_from_particles
from smc_alg import SMCAlgorithm

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    b = 128  # speed of constant motion
    sigma = 0.02  # standard deviation of normal
    n_iter = 100  # number of time steps
    n_particles = 5000  # number of particles
    epsilon = 1e-3  # smoothing parameter
    
    print("SMC Motion Deblurring - Python Implementation")
    print(f"Parameters: b={b}, sigma={sigma}, N={n_particles}, iterations={n_iter}")
    print("\nThis is a translation of the MATLAB code from the Fredholm equation paper.")
    print("To use: load an image, blur it, add noise, then run SMC deblurring.")

    # Load image
    image = io.imread('BC.png')
    image = img_as_float(image)
    if image.ndim == 3:
        image = image[:, :, 0]  # Convert to grayscale
    
    print(f"Original image shape: {image.shape}")
    # Import blurred image
    image_h = io.imread('BCblurred.png')
    image_h = img_as_float(image_h)
    if image_h.ndim == 3:
        image_h = image_h[:, :, 0]  # Convert to grayscale
    # Add multiplicative noise
    # image_h_noisy = multiplicative_noise(image_h, alpha=0.1, beta=0.1)

    # Run SMC deblurring
    smc_algo = SMCAlgorithm(
        n_particles=n_particles,
        image=image_h,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon
    )
    x_particles, y_particles, weights, ess_history = smc_algo.run()
    print("SMC deblurring completed.")
    
    # Plot ESS history
    plt.figure()
    plt.plot(ess_history)
    plt.title('Effective Sample Size over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('ESS')
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