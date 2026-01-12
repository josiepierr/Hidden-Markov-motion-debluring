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

def fix_point_residual_plot(n_particles, image_b, image_original, sigma, n_iter, b, epsilon, save_every):

    seeds = np.arange(1, 11)
    fix_point_errors_all = []
    errors_all = []
    iterations = None
    for seed in seeds:
        np.random.seed(seed)
        smc_algo = SMCAlgorithm(
            n_particles=n_particles,
            image=image_b,
            original=image_original,
            sigma=sigma,
            n_iter=n_iter,
            b=b,
            epsilon=epsilon,
            save_every=save_every
        )
        x_particles, y_particles, weights, ess_history, reconstruction_errors, fix_point_errors = smc_algo.run()
        fix_p_errors = [err for (_, err) in fix_point_errors]
        errors_all.append([err for (_, err) in reconstruction_errors])
        fix_point_errors_all.append(fix_p_errors)
        if iterations is None:
            iterations = [it for (it, _) in fix_point_errors]

    errors_all = np.array(errors_all)  # shape (n_seeds, n_records)
    mean_errors = np.mean(errors_all, axis=0)
    std_errors = np.std(errors_all, axis=0)
    # Reconstruction error plot (mean+std over seeds)
    plt.figure()
    plt.errorbar(iterations, mean_errors, yerr=std_errors, marker='o', capsize=5)
    plt.title('Reconstruction Error over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error (L2 Norm)')
    plt.grid()
    plt.show()
    
    fix_point_errors_all = np.array(fix_point_errors_all)  # shape (n_seeds, n_records)
    mean_fix_point_errors = np.mean(fix_point_errors_all, axis=0)
    std_fix_point_errors = np.std(fix_point_errors_all, axis=0)
    # Fix-point residual plot (mean+std over seeds)
    plt.figure()
    plt.errorbar(iterations, mean_fix_point_errors, yerr=std_fix_point_errors, marker='o', capsize=5)
    plt.title('Fix-point Residual over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Fix-point Residual (L2 Norm)')
    plt.grid()
    plt.show()

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
    parser.add_argument("--save_every", type=int, default=10,
                        help="Frequency of saving reconstruction errors.")
    
    args = parser.parse_args()

    # Parameters
    b = args.b  # speed of constant motion
    sigma = args.sigma  # standard deviation of normal
    n_iter = args.n_iter  # number of time steps
    n_particles = args.n_particles  # number of particles
    epsilon = args.epsilon  # smoothing parameter
    orig_filename = args.orig_file
    blur_filename = args.blur_file
    save_every = args.save_every

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

    fix_point_residual_plot(
        n_particles=n_particles,
        image_b=image_h,
        image_original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=save_every
    )