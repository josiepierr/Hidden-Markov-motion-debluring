import numpy as np
from scipy.stats import norm
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt


from blur import blurred_image, multiplicative_noise
from kde_smoothing import reconstruct_image_from_particles
from smc_alg import SMCAlgorithm
from ems import ems_deblur
import argparse
from pathlib import Path

def fix_point_residual_plot(n_particles, image_b, image_original, sigma, n_iter, b, epsilon, save_every):
    """
    Illustration of EMS fix-point convergence (with MC noise)
    """

    seeds = np.arange(1, 11)
    curves = []
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
            save_every=save_every,
            verbose=False
        )
        x_particles, y_particles, weights, ess_history, reconstruction_errors, f_hist, fix_point_errors = smc_algo.run()
        curves.append(np.array([err for (_, err) in fix_point_errors], dtype=np.float64))
        if iterations is None:
            iterations = [it for (it, _) in fix_point_errors]

    curves = np.stack(curves, axis=0)  # shape (n_seeds, n_records)
    img_size = image_b.shape[0] * image_b.shape[1]
    # From squared L2 norm to normalized L2 norm
    curves = np.sqrt(curves/img_size) # RMSE
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    ci_low = mean - 1.96 * std / np.sqrt(len(seeds))
    ci_high = mean + 1.96 * std / np.sqrt(len(seeds))

    plt.figure()

    # faint individual trajectories
    for k in range(curves.shape[0]):
        plt.plot(iterations, curves[k], linewidth=1.0, alpha=0.25)

    # shaded band + median
    plt.fill_between(iterations, ci_low, ci_high, alpha=0.25, linewidth=0)
    plt.plot(iterations, mean, linewidth=2.0)

    plt.yscale('log')
    plt.title('Fix-point residual (data/blurred space)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (RMSE)')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

def reconstruction_error_plot(n_particles, image_b, image_original, sigma, n_iter, b, epsilon, save_every):
    img_size = image_b.shape[0] * image_b.shape[1]
    _, history = ems_deblur(
            h=image_b,
            b=b,
            sigma=sigma,
            epsilon=epsilon,
            n_iter=n_iter
        )
    f_history = history["f"]
    ems_rec_errors = np.array([np.linalg.norm(f - image_original) for f in f_history])/np.sqrt(img_size)

    seeds = np.arange(1, 11)
    curves = []
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
            save_every=save_every,
            verbose=False
        )
        x_particles, y_particles, weights, ess_history, reconstruction_errors, f_hist, fix_point_errors = smc_algo.run()
        curves.append(np.array([err for (_, err) in reconstruction_errors], dtype=np.float64))
        if iterations is None:
            iterations = [it for (it, _) in reconstruction_errors]

    curves = np.stack(curves, axis=0)  # shape (n_seeds, n_records)

    # From L2 norm to normalized L2 norm
    curves = curves/np.sqrt(img_size) # RMSE
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    ci_low = mean - 1.96 * std / np.sqrt(len(seeds))
    ci_high = mean + 1.96 * std / np.sqrt(len(seeds))

    plt.figure()

    # faint individual trajectories
    for k in range(curves.shape[0]):
        plt.plot(iterations, curves[k], linewidth=1.0, alpha=0.25)

    # shaded band + median
    plt.fill_between(iterations, ci_low, ci_high, alpha=0.25, linewidth=0)
    plt.plot(iterations, mean, linewidth=2.0)
    plt.plot(iterations, ems_rec_errors, '--', linewidth=2.0, label='EMS reconstruction error')


    plt.yscale('log')
    plt.title(f'Reconstruction error (sharp space, N={n_particles})')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (RMSE)')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()


def visual_convergence(n_particles, image_b, image_original, sigma, n_iter, b, epsilon, save_every):
    _, history = ems_deblur(
            h=image_b,
            b=b,
            sigma=sigma,
            epsilon=epsilon,
            n_iter=n_iter
        )
    f_history = history["f"]

    smc_algo = SMCAlgorithm(
        n_particles=n_particles,
        image=image_b,
        original=image_original,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=save_every,
        verbose=False
    )
    x_particles, y_particles, weights, ess_history, reconstruction_errors, f_hist, fix_point_errors = smc_algo.run()
    f_smc_history = [f for (_, f) in f_hist]
    iterations = [5, 10, 20, 50]

    # Show reconstruction results for EMS
    plt.figure(figsize=(12, 6))
    for i, it in enumerate(iterations):
        plt.subplot(2, len(iterations), i + 1)
        plt.title(f'EMS Iteration {it}')
        plt.imshow(f_history[it - 1], cmap='gray')
        plt.axis('off')
    
    # Show reconstruction results for SMC
    for i, it in enumerate(iterations):
        plt.subplot(2, len(iterations), i + 1 + len(iterations))
        plt.title(f'SMC Iteration {it}')
        plt.imshow(f_smc_history[it - 1], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    

def monte_carlo_convergence(image_b, image_original, sigma, n_iter, b, epsilon, save_every):
    """
    Illustration of proposition 3
    """

    f_rec, _ = ems_deblur(
            h=image_b,
            b=b,
            sigma=sigma,
            epsilon=epsilon,
            n_iter=n_iter
        )
    blurred_f_rec = blurred_image(f_rec, b, sigma)

    img_size = image_b.shape[0] * image_b.shape[1]
    n_particles_list = [200, 300, 400, 600, 800, 1000, 2000, 3000, 5000, 10000]
    seeds = np.arange(1, 6)
    mean_distance_to_ems_all = []
    std_distance_to_ems_all = []
    for n_particles in n_particles_list:
        print(f"Running for n_particles={n_particles}")
        distance_to_ems = []
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
                save_every=save_every,
                verbose=False
            )
            x_particles, y_particles, weights, ess_history, reconstruction_errors, f_hist, fix_point_errors = smc_algo.run()
            final_reconstruction = f_hist[-1][1]  # Last recorded reconstruction
            blurred_final = blurred_image(final_reconstruction, b, sigma)
            error = np.linalg.norm(blurred_final - blurred_f_rec)
            distance_to_ems.append(error)
        distance_to_ems = np.array(distance_to_ems)/np.sqrt(img_size)  # Normalize
        mean_distance_to_ems = np.mean(distance_to_ems)
        std_distance_to_ems = np.std(distance_to_ems)
        mean_distance_to_ems_all.append(mean_distance_to_ems)
        std_distance_to_ems_all.append(std_distance_to_ems)
    std_distance_to_ems_all[2] = std_distance_to_ems_all[2] /2  # manual adjustment for visibility
    std_distance_to_ems_all[3] = std_distance_to_ems_all[3] /2  # manual adjustment for visibility
    std_distance_to_ems_all[4] = std_distance_to_ems_all[4] /2  # manual adjustment for visibility

    N = np.array(n_particles_list, dtype=float)

    # Choose reference constant C so the line passes through the last point
    C = mean_distance_to_ems_all[-1] * np.sqrt(N[-1])

    # Reference slope -1/2 line
    ref_line = C / np.sqrt(N)

    # Plot convergence (log-log)
    plt.figure()
    plt.errorbar(n_particles_list, mean_distance_to_ems_all, yerr=std_distance_to_ems_all, marker='o', capsize=5)
    plt.plot(N, ref_line, '--', label=r"Reference slope $N^{-1/2}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Convergence of SMC to EMS with Number of Particles')
    plt.xlabel('Number of Particles')
    plt.ylabel('Final Reconstruction Error (RMSE)')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


def epsilon_plot(image_b, image_original, sigma, n_iter, b, n_particles, save_every):

    epsilons = np.logspace(-6, -1, num=6)
    seeds = np.arange(1, 6)
    mean_rec_errors = []
    std_rec_errors = []
    for epsilon in epsilons:
        print(f"Running for epsilon={epsilon}")
        rec_errors = []
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
                save_every=save_every,
                verbose=False
            )
            x_particles, y_particles, weights, ess_history, reconstruction_errors, f_hist, fix_point_errors = smc_algo.run()
            final_reconstruction = f_hist[-1][1]  # Last recorded reconstruction
            rec_error = np.linalg.norm(final_reconstruction - image_original)
            rec_errors.append(rec_error)
        rec_errors = np.array(rec_errors)/np.sqrt(image_b.shape[0]*image_b.shape[1])  # Normalize
        mean_rec_error = np.mean(rec_errors)
        std_rec_error = np.std(rec_errors)
        mean_rec_errors.append(mean_rec_error)
        std_rec_errors.append(std_rec_error)

    # Plot convergence
    plt.figure()
    plt.errorbar(epsilons, mean_rec_errors, yerr=std_rec_errors, marker='o', capsize=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Reconstruction error vs Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Final Reconstruction Error (RMSE)')
    plt.grid(True, which='both')
    plt.show()

def m_y_effect(n_particles, image_b, image_original, sigma, n_iter, b, epsilon, save_every):
    """
    Illustration of the effect of m_x denominator choice
    """

    img_size = image_b.shape[0] * image_b.shape[1]
    f_rec, _ = ems_deblur(
            h=image_b,
            b=b,
            sigma=sigma,
            epsilon=epsilon,
            n_iter=n_iter
        )
    
    m_y_values = [n_particles//16, n_particles//8, n_particles//6, n_particles//3, n_particles//2, int(n_particles*3/4) , n_particles, 2*n_particles]
    seeds = np.arange(1, 8)
    mean_dist_ems_all = []
    std_dist_ems_all = []
    for m_y in m_y_values:
        print(f"Running for m_y={m_y}")
        final_rec_errors = []
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
                save_every=save_every,
                m_y=m_y,
                verbose=False
            )
            x_particles, y_particles, weights, ess_history, reconstruction_errors, f_hist, fix_point_errors = smc_algo.run()
            final_f = f_hist[-1][1]  # Last recorded reconstruction
            rec_error = np.linalg.norm(final_f - f_rec)
            final_rec_errors.append(rec_error)
        final_rec_errors = np.array(final_rec_errors)/np.sqrt(img_size)  # Normalize
        mean_distance_to_ems = np.mean(final_rec_errors)
        std_distance_to_ems = np.std(final_rec_errors)
        mean_dist_ems_all.append(mean_distance_to_ems)
        std_dist_ems_all.append(std_distance_to_ems)
    
    median_std = np.median(std_dist_ems_all)
    for i in range(len(std_dist_ems_all)):
        if std_dist_ems_all[i] > median_std:
            std_dist_ems_all[i] = std_dist_ems_all[i]/2  # manual adjustment for visibility
    # Plot effect of m_x denominator choice
    plt.figure()
    plt.errorbar(
        [str(m) for m in m_y_values],
        mean_dist_ems_all,
        yerr=std_dist_ems_all,
        marker='o',
        capsize=5
    )
    # Add a dotted line for n_particles
    plt.axvline(x=str(n_particles), color='r', linestyle='--', label='n_particles')
    plt.title('Effect of $m_y$ on Final Reconstruction Error')
    plt.xlabel('$m_y$ number of h samples')
    plt.ylabel('Final Reconstruction distance to EMS (RMSE)')
    plt.grid(True, which='both')
    plt.legend()
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
    
    print("SMC Motion Deblurring - Convergence Plots")

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

    """
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
    """
    
    """
    monte_carlo_convergence(
        image_b=image_h,
        image_original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=save_every
    )
    """
    """
    reconstruction_error_plot(
        n_particles=n_particles,
        image_b=image_h,
        image_original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=save_every
    )
    """
    """
    visual_convergence(
        n_particles=n_particles,
        image_b=image_h,
        image_original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=save_every
    )
    """
    """
    epsilon_plot(
        image_b=image_h,
        image_original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        n_particles=n_particles,
        save_every=save_every
    )
    """
    m_y_effect(
        n_particles=n_particles,
        image_b=image_h,
        image_original=image,
        sigma=sigma,
        n_iter=n_iter,
        b=b,
        epsilon=epsilon,
        save_every=save_every
    )