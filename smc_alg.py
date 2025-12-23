import numpy as np
from scipy.stats import norm
from kde_smoothing import reconstruct_image_from_particles
from numba import njit, prange

def pinky(y_in, x_in, image, n_samples):
    """
    Sample from the blurred image
    IMPORTANT: image should be transposed before passing here (like MATLAB's I')
    """
    # Flatten image and create probability distribution
    image_flat = image.T.ravel()  # Match MATLAB's column-major ordering
    image_flat = np.maximum(image_flat, 0)  # Ensure non-negative
    
    if image_flat.sum() == 0:
        print("Warning: Image sum is zero!")
        image_flat = np.ones_like(image_flat)
    
    image_flat = image_flat / image_flat.sum()
    
    # Sample indices
    indices = np.random.choice(len(image_flat), size=n_samples, p=image_flat)
    
    # Convert to 2D coordinates (transposed indexing)
    rows = indices % len(y_in)
    cols = indices // len(y_in)
    
    samples = np.column_stack([y_in[rows], x_in[cols]])
    return samples


def mult_resample(weights, n):
    """
    Multinomial resampling
    
    Parameters:
    -----------
    weights : ndarray
        Particle weights
    n : int
        Number of particles
    
    Returns:
    --------
    indices : ndarray
        Resampled indices
    """
    # Normalize weights
    weights = weights / weights.sum()
    
    # Multinomial resampling
    indices = np.random.choice(n, size=n, p=weights)
    return indices


@njit(parallel=True, fastmath=True)
def hNn_numba(y_samples, x_particles, y_particles, weights, b, sigma):
    """
    Compute h^N_n for each y_j sample.
    h^N_n(y_j) = (1/N) sum_{i=1}^N g(y_j | x_i)
    which is an approximation of the integral f_n(z) g(y_j | z) dz
    """
    n_particles = x_particles.shape[0]
    h_n = np.zeros(n_particles, dtype=np.float64)
    inv_b = 1.0 / b
    inv_sigma = 1.0 / sigma
    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    for j in prange(n_particles):
        y0 = y_samples[j, 0]
        x0 = y_samples[j, 1]
        acc = 0.0
        for i in range(n_particles):
            dy = y0 - y_particles[i]
            # Normal pdf inline
            normal_part = norm_const * np.exp(-0.5 * (dy * inv_sigma) * (dy * inv_sigma))
            dx = x_particles[i] - x0
            uniform_part = inv_b if (dx <= b * 0.5 and dx >= -b * 0.5) else 0.0
            acc += weights[i] * normal_part * uniform_part
        h_n[j] = acc
    return h_n


@njit(parallel=True, fastmath=True)
def weight_update_numba(y_samples, x_particles, y_particles, weights, h_n, b, sigma):
    """
    Numba-accelerated weight update. Avoids SciPy and large N×N allocations.
    """
    n_particles = x_particles.shape[0]
    new_weights = np.zeros(n_particles, dtype=np.float64)
    inv_b = 1.0 / b
    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    # Iterate over particles i in parallel
    for i in prange(n_particles):
        acc = 0.0
        # Average over all y_samples (j)
        for j in range(n_particles):
            dy = y_samples[j, 0] - y_particles[i]
            # inline Normal pdf
            normal_part = norm_const * np.exp(-0.5 * (dy / sigma) * (dy / sigma))
            dx = x_particles[i] - y_samples[j, 1]
            uniform_part = inv_b if (dx <= b * 0.5 and dx >= -b * 0.5) else 0.0
            denom = h_n[j] + 1e-10
            acc += (normal_part * uniform_part) / denom
        potential = acc / n_particles  # mean over j
        # guard against NaN (rare with our denom, but keep parity)
        if np.isnan(potential):
            potential = 0.0
        new_weights[i] = weights[i] * potential

    # Normalize
    s = new_weights.sum()
    if s > 0.0:
        new_weights /= s
    return new_weights


class SMCAlgorithm:
    def __init__(self, n_particles, image, original, sigma, n_iter, b, epsilon, save_every=10):
        """
        SMC for motion deblurring
        
        Parameters:
        -----------
        n_particles : int
            Number of particles
        n_iter : int
            Number of time steps
        epsilon : float
            Scale parameter for the gaussian kernel
        image : ndarray
            Blurred and noisy image
        original : ndarray
            Original sharp image
        sigma : float
            Standard deviation for Normal approximating Dirac delta
        b : float
            Velocity of motion
        """
        self.n_particles = n_particles
        self.image = image
        self.original = original
        self.sigma = sigma
        self.img_shape = image.shape
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.reconstruction_errors = []
        self.fix_point_errors = []
        self.save_every = save_every  # Save reconstruction error every n iterations

        # Get dimension of image
        H, W = self.img_shape
        print(f'Image dimensions: {H} x {W}')
        ymax = H/W
        
        # Normalize velocity
        self.b = b * 2 / W  # Normalize by image width to match x in [-1, 1]
        
        # x is in [-1, 1]
        self.x_in = np.linspace(-1 + 1/W, 1 - 1/W, W)
        # y is in [-ymax, ymax]
        self.y_in = np.linspace(ymax - 1/H, -ymax + 1/H, H)
        
        # Initialize particle arrays
        self.x = np.zeros((n_iter+1, n_particles))
        self.y = np.zeros((n_iter+1, n_particles))
        self.W = np.zeros((n_iter+1, n_particles))
        
        # Sample random particles for time step n = 1
        self.x[0, :] = 2 * np.random.rand(n_particles) - 1  # uniform in [-1, 1]
        self.y[0, :] = ymax * (2 * np.random.rand(n_particles) - 1)  # uniform in [-ymax, ymax]
        self.W[0, :] = 1.0 / n_particles

        self.n = 0  # Current iteration
        self.previous_reconstructed = reconstruct_image_from_particles(
                    self.x[self.n, :],
                    self.y[self.n, :],
                    self.W[self.n, :],
                    self.img_shape,
                    self.epsilon
                ) # For fix-point error calculation


    def step(self):
        self.n += 1
        print(f'Iteration {self.n}/{self.n_iter}')

        # Get N samples from blurred image
        h_sample = pinky(self.y_in, self.x_in, self.image, self.n_particles)
        
        # Calculate ESS
        ess = 1.0 / np.sum(self.W[self.n-1, :]**2)
        
        # RESAMPLING
        if ess < self.n_particles / 2:
            print(f'Resampling at Iteration {self.n+1}')
            indices = mult_resample(self.W[self.n-1, :], self.n_particles)
            self.x[self.n, :] = self.x[self.n-1, indices]
            self.y[self.n, :] = self.y[self.n-1, indices]
            self.W[self.n, :] = 1.0 / self.n_particles
        else:
            self.x[self.n, :] = self.x[self.n-1, :]
            self.y[self.n, :] = self.y[self.n-1, :]
            self.W[self.n, :] = self.W[self.n-1, :]

        # Compute h^N_n for each y_j
        h_n = hNn_numba(h_sample, self.x[self.n, :], self.y[self.n, :], self.W[self.n, :], self.b, self.sigma)
        
        
        # Apply Markov kernel
        self.x[self.n, :] = self.x[self.n, :] + self.epsilon * np.random.randn(self.n_particles)
        self.y[self.n, :] = self.y[self.n, :] + self.epsilon * np.random.randn(self.n_particles)

        # Update weights
        self.W[self.n, :] = weight_update_numba(
            h_sample,
            self.x[self.n, :], # Post-kernel x particles
            self.y[self.n, :], # Post-kernel y particles
            self.W[self.n, :], # Previous weights
            h_n,
            self.b,
            self.sigma,
        )
        
    def run(self):
        """
        Returns:
        --------
        x : ndarray
            Particle x-coordinates (n_iter x n_particles)
        y : ndarray
            Particle y-coordinates (n_iter x n_particles)
        W : ndarray
            Particle weights (n_iter x n_particles)
        ess_history : ndarray
            Effective sample size at each iteration
        reconstruction_errors : list of tuples
            List of (iteration, reconstruction error) tuples
        fix_point_errors : list of tuples
            List of (iteration, error between previous and current iterate) tuples
        """
        print("Starting SMC Algorithm")
        for _ in range(1, self.n_iter+1):
            self.step()
            if self.n % self.save_every == 0:
                reconstructed_image = reconstruct_image_from_particles(
                    self.x[self.n, :],
                    self.y[self.n, :],
                    self.W[self.n, :],
                    self.img_shape,
                    self.epsilon
                )
                error = np.linalg.norm(reconstructed_image - self.original)
                fix_point_error = np.linalg.norm(reconstructed_image - self.previous_reconstructed)
                self.previous_reconstructed = reconstructed_image
                print(f'Reconstruction error at iteration {self.n}: {error:.6f}')
                print(f'Fix-point error at iteration {self.n}: {fix_point_error:.6f}')
                self.reconstruction_errors.append((self.n, error))
                self.fix_point_errors.append((self.n, fix_point_error))
        ess_history = 1.0 / np.sum(self.W**2, axis=1)
        return self.x, self.y, self.W, ess_history, self.reconstruction_errors, self.fix_point_errors
