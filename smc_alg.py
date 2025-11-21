import numpy as np
from scipy.stats import norm

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


def optimal_bandwidth_ess(particles, weights):
    """
    Calculate optimal bandwidth using effective sample size
    Matches MATLAB's optimal_bandwidthESS function
    """
    # Weighted mean
    weighted_mean = np.sum(weights * particles)
    # Weighted variance
    weighted_var = np.sum(weights * (particles - weighted_mean)**2)
    weighted_std = np.sqrt(weighted_var)
    
    # Effective sample size
    n_eff = 1.0 / np.sum(weights**2)
    
    # Silverman's rule of thumb
    bandwidth = 1.06 * weighted_std * n_eff**(-0.2)
    return bandwidth


class SMCAlgorithm:
    def __init__(self, n_particles, image, sigma, n_iter, b, epsilon):
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
        sigma : float
            Standard deviation for Normal approximating Dirac delta
        b : float
            Velocity of motion
        """
        self.n_particles = n_particles
        self.image = image
        self.sigma = sigma
        self.img_shape = image.shape
        self.n_iter = n_iter
        self.epsilon = epsilon

        # Get dimension of image
        pixels = self.img_shape
        
        # Normalize velocity
        self.b = b / 300
        
        # x is in [-1, 1]
        self.x_in = np.linspace(-1 + 1/pixels[1], 1 - 1/pixels[1], pixels[1])
        # y is in [-0.5, 0.5]
        self.y_in = np.linspace(0.5 - 1/pixels[0], -0.5 + 1/pixels[0], pixels[0])
        
        # Initialize particle arrays
        self.x = np.zeros((n_iter+1, n_particles))
        self.y = np.zeros((n_iter+1, n_particles))
        self.W = np.zeros((n_iter+1, n_particles))
        
        # Sample random particles for time step n = 1
        self.x[0, :] = 2 * np.random.rand(n_particles) - 1  # uniform in [-1, 1]
        self.y[0, :] = np.random.rand(n_particles) - 0.5    # uniform in [-0.5, 0.5]
        self.W[0, :] = 1.0 / n_particles

        self.n = 0  # Current iteration


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
        h_n = np.zeros(self.n_particles)
        for j in range(self.n_particles):
            normal_part = norm.pdf(h_sample[j, 0] - self.y[self.n, :], loc=0, scale=self.sigma)
            uniform_part = ((self.x[self.n, :] - h_sample[j, 1] <= self.b/2) & 
                          (self.x[self.n, :] - h_sample[j, 1] >= -self.b/2)).astype(float) / self.b
            h_n[j] = np.mean(self.W[self.n, :] * normal_part * uniform_part)
        
        # Apply Markov kernel
        self.x[self.n, :] = self.x[self.n, :] + self.epsilon * np.random.randn(self.n_particles)
        self.y[self.n, :] = self.y[self.n, :] + self.epsilon * np.random.randn(self.n_particles)
        
        # Update weights
        for i in range(self.n_particles):
            g = (norm.pdf(h_sample[:, 0] - self.y[self.n, i], loc=0, scale=self.sigma) *
                 ((self.x[self.n, i] - h_sample[:, 1] <= self.b/2) & 
                  (self.x[self.n, i] - h_sample[:, 1] >= -self.b/2)).astype(float) / self.b)
            
            # Potential at time n
            potential = np.mean(g / (h_n + 1e-10))  # Add small constant to avoid division by zero
            
            # Check for NaN
            if np.isnan(potential):
                potential = 0
            
            # Update weight
            self.W[self.n, i] = self.W[self.n, i] * potential
        
        # Normalize weights
        self.W[self.n, :] = self.W[self.n, :] / (self.W[self.n, :].sum() + 1e-10)

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
        """
        print("Starting SMC Algorithm")
        for _ in range(1, self.n_iter+1):
            self.step()
        ess_history = 1.0 / np.sum(self.W**2, axis=1)
        return self.x, self.y, self.W, ess_history

