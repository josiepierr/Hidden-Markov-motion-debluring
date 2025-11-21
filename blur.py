import numpy as np
from scipy.stats import norm
from skimage.util import img_as_float

def blurred_image(image_f, b, sigma):
    """
    Blur caused by constant speed motion
    
    Parameters:
    -----------
    image_f : ndarray
        Original sharp image
    b : float
        Speed of motion in horizontal direction
    sigma : float
        Variance of gaussian describing motion in vertical direction
    
    Returns:
    --------
    image_h : ndarray
        Blurred image
    """
    # Convert to grayscale and double
    if len(image_f.shape) == 3:
        image_f = image_f[:, :, 0]
    image_f = img_as_float(image_f)
    
    # Image dimensions
    pixels = image_f.shape
    
    # Normalize velocity
    b = b / pixels[0]
    
    # Create empty image
    image_h = np.zeros(pixels)
    
    # Set coordinate system over image
    # x is in [-1, 1]
    eval_x = np.linspace(-1 + 1/pixels[1], 1 - 1/pixels[1], pixels[1])
    # y is in [-0.5, 0.5]
    eval_y = np.linspace(0.5 - 1/pixels[0], -0.5 + 1/pixels[0], pixels[0])
    
    # Build grid with these coordinates
    grid_x, grid_y = np.meshgrid(eval_x, eval_y)
    
    for i in range(pixels[1]):
        u = eval_x[i]
        for j in range(pixels[0]):
            v = eval_y[j]
            # New pixel value after blur
            normal_component = norm.pdf(v, loc=grid_y, scale=sigma)
            uniform_component = uniform_pdf(grid_x - u, -b/2, b/2)
            image_h[j, i] = np.sum(image_f * normal_component * uniform_component)
    
    # Normalize image to [0, 1]
    image_h = (image_h - image_h.min()) / (image_h.max() - image_h.min())
    return image_h


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