import numpy as np
from .d1 import d1
from .Link import Link

def f1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3):
    """
    Objective function for ultrasound image update
    """
    # Get original image shape
    if len(y2.shape) == 2:
        img_shape = y2.shape
        y2_vec = y2.reshape(-1, 1)
    else:
        y2_vec = y2.reshape(-1, 1)
        n_pixels = y2_vec.shape[0]
        n = int(np.sqrt(n_pixels))
        img_shape = (n, n)
    
    # Reshape inputs with clipping for stability
    x_vec = np.clip(x.reshape(-1, 1), -10, 10)
    y2_vec = np.clip(y2_vec, -10, 10)
    x1k_img = x1k.reshape(img_shape) if len(x1k.shape) == 2 else x1k.reshape(img_shape)
    x2k_vec = x2k.reshape(-1, 1)
    
    # Compute X term with clipping
    try:
        X = x2k_vec - 4 * (x2k_vec - Link(x1k_img, c).flatten().reshape(-1, 1))
        X = np.clip(X, -10, 10)
    except:
        X = x2k_vec.copy()
    
    # Compute objective function with stable computations
    
    # Exponential term (stable computation)
    diff = y2_vec - x_vec
    diff = np.clip(diff, -50, 50)  # Clip to prevent overflow
    exp_term = gama * np.exp(diff) - diff
    term1 = tau1 * np.sum(exp_term)
    
    # TV term - reshape x to image for gradient computation
    x_img = x_vec.reshape(img_shape)
    grad_x = d1(x_img)
    term2 = tau2 / 2 * np.sum(grad_x**2)
    
    # Distance term
    term3 = tau3 * np.linalg.norm(x_vec - X)**2
    
    return term1 + term2 + term3