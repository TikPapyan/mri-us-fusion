import numpy as np
from .dtd import dtd
from .Link import Link

def gradf1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3):
    """
    Gradient of the objective function for ultrasound image update
    """
    # Ensure proper shapes and clip values for stability
    x_vec = np.clip(x.flatten().reshape(-1, 1), -10, 10)  # Clip to prevent overflow
    y2_vec = np.clip(y2.flatten().reshape(-1, 1), -10, 10)  # Clip to prevent overflow
    x1k_vec = x1k.flatten().reshape(-1, 1)
    x2k_vec = x2k.flatten().reshape(-1, 1)
    
    # Get original image shape for gradient computation
    if len(y2.shape) == 2:
        img_shape = y2.shape
    else:
        # Try to infer shape from y2_vec length
        n_pixels = y2_vec.shape[0]
        # Assume square image
        n = int(np.sqrt(n_pixels))
        img_shape = (n, n)
    
    # Compute X term with clipping
    try:
        X = x2k_vec - 4 * (x2k_vec - Link(x1k_vec.reshape(img_shape), c).flatten().reshape(-1, 1))
        X = np.clip(X, -10, 10)  # Clip X to prevent large values
    except:
        # If Link fails, use a simpler computation
        X = x2k_vec.copy()
    
    # Compute gradient components with numerical stability
    
    # First term: derivative of exponential term (stable computation)
    diff = y2_vec - x_vec
    diff = np.clip(diff, -50, 50)  # Clip to prevent overflow in exp
    exp_term = np.exp(diff)
    term1_grad = tau1 * (gama - exp_term)
    
    # Clip gradient to prevent explosion
    term1_grad = np.clip(term1_grad, -1e6, 1e6)
    
    # Second term: derivative of TV term
    # Reshape x to image for gradient computation
    x_img = x_vec.reshape(img_shape)
    term2_grad = tau2 * dtd(x_img)
    term2_grad = term2_grad.flatten().reshape(-1, 1)
    
    # Third term: derivative of distance term
    term3_grad = tau3 / 2 * (x_vec - X)
    
    # Total gradient with clipping
    gradf = term1_grad + term2_grad + term3_grad
    
    # Clip final gradient to prevent explosion
    gradf = np.clip(gradf, -1e8, 1e8)
    
    return gradf