import numpy as np
import matplotlib.pyplot as plt
from .d1 import d1
from .dtd import dtd
from .f1_NL import f1_NL
from .gradf1_NL import gradf1_NL
from .Link import Link

def Descente_grad_xus_NL(y2, x1k, x2k, c, gama, tau1, tau2, tau3, display=False, alpha=0.01):
    """
    Gradient descent for updating ultrasound image in PALM algorithm
    """
    n1, n2 = y2.shape
    
    # Reshape to vectors while keeping track of original shape
    y2_vec = y2.reshape(n1 * n2, 1)
    x1k_vec = x1k.reshape(n1 * n2, 1)
    x2k_vec = x2k.reshape(n1 * n2, 1)
    
    # Initialization - use current estimate
    x0 = x2k.reshape(-1, 1).copy()
    
    # Termination tolerance - reduced for more iterations
    tol = 1e-4  # Reduced from 1e-3
    maxiter = 20  # Increased from 10
    dxmin = 1e-4
    
    # Initialize variables
    gnorm = np.inf
    x = x0.copy()
    niter = 0
    dx = np.inf
    
    # Store objective values for monitoring
    f_values = []
    
    # Gradient descent loop with adaptive learning rate
    while gnorm >= tol and niter < maxiter and dx >= dxmin:
        # Calculate gradient
        g = gradf1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3)
        
        # Check for NaN in gradient
        if np.any(np.isnan(g)) or np.any(np.isinf(g)):
            print(f"    Warning: NaN/Inf in gradient at iteration {niter}")
            break
        
        # Adaptive learning rate
        current_alpha = alpha
        g_norm = np.linalg.norm(g)
        
        if g_norm > 1e3:
            current_alpha = alpha / 10
        elif g_norm < 1e-3:
            current_alpha = alpha * 2
        
        # Take step
        xnew = x - current_alpha * g
        
        # Clip to valid range
        xnew = np.clip(xnew, 0, 1)
        
        # Update termination metrics
        niter += 1
        dx = np.linalg.norm(xnew - x)
        x = xnew.copy()
        gnorm = g_norm
        
        # Compute objective value for monitoring
        if niter % 5 == 0:
            f_val = f1_NL(x, y2_vec, x1k_vec, x2k_vec, c, gama, tau1, tau2, tau3)
            f_values.append(f_val)
    
    # Compute final objective function value
    try:
        fopt = f1_NL(x, y2_vec, x1k_vec, x2k_vec, c, gama, tau1, tau2, tau3)
        print(f"    Descente_grad_xus_NL: {niter} iterations, f={fopt:.4f}, ‖g‖={gnorm:.2e}")
    except:
        print(f"    Descente_grad_xus_NL: {niter} iterations (fopt computation failed)")
        fopt = 0
    
    # Reshape back to image
    x2 = x.reshape(n1, n2)
    
    # Display convergence plot if requested
    if display and len(f_values) > 1:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(x2, cmap='gray')
        plt.title('Updated Ultrasound Image')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(0, len(f_values)*5, 5), f_values, 'b.-')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.title('Convergence Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return x2