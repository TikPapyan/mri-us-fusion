import numpy as np
from scipy import signal

def Link(x1, c):
    """
    Polynomial function linking MRI and ultrasound images with numerical stability
    """
    # Ensure x1 is in [0, 1] range
    x1 = np.clip(x1, 0, 1)
    
    # Handle vector input
    original_shape = x1.shape
    if len(x1.shape) == 1 or (len(x1.shape) == 2 and (x1.shape[0] == 1 or x1.shape[1] == 1)):
        # Vector input
        x1_vec = x1.flatten()
        n_pixels = len(x1_vec)
        
        # Simple gradient approximation for 1D
        Jx = np.zeros_like(x1_vec)
        if len(x1_vec) > 1:
            Jx[:-1] = x1_vec[1:] - x1_vec[:-1]
        Jy = np.zeros_like(x1_vec)
        gradY = np.sqrt(Jx**2 + Jy**2 + 1e-10)
        
        # Compute polynomial terms with clipping
        x1_sq = np.clip(x1_vec**2, 0, 10)
        x1_cu = np.clip(x1_vec**3, 0, 10)
        x1_qu = np.clip(x1_vec**4, 0, 10)
        
        gradY_sq = np.clip(gradY**2, 0, 10)
        gradY_cu = np.clip(gradY**3, 0, 10)
        gradY_qu = np.clip(gradY**4, 0, 10)
        
        # Compute with clipping to prevent overflow
        x2 = (c[0] + 
              np.clip(c[1] * x1_vec, -1e3, 1e3) +
              np.clip(c[2] * x1_sq, -1e3, 1e3) +
              np.clip(c[3] * x1_cu, -1e3, 1e3) +
              np.clip(c[4] * x1_qu, -1e3, 1e3) +
              np.clip(c[5] * gradY, -1e3, 1e3) +
              np.clip(c[6] * gradY * x1_vec, -1e3, 1e3) +
              np.clip(c[7] * gradY * x1_sq, -1e3, 1e3) +
              np.clip(c[8] * gradY * x1_cu, -1e3, 1e3) +
              np.clip(c[9] * gradY_sq, -1e3, 1e3) +
              np.clip(c[10] * gradY_sq * x1_vec, -1e3, 1e3) +
              np.clip(c[11] * gradY_sq * x1_sq, -1e3, 1e3) +
              np.clip(c[12] * gradY_cu, -1e3, 1e3) +
              np.clip(c[13] * gradY_cu * x1_vec, -1e3, 1e3) +
              np.clip(c[14] * gradY_qu, -1e3, 1e3))
        
        return x2.reshape(original_shape)
    else:
        # 2D image input
        # Compute gradient
        Jx = signal.convolve2d(x1, np.array([[-1, 1]]), mode='same', boundary='symm')
        Jy = signal.convolve2d(x1, np.array([[-1, 1]]).T, mode='same', boundary='symm')
        gradY = np.sqrt(Jx**2 + Jy**2 + 1e-10)
        
        # Clip high-degree terms to prevent overflow
        x1_sq = np.clip(x1**2, 0, 10)
        x1_cu = np.clip(x1**3, 0, 10)
        x1_qu = np.clip(x1**4, 0, 10)
        
        gradY_sq = np.clip(gradY**2, 0, 10)
        gradY_cu = np.clip(gradY**3, 0, 10)
        gradY_qu = np.clip(gradY**4, 0, 10)
        
        # Compute with clipping to prevent overflow
        x2 = (c[0] + 
              np.clip(c[1] * x1, -1e3, 1e3) +
              np.clip(c[2] * x1_sq, -1e3, 1e3) +
              np.clip(c[3] * x1_cu, -1e3, 1e3) +
              np.clip(c[4] * x1_qu, -1e3, 1e3) +
              np.clip(c[5] * gradY, -1e3, 1e3) +
              np.clip(c[6] * gradY * x1, -1e3, 1e3) +
              np.clip(c[7] * gradY * x1_sq, -1e3, 1e3) +
              np.clip(c[8] * gradY * x1_cu, -1e3, 1e3) +
              np.clip(c[9] * gradY_sq, -1e3, 1e3) +
              np.clip(c[10] * gradY_sq * x1, -1e3, 1e3) +
              np.clip(c[11] * gradY_sq * x1_sq, -1e3, 1e3) +
              np.clip(c[12] * gradY_cu, -1e3, 1e3) +
              np.clip(c[13] * gradY_cu * x1, -1e3, 1e3) +
              np.clip(c[14] * gradY_qu, -1e3, 1e3))
        
        return np.clip(x2, 0, 1)  # Ensure output is in [0, 1]