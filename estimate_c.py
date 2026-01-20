import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt

def estimate_c(irm, us):
    """
    Estimate polynomial coefficients c that relate MRI and ultrasound images
    """
    # Normalize images to [0, 1] range first
    irm_norm = (irm - np.min(irm)) / (np.max(irm) - np.min(irm) + 1e-10)
    us_norm = (us - np.min(us)) / (np.max(us) - np.min(us) + 1e-10)
    
    y1 = irm_norm
    y2 = us_norm
    
    n1, n2 = y2.shape
    
    # Resize MRI image
    yint = cv2.resize(y1, (n2, n1), interpolation=cv2.INTER_CUBIC)
    
    # Compute gradient
    from scipy import signal
    Jx = signal.convolve2d(yint, np.array([[-1, 1]]), mode='same', boundary='symm')
    Jy = signal.convolve2d(yint, np.array([[-1, 1]]).T, mode='same', boundary='symm')
    gradY = np.sqrt(Jx**2 + Jy**2)
    
    # Image vectorization
    yi = yint.reshape(n1 * n2, 1)
    yu = y2.reshape(n1 * n2, 1)
    dyi = gradY.reshape(n1 * n2, 1)
    
    # Scale features to prevent numerical issues
    # Normalize x and gradient values
    yi_scaled = (yi - np.mean(yi)) / (np.std(yi) + 1e-10)
    dyi_scaled = (dyi - np.mean(dyi)) / (np.std(dyi) + 1e-10)
    
    # Compute matrix A with scaled features
    A = np.hstack([
        np.ones((n1 * n2, 1)),
        yi_scaled,
        yi_scaled**2,
        yi_scaled**3,
        yi_scaled**4,
        dyi_scaled,
        dyi_scaled * yi_scaled,
        dyi_scaled * yi_scaled**2,
        dyi_scaled * yi_scaled**3,
        dyi_scaled**2,
        dyi_scaled**2 * yi_scaled,
        dyi_scaled**2 * yi_scaled**2,
        dyi_scaled**3,
        dyi_scaled**3 * yi_scaled,
        dyi_scaled**4
    ])
    
    # Add L2 regularization to prevent overfitting
    lambda_reg = 0.1  # Regularization parameter
    n_features = A.shape[1]
    
    # Regularized least squares: (A^T A + lambda*I)^{-1} A^T y
    ATA = A.T @ A
    reg_matrix = lambda_reg * np.eye(n_features)
    
    # Solve with regularization
    try:
        cest = np.linalg.solve(ATA + reg_matrix, A.T @ yu)
    except np.linalg.LinAlgError:
        # Use pseudoinverse if matrix is singular
        cest = np.linalg.pinv(A) @ yu
    
    print(f"Estimated coefficients (first 5): {cest[:5].flatten()}")
    print(f"Coefficient range: [{np.min(cest):.2e}, {np.max(cest):.2e}]")
    
    # Scale coefficients back to original space
    # Note: Since we scaled features, coefficients are for scaled features
    # For simplicity, we'll use them as-is and rely on image normalization
    
    return cest.flatten()

if __name__ == '__main__':
    # Test the function
    irm_data = sio.loadmat('images/irm.mat')
    us_data = sio.loadmat('images/us.mat')
    
    irm = irm_data['irm'] if 'irm' in irm_data else irm_data[list(irm_data.keys())[-1]]
    us = us_data['us'] if 'us' in us_data else us_data[list(us_data.keys())[-1]]
    
    c = estimate_c(irm, us)
    print(f"Polynomial coefficients shape: {c.shape}")