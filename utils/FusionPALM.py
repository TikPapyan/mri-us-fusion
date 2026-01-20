import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from .FSR_xirm_NL import FSR_xirm_NL
from .Descente_grad_xus_NL import Descente_grad_xus_NL

def FusionPALM(y1, y2, c, tau1, tau2, tau3, tau4, plot_fused_image=False):
    """
    Main PALM algorithm for MRI and ultrasound image fusion
    """
    n1, n2 = y2.shape
    print(f"FusionPALM: Input shapes - y1={y1.shape}, y2={y2.shape}")
    
    # Create Gaussian blur kernel
    B = np.zeros((5, 5), dtype=np.float64)
    B[2, 2] = 1
    B = cv2.GaussianBlur(B, (5, 5), 4)
    print(f"FusionPALM: Blur kernel B shape={B.shape}")
    
    # Resize MRI image
    yint = cv2.resize(y1, (n2, n1), interpolation=cv2.INTER_CUBIC)
    print(f"FusionPALM: Resized MRI yint shape={yint.shape}")
    
    # Compute gradient
    Jx = signal.convolve2d(yint, np.array([[-1, 1]], dtype=np.float64), 
                          mode='same', boundary='symm')
    Jy = signal.convolve2d(yint, np.array([[-1, 1]], dtype=np.float64).T, 
                          mode='same', boundary='symm')
    gradY = np.sqrt(Jx**2 + Jy**2)
    print(f"FusionPALM: gradY shape={gradY.shape}")
    
    # PALM parameters
    m_iteration = 10
    gama = 1e-3
    
    # Define difference operator kernels
    dh = np.zeros((n1, n2), dtype=np.float64)
    dh[0, 0] = 1
    dh[0, 1] = -1
    
    dv = np.zeros((n1, n2), dtype=np.float64)
    dv[0, 0] = 1
    dv[1, 0] = -1
    
    # Compute FFTs for filtering
    FDH = np.fft.fft2(dh)
    F2DH = np.abs(FDH)**2
    FDV = np.fft.fft2(dv)
    FDV = np.conj(FDV)
    F2DV = np.abs(FDV)**2
    
    c1 = 1e-8
    F2D = F2DH + F2DV + c1
    print(f"FusionPALM: F2D shape={F2D.shape}")
    
    # Parameter settings
    taup = 1
    tau = taup  # MRI parameter (TV influence)
    tau10 = tau1  # MRI parameter (echo influence)
    tau1_us = tau2  # US parameter (observation influence)
    tau2_us = tau3  # US parameter (TV influence)
    tau3_us = tau4  # US parameter (MRI influence)
    
    # PALM initialization
    d = 6
    x2 = y2 + c1
    x1 = yint.astype(np.float64)
    
    print(f"FusionPALM: Initial x1 shape={x1.shape}, x2 shape={x2.shape}")
    print(f"FusionPALM: Starting {m_iteration} iterations...")
    
    # PALM iterations
    for i in range(m_iteration):
        print(f"\nPALM Iteration {i+1}/{m_iteration}")
        
        # Update MRI image
        print(f"  Updating MRI image...")
        x1 = FSR_xirm_NL(x1, y1, x2, gradY, B, d, c, F2D, tau, tau10, False)
        print(f"  Updated x1 shape={x1.shape}")
        
        # Update ultrasound image
        print(f"  Updating ultrasound image...")
        x2 = Descente_grad_xus_NL(y2, x1, x2, c, gama, tau1_us, tau2_us, tau3_us, False, 0.2)
        print(f"  Updated x2 shape={x2.shape}")
    
    # Plot fused image
    if plot_fused_image:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(y1, cmap='gray')
        plt.title('Original MRI')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(x2, cmap='gray')
        plt.title('Fused Image (PALM Algorithm)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    return x2