import numpy as np
import matplotlib.pyplot as plt
from .HXconv import HXconv
from .Link import Link

def FSR_xirm_NL(x1k, y1, xus, gradY, B, d, c, F2D, tau, tau10, display=False):
    """
    OPTIMIZED Fast Super-Resolution update for MRI image
    Focus: Preserve MRI information while fusing with US
    """
    n1y, n2y = y1.shape
    n1, n2 = xus.shape
    
    # Preserve MRI intensity range
    x1k_original = x1k.copy()
    x1k = np.clip(x1k.astype(np.float64), 0, 1)
    gradY = np.clip(gradY.astype(np.float64), 0, 1)
    xus = np.clip(xus.astype(np.float64), 0, 1)
    
    print(f"  FSR_xirm_NL: x1k range=[{np.min(x1k):.3f}, {np.max(x1k):.3f}]")
    
    # Compute linked image
    linked = Link(x1k, c)
    
    # ADAPTIVE UPDATE: Only update where US provides new information
    diff = xus - linked
    diff_magnitude = np.abs(diff)
    
    # Create update mask: only update where difference is significant
    update_threshold = 0.1  # Only update if difference > 10%
    update_mask = diff_magnitude > update_threshold
    
    # Very conservative update to preserve MRI
    update_strength = 0.1  # Only 10% update
    
    # Compute X: Blend based on update mask
    X = x1k.copy()
    if np.any(update_mask):
        # Simple first-order polynomial derivative
        poly_derivative = c[1] + 2 * c[2] * x1k
        poly_derivative = np.clip(poly_derivative, -0.5, 0.5)
        
        # Apply update only where needed
        update = update_strength * poly_derivative * diff
        X[update_mask] = x1k[update_mask] - update[update_mask]
    
    # Strong preservation of original MRI
    preservation = 0.9  # Keep 90% of original MRI
    X = preservation * x1k + (1 - preservation) * X
    X = np.clip(X, 0, 1)
    
    print(f"  FSR_xirm_NL: X range=[{np.min(X):.3f}, {np.max(X):.3f}], "
          f"update pixels: {np.sum(update_mask)}/{x1k.size} "
          f"({np.sum(update_mask)/x1k.size*100:.1f}%)")
    
    # Prepare y_irm = S^T y1
    STy = np.zeros((n1, n2), dtype=np.complex128)
    STy[::d, ::d] = y1.astype(np.complex128)
    
    # HX convolution
    FB, FBC, F2B = HXconv(STy, B, 'Hx')
    
    # Compute FR in Fourier domain
    FR = FBC * np.fft.fft2(STy) + np.fft.fft2(2 * tau10 * X)
    
    # Compute l1 term with safe division
    denominator = F2D + 100 * tau10 / tau
    denominator[denominator == 0] = 1e-10
    l1 = FB * FR / denominator
    
    # Block matrix multiplication
    Nb_blocks = (n1 // n1y) * (n2 // n2y) if n1 % n1y == 0 and n2 % n2y == 0 else d*d
    FBR = BlockMM(n1y, n2y, Nb_blocks, n1y*n2y, l1)
    invW = BlockMM(n1y, n2y, Nb_blocks, n1y*n2y, F2B / denominator)
    
    # Compute invWBR
    divisor = invW + tau * 4
    divisor[divisor == 0] = 1e-10
    invWBR = FBR / divisor
    
    # Compute FCBinvWBR
    FCBinvWBR = np.zeros_like(FBC, dtype=np.complex128)
    
    block_rows = min(n1y, FBC.shape[0])
    block_cols = min(n2y, FBC.shape[1])
    
    for i in range(0, FCBinvWBR.shape[0], block_rows):
        for j in range(0, FCBinvWBR.shape[1], block_cols):
            i_end = min(i + block_rows, FCBinvWBR.shape[0])
            j_end = min(j + block_cols, FCBinvWBR.shape[1])
            
            FBC_block = FBC[i:i_end, j:j_end]
            
            # Get corresponding invWBR block
            invWBR_i = i // block_rows
            invWBR_j = j // block_cols
            
            if (invWBR_i * block_rows < invWBR.shape[0] and 
                invWBR_j * block_cols < invWBR.shape[1]):
                
                invWBR_i_end = min((invWBR_i + 1) * block_rows, invWBR.shape[0])
                invWBR_j_end = min((invWBR_j + 1) * block_cols, invWBR.shape[1])
                
                invWBR_block = invWBR[invWBR_i*block_rows:invWBR_i_end, 
                                      invWBR_j*block_cols:invWBR_j_end]
                
                if invWBR_block.shape == FBC_block.shape:
                    FCBinvWBR[i:i_end, j:j_end] = FBC_block * invWBR_block
    
    # Final solution
    FX = (FR - FCBinvWBR) / denominator / tau
    
    # Inverse Fourier transform
    x1 = np.real(np.fft.ifft2(FX))
    
    # Preserve original MRI range
    x1 = np.clip(x1, 0, 1)
    
    # Ensure we don't lose too much MRI information
    current_range = np.max(x1) - np.min(x1)
    original_range = np.max(x1k_original) - np.min(x1k_original)
    
    if current_range < 0.5 * original_range:
        # If we lost too much range, blend back with original
        blend_factor = 0.3
        x1 = (1 - blend_factor) * x1 + blend_factor * x1k_original
    
    print(f"  FSR_xirm_NL: Final x1 range=[{np.min(x1):.3f}, {np.max(x1):.3f}], "
          f"range preserved: {current_range/original_range*100:.1f}%")
    
    if display:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(x1k_original, cmap='gray')
        plt.title('Input MRI (before update)')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(update_mask, cmap='gray')
        plt.title(f'Update Mask\n({np.sum(update_mask)}/{x1k.size} pixels)')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(x1, cmap='gray')
        plt.title('Updated MRI')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    return x1

def BlockMM(nr, nc, Nb, m, x1):
    """Block matrix multiplication (unchanged)"""
    result = np.zeros((nr, nc), dtype=np.complex128)
    
    total_rows = x1.shape[0]
    total_cols = x1.shape[1]
    
    for i in range(0, total_rows, nr):
        for j in range(0, total_cols, nc):
            i_end = min(i + nr, total_rows)
            j_end = min(j + nc, total_cols)
            
            block = x1[i:i_end, j:j_end]
            
            if block.shape[0] < nr or block.shape[1] < nc:
                padded_block = np.zeros((nr, nc), dtype=np.complex128)
                padded_block[:block.shape[0], :block.shape[1]] = block
                result += padded_block
            else:
                result += block
    
    return result