import numpy as np
from scipy import ndimage

def HXconv(x, B, conv=None):
    """
    Convolution operations in frequency domain
    """
    m, n = x.shape
    m0, n0 = B.shape
    
    # Pad B to match x dimensions
    pad_top = (m - m0 + 1) // 2
    pad_bottom = (m - m0 - 1) // 2
    pad_left = (n - n0 + 1) // 2
    pad_right = (n - n0 - 1) // 2
    
    Bpad = np.pad(B, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    # Shift and compute FFT
    Bpad = np.fft.ifftshift(Bpad)
    BF = np.fft.fft2(Bpad)
    BCF = np.conj(BF)
    B2F = np.abs(BF)**2
    
    if conv is None:
        return BF, BCF, B2F
    elif conv == 'Hx':
        # Ensure x is complex for FFT
        x_complex = x.astype(np.complex128) if np.isrealobj(x) else x
        y = np.real(np.fft.ifft2(BF * np.fft.fft2(x_complex)))
        return BF, BCF, B2F
    elif conv == 'HTx':
        x_complex = x.astype(np.complex128) if np.isrealobj(x) else x
        y = np.real(np.fft.ifft2(BCF * np.fft.fft2(x_complex)))
        return BF, BCF, B2F
    elif conv == 'HTHx':
        x_complex = x.astype(np.complex128) if np.isrealobj(x) else x
        y = np.real(np.fft.ifft2(B2F * np.fft.fft2(x_complex)))
        return BF, BCF, B2F
    else:
        raise ValueError("conv must be 'Hx', 'HTx', or 'HTHx'")