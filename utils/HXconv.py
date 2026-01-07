import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from numpy import pad

def HXconv(x, B, conv=None):
    m, n = x.shape
    m0, n0 = B.shape

    Bpad = pad(
        B,
        (
            (int(np.floor((m - m0 + 1) / 2)), int(np.ceil((m - m0 - 1) / 2))),
            (int(np.floor((n - n0 + 1) / 2)), int(np.ceil((n - n0 - 1) / 2)))
        ),
        mode='constant'
    )

    Bpad = fftshift(Bpad)
    BF = fft2(Bpad)
    BCF = np.conj(BF)
    B2F = np.abs(BF) ** 2

    if conv is None:
        return BF, BCF, B2F, None, Bpad

    if conv == 'Hx':
        y = np.real(ifft2(BF * fft2(x)))
    elif conv == 'HTx':
        y = np.real(ifft2(BCF * fft2(x)))
    elif conv == 'HTHx':
        y = np.real(ifft2(B2F * fft2(x)))
    else:
        raise ValueError("Unknown convolution mode")

    return BF, BCF, B2F, y
