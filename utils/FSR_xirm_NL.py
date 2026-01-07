import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import zoom

from utils.Link import Link
from utils.HXconv import HXconv

def blockproc(arr, block_shape, func):
    return func(arr)

def FSR_xirm_NL(x1k, y1, xus, gradY, B, d, c, F2D, tau, tau10, display):
    n1y, n2y = y1.shape
    n = xus.shape

    x1k_clip = np.clip(x1k, -1.0, 1.0)
    xus_clip = np.clip(xus, -1.0, 1.0)
    gradY_clip = np.clip(gradY, -1.0, 1.0)

    X = x1k_clip - 4 * (
        c[1] + 2*c[2]*x1k_clip + 3*c[3]*x1k_clip**2 + 4*c[4]*x1k_clip**3 +
        c[6]*gradY_clip + 2*c[7]*gradY_clip*x1k_clip + 3*c[8]*gradY_clip*x1k_clip**2 +
        c[10]*gradY_clip**2 + 2*c[11]*gradY_clip**2*x1k_clip + c[13]*gradY_clip**3
    ) * (xus_clip - Link(x1k_clip, c))
    
    X = np.clip(X, -1e3, 1e3)

    STy = np.zeros(n, dtype=float)
    target_shape = STy[::d, ::d].shape
    y1_resized = zoom(y1, (target_shape[0]/n1y, target_shape[1]/n2y), order=3)
    STy[::d, ::d] = y1_resized

    FB, FBC, F2B, _ = HXconv(STy, B, 'Hx')

    FR = FBC * fft2(STy) + fft2(2 * tau10 * X)

    l1 = FB * FR / (F2D + 100 * tau10 / tau)
    FBR = BlockMM(n1y, n2y, d*d, n1y*n2y, l1)

    invW = BlockMM(n1y, n2y, d*d, n1y*n2y, F2B / (F2D + 100 * tau10 / tau))
    invWBR = FBR / (invW + tau * 4)

    invWBR_resized = zoom(invWBR, (FBC.shape[0] / invWBR.shape[0],
                                   FBC.shape[1] / invWBR.shape[1]),
                           order=1)

    FCBinvWBR = FBC * invWBR_resized

    FX = (FR - FCBinvWBR) / (F2D + 100 * tau10 / tau) / tau

    x1 = np.real(ifft2(FX))

    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(x1, cmap='gray')
        plt.show()

    return x1


def BlockMM(nr, nc, Nb, m, x1):
    x1_flat = x1.flatten()
    if x1_flat.size % m != 0:
        Nb = x1_flat.size // m
    x1_blocks = x1_flat[:m*Nb].reshape((m, Nb))
    x = np.sum(x1_blocks, axis=1)
    x = x.reshape((nr, nc))
    print("BlockMM output shape:", x.shape)
    return x
