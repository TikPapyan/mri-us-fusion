import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2
from scipy.ndimage import zoom

from utils.FSR_xirm_NL import FSR_xirm_NL
from utils.Descente_grad_xus_NL import Descente_grad_xus_NL

def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def FusionPALM(y1, y2, c, tau1, tau2, tau3, tau4, plot_fused_image):
    n1, n2 = y2.shape

    B = gaussian_kernel(5, 4)

    scale_x = y2.shape[0] / y1.shape[0]
    scale_y = y2.shape[1] / y1.shape[1]
    yint = zoom(y1, (scale_x, scale_y), order=3)

    print("yint.shape =", yint.shape, "x2.shape =", y2.shape)

    Jx = convolve2d(yint, [[-1, 1]], mode='same')
    Jy = convolve2d(yint, [[-1], [1]], mode='same')
    gradY = np.sqrt(Jx**2 + Jy**2)

    m_iteration = 10
    gama = 1e-3

    dh = np.zeros((n1, n2))
    dh[0, 0] = 1
    dh[0, 1] = -1

    dv = np.zeros((n1, n2))
    dv[0, 0] = 1
    dv[1, 0] = -1

    FDH = fft2(dh)
    F2DH = np.abs(FDH)**2

    FDV = fft2(dv)
    F2DV = np.abs(FDV)**2

    c1 = 1e-8
    F2D = F2DH + F2DV + c1

    taup = 1
    tau = taup
    tau10 = tau1
    tau1 = tau2
    tau2 = tau3
    tau3 = tau4

    d = 6
    x2 = y2 + c1
    x1 = yint

    for _ in range(m_iteration):
        x1 = FSR_xirm_NL(x1, y1, x2, gradY, B, d, c, F2D, tau, tau10, False)
        x2, _, _ = Descente_grad_xus_NL(y2, x1, x2, c, gama, tau1, tau2, tau3, False, 0.01)

    if plot_fused_image:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(x2, cmap='gray')
        plt.show()

    return x2
