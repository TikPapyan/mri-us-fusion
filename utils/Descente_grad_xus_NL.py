import numpy as np

from utils.f1_NL import f1_NL
from utils.gradf1_NL import gradf1_NL

def Descente_grad_xus_NL(y2, x1k, x2k, c, gama, tau1, tau2, tau3, display, alpha):
    n1, n2 = y2.shape
    c1 = 1e-8

    y2 = y2.reshape(n1 * n2, 1)
    x1k = x1k.reshape(n1 * n2, 1)
    x2k = x2k.reshape(n1 * n2, 1)

    x0 = y2 + c1

    f = lambda x: f1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3)
    gradf = lambda x: gradf1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3)

    tol = 1e-6
    maxiter = 100
    dxmin = 1e-2

    gnorm = np.inf
    x = x0
    niter = 0
    dx = np.inf

    while (gnorm >= tol) and (niter <= maxiter) and (dx >= dxmin):
        g = gradf(x)
        xnew = x - alpha * g

        if not np.all(np.isfinite(xnew)):
            print(f"Number of iterations: {niter}")
            raise ValueError("x is inf or NaN")

        niter += 1
        dx = np.linalg.norm(xnew - x)
        x = xnew

    fopt = f(x)
    niter -= 1

    x = x.reshape(n1, n2)
    x2 = x

    if display is True:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(x2, cmap='gray')
        plt.show()

    return x2, fopt, niter
