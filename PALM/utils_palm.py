# --- Librairies ---
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks


# --- Fichiers ---
from ResizeRight.resize_right import resize
from ResizeRight.interp_methods import cubic
from matlab_tools import blockproc, fspecial

# --- Fonctions ---
def d1(u):
    """Dérivée en abscisse"""
    return convolve1d(u, [1, -1], axis=1, mode='nearest')
    
def d2(u):
    """Dérivée en ordonnée"""
    return convolve1d(u, [1, -1], axis=0, mode='nearest')

def dtd(u):
    """Laplacien"""
    return convolve1d(u, np.array([1, -2, 1]), mode='nearest')

def Link(x1, c):
    """Approximation polynomiale du lien IRM/US"""
    Jx = convolve2d(x1, [[-1, 1]], mode='same', boundary='symm')
    Jy = convolve2d(x1, [[-1], [1]], mode='same', boundary='symm')
    gradY = np.sqrt(Jx**2 + Jy**2)
    x1_2 = x1**2
    x1_3 = x1 * x1_2
    x1_4 = x1 * x1_3
    gradY2 = gradY**2
    gradY3 = gradY * gradY2
    gradY4 = gradY2**2
    terms = (
        c[0], 
        c[1]*x1,
        c[2]*x1_2,
        c[3]*x1_3,
        c[4]*x1_4,
        c[5]*gradY,
        c[6]*gradY*x1,
        c[7]*gradY*x1_2,
        c[8]*gradY*x1_3,
        c[9]*gradY2,
        c[10]*gradY2*x1,
        c[11]*gradY2*x1_2,
        c[12]*gradY3,
        c[13]*gradY3*x1,
        c[14]*gradY4
    )
    return sum(terms)

def f1_NL(x, y2, x1k, x2k, c, gamma, tau1, tau2, tau3):
    """Critère"""
    X = x2k - 4 * (x2k - Link(x1k, c))
    return tau1 * np.sum(gamma * np.exp(y2 - x) - (y2 - x)) + (tau2 / 2) * np.linalg.norm(d1(x))**2 + tau3 * np.linalg.norm(x - X)

def gradf1_NL(x, y2, x1k, x2k, c, gamma, tau1, tau2, tau3):
    """Gradient du critère"""
    X = x2k - 4 * (x2k - Link(x1k, c))
    return tau1 * (gamma - np.exp(y2 - x)) + tau2 * dtd(x) + (tau3 / 2) * (x - X)

def Descente_grad_xus_NL(y2, x1k, x2k, c, gamma, tau1, tau2, tau3, alpha=10): #alpha = 0.2
    # Mise en forme
    n1, n2 = y2.shape
    c1 = 1e-8
    y2  = y2.reshape((n1 * n2, 1))
    x1k = x1k.reshape((n1 * n2, 1))
    x2k = x2k.reshape((n1 * n2, 1))
    x0 = y2 + c1
    # Fonctions objectif et gradient
    f = lambda x: f1_NL(x, y2, x1k, x2k, c, gamma, tau1, tau2, tau3)
    gradf = lambda x: gradf1_NL(x, y2, x1k, x2k, c, gamma, tau1, tau2, tau3)
    # Critères d’arrêt
    tol, maxiter, dxmin, gnorm = 1e-6, 100, 1e-2, np.inf
    x = x0.copy()
    niter = 0
    dx = np.inf
    while gnorm >= tol and niter <= maxiter and dx >= dxmin:
        g = gradf(x)
        gnorm = np.linalg.norm(g)
        xnew = x - alpha * g
        if not np.all(np.isfinite(xnew)):
            print(f"Nombre d'itérations : {niter}")
            raise ValueError("x contient des valeurs infinies ou NaN")
        niter += 1
        dx = np.linalg.norm(xnew - x)
        x = xnew
    fopt = f(x)
    niter -= 1
    x2 = x.reshape((n1, n2))
    return x2, fopt, niter

def HXconv(x, B, conv='Hx'):
    m, n = x.shape
    m0, n0 = B.shape
    # Padding symétrique (pre puis post)
    pre_pad  = (np.floor([(m - m0 + 1) / 2, (n - n0 + 1) / 2])).astype(int)
    post_pad = (np.round([(m - m0 - 1) / 2, (n - n0 - 1) / 2])).astype(int)
    Bpad = np.pad(B, ((pre_pad[0], post_pad[0]), (pre_pad[1], post_pad[1])), mode='constant')
    Bpad = fftshift(Bpad)
    BF = fft2(Bpad)
    BCF = np.conj(BF)
    B2F = np.abs(BF)**2
    if conv == 'Hx':
        y = np.real(ifft2(BF * fft2(x)))
    elif conv == 'HTx':
        y = np.real(ifft2(BCF * fft2(x)))
    elif conv == 'HTHx':
        y = np.real(ifft2(B2F * fft2(x)))
    else:
        raise NotImplementedError("conv doit être l'une des chaînes : 'Hx', 'HTx', 'HTHx'")
    return BF, BCF, B2F, y, Bpad

def BlockMM(nr, nc, Nb, m, x1):
    blocks = view_as_blocks(x1, block_shape=(nr, nc))
    Nb1, Nb2, _, _ = blocks.shape
    return np.sum(blocks.reshape(Nb1 * Nb2, m).T, axis=1).reshape(nr, nc)

def FSR_xirm_NL(x1k, y1, xus, gradY, B, d, c, F2D, tau, tau10):
    n1y, n2y = y1.shape
    x1k2 = x1k**2
    x1k3 = x1k*x1k2
    gradY2 = gradY**2
    gradY3 = gradY*gradY2
    X = x1k - 4 * (
        c[1] + 2 * c[2] * x1k + 3 * c[3] * x1k2 + 4 * c[4] * x1k3 +
        c[6] * gradY + 2 * c[7] * gradY * x1k + 3 * c[8] * gradY * x1k2 +
        c[10] * gradY2 + 2 * c[11] * gradY2 * x1k + c[13] * gradY3
    ) * (xus - Link(x1k, c))
    STy = np.zeros_like(xus)
    STy[::d, ::d] = y1
    FB, FBC, F2B, _, _ = HXconv(STy, B, 'Hx')
    FR = FBC * fft2(STy) + fft2(2*tau10*X.reshape(STy.shape))
    l1 = FB * FR / (F2D + 100 * tau10 / tau)
    FBR = BlockMM(n1y, n2y, d*d, n1y*n2y, l1)
    invW = BlockMM(n1y, n2y, d*d, n1y*n2y, F2B / (F2D + 100*tau10/tau))
    invWBR = FBR/(invW + tau*4)
    def fun(block):
        return block * invWBR
    FCBinvWBR = blockproc(FBC, (n1y, n2y), fun)
    FX = (FR - FCBinvWBR) / (F2D + 100*tau10/tau) / tau
    x1 = np.real(ifft2(FX))
    return x1

def estimate_c(irm, us, d=6):
    # Normalisation des images
    y1 = irm / np.linalg.norm(irm)
    y2 = us / np.linalg.norm(us)
    n1, n2 = y2.shape
    # Redimensionnement bicubique de y1
    yint = resize(y1, scale_factors=(d,d), interp_method=cubic)
    # Convolution 2D avec padding 'same' (taille de sortie égale à yint)
    Jx = convolve2d(yint, np.array([[-1, 1]]), mode='same', boundary='symm')
    Jy = convolve2d(yint, np.array([[-1], [1]]), mode='same', boundary='symm')
    # Calcul du gradient
    gradY = np.sqrt(Jx**2 + Jy**2)
    # Vectorisation (aplatissement en vecteur colonne)
    yi = yint.reshape(n1 * n2, 1)
    yu = y2.reshape(n1 * n2, 1)
    dyi = gradY.reshape(n1 * n2, 1)
    # Construction de la matrice A (polynôme + termes en gradient)
    A = np.hstack([
        np.ones_like(yi), 
        yi, 
        yi**2, 
        yi**3, 
        yi**4,
        dyi, 
        dyi * yi,
        dyi * yi**2,
        dyi * yi**3,
        dyi**2,
        dyi**2 * yi,
        dyi**2 * yi**2,
        dyi**3,
        dyi**3 * yi,
        dyi**4
    ])
    # Calcul de la pseudo-inverse et estimation des coefficients c
    cest = np.linalg.pinv(A) @ yu
    # Calcul de xu = A * cest
    xu = A @ cest
    xu = xu.reshape((n1, n2))
    return cest.flatten(), xu

def FusionPALM(y1, y2, c, tau1, tau2, tau3, tau4, d, m_iteration=10):
    n1, n2 = y2.shape
    B = fspecial('gaussian', (5,5), 4)
    
    # Super-résolution bicubique de y1 (IRM)
    yint = resize(y1, scale_factors=(d,d), interp_method=cubic)
    
    # Gradient IRM
    Jx = convolve2d(yint, np.array([[-1, 1]]), mode='same', boundary='symm')
    Jy = convolve2d(yint, np.array([[-1], [1]]), mode='same', boundary='symm')
    gradY = np.sqrt(Jx**2 + Jy**2)
    
    # Paramètres
    #m_iteration = 10
    gamma = 1e-3
    
    # Opérateurs de différence horizontale et verticale
    dh = np.zeros((n1, n2))
    dh[0, 0] = 1
    dh[0, 1] = -1
    dv = np.zeros((n1, n2))
    dv[0, 0] = 1
    dv[1, 0] = -1
    
    # Transformées de Fourier
    FDH = fft2(dh)
    F2DH = np.abs(FDH)**2
    FDV = fft2(dv)
    FDV = np.conj(FDV)
    F2DV = np.abs(FDV)**2
    c1 = 1e-8
    F2D = F2DH + F2DV + c1
    
    # Réglages des paramètres tau
    taup = 1
    tau = taup      # IRM (influence TV)
    tau10 = tau1    # IRM (influence echo)
    tau1 = tau2     # US (influence observation)
    tau2 = tau3     # US (influence TV)
    tau3 = tau4     # US (influence IRM)
    
    x2 = y2 + c1
    x1 = yint
    
    # Boucle d'itération PALM
    for i in range(m_iteration):
        x1 = FSR_xirm_NL(x1, y1, x2, gradY, B, d, c, F2D, tau, tau10)
        x2, _, _ = Descente_grad_xus_NL(y2, x1, x2, c, gamma, tau1, tau2, tau3)
    return x2