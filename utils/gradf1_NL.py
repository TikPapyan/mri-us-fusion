import numpy as np

from utils.Link import Link
from utils.dtd import dtd

def gradf1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3):
    X = x2k - 4 * (x2k - Link(x1k, c))
    gradf = tau1 * (gama - np.exp(y2 - x)) + tau2 * dtd(x) + tau3 / 2 * (x - X)
    X = np.clip(x2k - 4*(x2k - Link(x1k,c)), -1e6, 1e6)
    return gradf
