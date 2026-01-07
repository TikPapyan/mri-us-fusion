import numpy as np

from utils.Link import Link
from utils.d1 import d1

def f1_NL(x, y2, x1k, x2k, c, gama, tau1, tau2, tau3):
    X = x2k - 4 * (x2k - Link(x1k, c))

    term1 = tau1 * np.sum(gama * np.exp(y2 - x) - (y2 - x))
    term2 = tau2 / 2 * (np.linalg.norm(d1(x)) ** 2)
    term3 = tau3 * np.linalg.norm(x - X)

    f = term1 + term2 + term3
    return f
