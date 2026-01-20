import numpy as np

def d1(u):
    """
    Returns the derivative in the first direction (horizontal)
    """
    d = np.zeros_like(u)
    d[:-1] = u[1:] - u[:-1]
    return d