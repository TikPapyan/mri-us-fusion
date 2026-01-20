import numpy as np

def d2(u):
    """
    Returns the derivative in the second direction (vertical)
    """
    d = np.zeros_like(u)
    d[:-1] = u[1:] - u[:-1]
    return d