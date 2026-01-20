import numpy as np

def dtd(u):
    """
    Second derivative operator
    """
    d = np.zeros_like(u)
    if len(u.shape) == 1:
        # 1D case
        n = len(u)
        if n > 2:
            d[1:-1] = 2 * u[1:-1] - u[:-2] - u[2:]
        if n > 0:
            d[0] = u[0] - u[1]
            d[-1] = u[-1] - u[-2]
    else:
        # 2D case - apply along each dimension
        d = np.zeros_like(u)
        # Apply in first dimension
        if u.shape[0] > 2:
            d[1:-1, :] = 2 * u[1:-1, :] - u[:-2, :] - u[2:, :]
        if u.shape[0] > 0:
            d[0, :] = u[0, :] - u[1, :]
            d[-1, :] = u[-1, :] - u[-2, :]
        
        # Apply in second dimension
        if u.shape[1] > 2:
            d[:, 1:-1] += 2 * u[:, 1:-1] - u[:, :-2] - u[:, 2:]
        if u.shape[1] > 0:
            d[:, 0] += u[:, 0] - u[:, 1]
            d[:, -1] += u[:, -1] - u[:, -2]
        
        # Average the two directions
        d = d / 2
    
    return d