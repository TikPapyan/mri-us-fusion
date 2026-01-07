import numpy as np

def d2(u):
    u = np.asarray(u)
    d = np.zeros(u.shape)

    u_flat = u.flatten()
    d_flat = d.flatten()

    d_flat[:-1] = u_flat[1:] - u_flat[:-1]

    return d_flat.reshape(u.shape)
