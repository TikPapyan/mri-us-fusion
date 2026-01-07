import numpy as np

def dtd(u):
    u = np.asarray(u)
    d = np.zeros(u.shape)

    u_flat = u.flatten()
    d_flat = d.flatten()

    d_flat[1:-1] = 2 * u_flat[1:-1] - u_flat[:-2] - u_flat[2:]
    d_flat[0] = u_flat[0] - u_flat[1]
    d_flat[-1] = u_flat[-1] - u_flat[-2]

    return d_flat.reshape(u.shape)
