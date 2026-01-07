import numpy as np
from scipy.signal import convolve2d

def Link(x1, c):
    x1_clip = np.clip(x1, -1.0, 1.0)
    Jx = convolve2d(x1_clip, [[-1, 1]], mode='same')
    Jy = convolve2d(x1_clip, [[-1], [1]], mode='same')
    gradY = np.sqrt(Jx**2 + Jy**2)
    gradY = np.clip(gradY, -1.0, 1.0)

    x2 = (
        c[0]
        + c[1] * x1_clip
        + c[2] * x1_clip**2
        + c[3] * x1_clip**3
        + c[4] * x1_clip**4
        + c[5] * gradY
        + c[6] * gradY * x1_clip
        + c[7] * gradY * x1_clip**2
        + c[8] * gradY * x1_clip**3
        + c[9] * gradY**2
        + c[10] * gradY**2 * x1_clip
        + c[11] * gradY**2 * x1_clip**2
        + c[12] * gradY**3
        + c[13] * gradY**3 * x1_clip
        + c[14] * gradY**4
    )

    x2 = np.clip(x2, -1e3, 1e3)

    return x2
