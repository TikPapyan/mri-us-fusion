import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import zoom
from PIL import Image

# %% Input images
# assumes `irm` and `us` already exist in workspace

# %% Load images

irm = np.array(Image.open("images/irm.png").convert("L"), dtype=float)
us  = np.array(Image.open("images/us.png").convert("L"), dtype=float)

y1 = irm / np.linalg.norm(irm)
y2 = us / np.linalg.norm(us)

n1, n2 = y2.shape

# Make yint the same shape as y2
scale_x = y2.shape[0] / y1.shape[0]
scale_y = y2.shape[1] / y1.shape[1]
yint = zoom(y1, (scale_x, scale_y), order=3)

print("yint.shape =", yint.shape, "y2.shape =", y2.shape)

# compute gradient
Jx = convolve2d(yint, [[-1, 1]], mode='same')
Jy = convolve2d(yint, [[-1], [1]], mode='same')
gradY = np.sqrt(Jx**2 + Jy**2)

# Image vectorization
yi = yint.reshape(n1 * n2, 1)
yu = y2.reshape(n1 * n2, 1)
dyi = gradY.reshape(n1 * n2, 1)

# Compute matrix A
A = np.hstack([
    np.ones((n1 * n2, 1)),
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

# %% Pseudo inverse
cest = np.linalg.pinv(A) @ yu
cest = np.clip(cest, -1e3, 1e3)

# %% Compute xu = f(ym)
xu = A @ cest
xu = xu.reshape(n1, n2)
