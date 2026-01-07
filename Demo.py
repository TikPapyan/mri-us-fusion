import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from utils.FusionPALM import FusionPALM

# **************************************************************************
# Demo.py - Fusion of MRI and US images
# **************************************************************************

# Close all figures
plt.close('all')

# %% Load images
irm = np.array(Image.open("images/irm.png").convert("L"), dtype=float)
us  = np.array(Image.open("images/us.png").convert("L"), dtype=float)

# %% Image normalization
ym = irm / np.max(irm)
yu = us / np.max(us)

# %% Compute polynomial coefficients
from estimate_c import cest
c = np.abs(cest)  # ensure positive

# %% Display observations
# plt.figure()
# plt.imshow(ym, cmap='gray')
# plt.title("Normalized MRI")
# plt.show()

# plt.figure()
# plt.imshow(yu, cmap='gray')
# plt.title("Normalized US")
# plt.show()

# %% Initialization of PALM
d = 6
xm0 = zoom(ym, d, order=3)

# NOTE: MATLAB denoisingNetwork / denoiseImage equivalent
# Currently just using normalized US image
xu0 = yu

# %% Regularization parameters
tau1 = 1e-12  # MRI echo
tau2 = 1e-4   # US observation
tau3 = 2e-4   # US TV
tau4 = 1e-4   # US MRI

# %% Run PALM fusion
x2 = FusionPALM(ym, xu0, c, tau1, tau2, tau3, tau4, plot_fused_image=True)

# %% Display result
plt.figure()
plt.imshow(x2, cmap='gray')
plt.title("Fused Image")
plt.show()
