import imageio.v2 as imageio
import numpy as np

mri = imageio.imread("images/Data1/irm.png")  # adjust path
us = imageio.imread("images/Data1/us.png")

print(f"MRI shape: {mri.shape}, dtype: {mri.dtype}, min: {mri.min()}, max: {mri.max()}")
print(f"US shape: {us.shape}, dtype: {us.dtype}, min: {us.min()}, max: {us.max()}")
