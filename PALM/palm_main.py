#**************************************************************************
# Denoising Diffusion Model for Multi-modality Image Fusion with Proximal
# Alternating Linearized Minimization algorithm
# Author: Tom Longin (2025 June)
# University of Toulouse, IRIT
# Email: tom.longin@irit.fr
#
# Copyright (2025): Tom Longin
# 
# Permission to use, copy, modify, and distribute this software for
# any purpose without fee is hereby granted, provided that this entire
# notice is included in all copies of any software which is or includes
# a copy or modification of this software and in all copies of the
# supporting documentation for such software.
# This software is being provided "as is", without any express or
# implied warranty.  In particular, the authors do not make any
# representation or warranty of any kind concerning the merchantability
# of this software or its fitness for any particular purpose."
#**************************************************************************

# --- Libraries ---
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from imageio import imwrite

# --- Project files (relative imports within PALM package) ---
from .matlab_tools import load_dncnn
from .utils_palm import estimate_c, FusionPALM


def show_image(img, title='Image'):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def chooseDataset():
    palm_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(palm_dir)
    data_dir = os.path.join(project_root, "data")

    irm_path = os.path.join(data_dir, "irm.mat")
    us_path = os.path.join(data_dir, "us.mat")

    irm_data = scipy.io.loadmat(irm_path)
    us_data = scipy.io.loadmat(us_path)

    irm = irm_data['irm'].astype(np.float64)
    us = us_data['us'].astype(np.float64)

    show_image(irm, 'MRI')
    show_image(us, 'Ultrasound')

    return irm, us


def solve_PALM(irm, us, m_iteration=1):

    # polynomial coefficients
    cest, _ = estimate_c(irm, us)
    c = np.abs(cest)

    # normalization
    ym = irm.astype(np.float64) / irm.max()
    yu = us.astype(np.float64) / us.max()

    # PALM parameters
    d = 6

    # US denoising with DnCNN
    xu0 = load_dncnn(yu)

    tau1 = 1e-12
    tau2 = 1e-15
    tau3 = 2e-4
    tau4 = 1e-4

    # run PALM
    x2 = FusionPALM(
        ym,
        xu0,
        c,
        tau1,
        tau2,
        tau3,
        tau4,
        d,
        m_iteration
    )

    return x2


def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img


def save_result(img, name):
    os.makedirs("results", exist_ok=True)

    img_norm = normalize(img)

    imwrite(
        f"results/{name}.png",
        (img_norm * 255).astype(np.uint8)
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

irm, us = chooseDataset()

print("Running PALM...")

res1  = solve_PALM(irm, us, m_iteration=1)
res5  = solve_PALM(irm, us, m_iteration=5)
res10 = solve_PALM(irm, us, m_iteration=10)

print("Saving results...")

save_result(res1, "baseline_palm1")
save_result(res5, "baseline_palm5")
save_result(res10, "palm_10")
save_result(irm, "mri")
save_result(us, "us")

print("Done.")

# visualization

plt.figure(figsize=(12,6))

plt.subplot(1,5,1)
plt.imshow(irm, cmap='gray')
plt.title("MRI")
plt.axis('off')

plt.subplot(1,5,2)
plt.imshow(us, cmap='gray')
plt.title("US")
plt.axis('off')

plt.subplot(1,5,3)
plt.imshow(res1, cmap='gray')
plt.title("PALM1")
plt.axis('off')

plt.subplot(1,5,4)
plt.imshow(res5, cmap='gray')
plt.title("PALM5")
plt.axis('off')

plt.subplot(1,5,5)
plt.imshow(res10, cmap='gray')
plt.title("PALM10")
plt.axis('off')

plt.tight_layout()
plt.show()