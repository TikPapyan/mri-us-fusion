import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import mean_squared_error
from skimage.transform import resize
import os

def normalize(img):
    img = img.astype(np.float64)
    img = img - img.min()
    img = img / img.max()
    return img

def evaluate(ref, img):
    psnr = peak_signal_noise_ratio(ref, img, data_range=1)
    ssim = structural_similarity(ref, img, data_range=1)
    return psnr, ssim

# ----------------------------
# Load images
# ----------------------------

# Load original images
mri = normalize(imageio.imread("PALM/results/mri.png"))
us = normalize(imageio.imread("PALM/results/us.png"))

# Load PALM results
palm1 = normalize(imageio.imread("PALM/results/baseline_palm1.png"))
palm5 = normalize(imageio.imread("PALM/results/baseline_palm5.png"))
palm10 = normalize(imageio.imread("PALM/results/palm_10.png"))

# Load DDFM result - FIXED: define the path first, then check if it exists
ddfm_path = "PALM/results/ddfm_baseline.png"
ddfm = None

if os.path.exists(ddfm_path):
    print(f"Found DDFM image at: {ddfm_path}")
    ddfm = normalize(imageio.imread(ddfm_path))
    # Convert RGB to grayscale if needed
    if len(ddfm.shape) == 3:
        ddfm = np.mean(ddfm, axis=2)
    print(f"DDFM image loaded, shape: {ddfm.shape}")
else:
    print(f"Warning: DDFM image not found at {ddfm_path}")
    ddfm = None

# Get target shape from PALM1 (all PALM results have same size)
target_shape = palm1.shape
print(f"Target shape (PALM): {target_shape}")

# Resize original images to match PALM resolution
if mri.shape != target_shape:
    mri = resize(mri, target_shape, anti_aliasing=True)
    print(f"MRI resized to: {mri.shape}")

if us.shape != target_shape:
    us = resize(us, target_shape, anti_aliasing=True)
    print(f"US resized to: {us.shape}")

# Resize DDFM if needed
if ddfm is not None and ddfm.shape != target_shape:
    ddfm = resize(ddfm, target_shape, anti_aliasing=True)
    print(f"DDFM resized to: {ddfm.shape}")

# Create results dictionary
results = {
    "PALM1": palm1,
    "PALM5": palm5,
    "PALM10": palm10,
}

if ddfm is not None:
    results["DDFM"] = ddfm

print("\n========================")
print("Fusion Evaluation")
print("========================\n")

for name, img in results.items():
    psnr_mri, ssim_mri = evaluate(mri, img)
    psnr_us, ssim_us = evaluate(us, img)
    
    print(name)
    print(f"  vs MRI -> PSNR: {psnr_mri:.4f} dB, SSIM: {ssim_mri:.4f}")
    print(f"  vs US  -> PSNR: {psnr_us:.4f} dB, SSIM: {ssim_us:.4f}")
    print()

# ----------------------------
# REDUNDANCY TEST
# ----------------------------

print("\n========================")
print("PALM5 vs PALM10 comparison")
print("========================\n")

mse_5_10 = mean_squared_error(palm5, palm10)
psnr_5_10 = peak_signal_noise_ratio(palm5, palm10, data_range=1)
ssim_5_10 = structural_similarity(palm5, palm10, data_range=1)

print(f"MSE: {mse_5_10:.8f}")
print(f"PSNR: {psnr_5_10:.4f} dB")
print(f"SSIM: {ssim_5_10:.4f}")

diff_5_10 = np.abs(palm5 - palm10)
print(f"Max pixel difference: {diff_5_10.max():.6f}")
print(f"Mean difference: {diff_5_10.mean():.8f}")

# ----------------------------
# DDFM vs PALM Comparison
# ----------------------------

if ddfm is not None:
    print("\n========================")
    print("DDFM vs PALM Comparison")
    print("========================\n")
    
    # Compare DDFM with PALM5
    mse_ddfm_palm5 = mean_squared_error(ddfm, palm5)
    psnr_ddfm_palm5 = peak_signal_noise_ratio(ddfm, palm5, data_range=1)
    ssim_ddfm_palm5 = structural_similarity(ddfm, palm5, data_range=1)
    
    print(f"DDFM vs PALM5:")
    print(f"  MSE: {mse_ddfm_palm5:.8f}")
    print(f"  PSNR: {psnr_ddfm_palm5:.4f} dB")
    print(f"  SSIM: {ssim_ddfm_palm5:.4f}")
    print()
    
    # Compare DDFM with PALM1
    mse_ddfm_palm1 = mean_squared_error(ddfm, palm1)
    psnr_ddfm_palm1 = peak_signal_noise_ratio(ddfm, palm1, data_range=1)
    ssim_ddfm_palm1 = structural_similarity(ddfm, palm1, data_range=1)
    
    print(f"DDFM vs PALM1:")
    print(f"  MSE: {mse_ddfm_palm1:.8f}")
    print(f"  PSNR: {psnr_ddfm_palm1:.4f} dB")
    print(f"  SSIM: {ssim_ddfm_palm1:.4f}")

# ----------------------------
# Visualization
# ----------------------------

# Determine number of subplots based on available results
n_methods = len(results)
fig, axes = plt.subplots(1, n_methods + 2, figsize=(3*(n_methods+2), 6))

# Plot original images
axes[0].imshow(mri, cmap='gray')
axes[0].set_title("MRI")
axes[0].axis('off')

axes[1].imshow(us, cmap='gray')
axes[1].set_title("US")
axes[1].axis('off')

# Plot fusion results
for idx, (name, img) in enumerate(results.items()):
    axes[idx + 2].imshow(img, cmap='gray')
    axes[idx + 2].set_title(name)
    axes[idx + 2].axis('off')

plt.tight_layout()
plt.savefig("fusion_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ----------------------------
# Summary Table
# ----------------------------

print("\n========================")
print("SUMMARY TABLE")
print("========================\n")
print(f"{'Method':<10} | {'vs MRI PSNR':<12} | {'vs MRI SSIM':<12} | {'vs US PSNR':<12} | {'vs US SSIM':<12}")
print("-" * 65)

for name, img in results.items():
    psnr_mri, ssim_mri = evaluate(mri, img)
    psnr_us, ssim_us = evaluate(us, img)
    print(f"{name:<10} | {psnr_mri:<12.4f} | {ssim_mri:<12.4f} | {psnr_us:<12.4f} | {ssim_us:<12.4f}")

# ----------------------------
# Difference Map Visualization
# ----------------------------

if ddfm is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Difference between DDFM and PALM5
    diff_ddfm_palm5 = np.abs(ddfm - palm5)
    im1 = axes[0].imshow(diff_ddfm_palm5, cmap='hot', vmin=0, vmax=0.5)
    axes[0].set_title("|DDFM - PALM5|")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Difference between DDFM and PALM1
    diff_ddfm_palm1 = np.abs(ddfm - palm1)
    im2 = axes[1].imshow(diff_ddfm_palm1, cmap='hot', vmin=0, vmax=0.5)
    axes[1].set_title("|DDFM - PALM1|")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference between PALM5 and PALM1
    diff_palm5_palm1 = np.abs(palm5 - palm1)
    im3 = axes[2].imshow(diff_palm5_palm1, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title("|PALM5 - PALM1|")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("difference_maps.png", dpi=150, bbox_inches='tight')
    plt.show()