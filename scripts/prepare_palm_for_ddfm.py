# prepare_palm_for_ddfm.py
import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize
import os

def prepare_palm_output(palm_path, output_path, target_size=(256, 256)):
    """Resize PALM output to 256x256 and save for DDFM"""
    
    print(f"Preparing {palm_path}...")
    
    # Load PALM output
    img = imageio.imread(palm_path)
    
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    print(f"  Original shape: {img.shape}")
    
    # Resize to 256x256
    img_resized = resize(img, target_size, anti_aliasing=True, preserve_range=True)
    img_resized = img_resized.astype(np.uint8)
    
    print(f"  Resized to: {img_resized.shape}")
    
    # Save
    imageio.imwrite(output_path, img_resized)
    print(f"  Saved to: {output_path}")
    
    return img_resized

def main():
    # Create output directory
    os.makedirs("experiments/inputs", exist_ok=True)
    
    # Prepare PALM1
    prepare_palm_output(
        "PALM/results/baseline_palm1.png",
        "experiments/inputs/palm1_256.png"
    )
    
    # Prepare PALM5
    prepare_palm_output(
        "PALM/results/baseline_palm5.png",
        "experiments/inputs/palm5_256.png"
    )
    
    print("\n✅ PALM outputs prepared for DDFM!")

if __name__ == "__main__":
    main()