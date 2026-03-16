#!/usr/bin/env python3
"""
MRI-US Fusion: Unified Master Script

Usage:
    python scripts/run.py --help
    python scripts/run.py --palm
    python scripts/run.py --ddfm
    python scripts/run.py --hybrids
    python scripts/run.py --all
    python scripts/run.py --evaluate
"""

import os
import sys
import argparse
import subprocess
import time
import yaml
import numpy as np
import imageio.v2 as imageio
import pandas as pd
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

_PALM_IMPORTED = False


def _import_palm():
    global _PALM_IMPORTED
    global load_dncnn, fspecial, DnCNN
    global estimate_c, Link, d1, d2, dtd, HXconv, BlockMM, FSR_xirm_NL, Descente_grad_xus_NL, FusionPALM

    if _PALM_IMPORTED:
        return

    # Add ResizeRight to path (it's in PALM/ResizeRight). This is only needed for PALM.
    resize_right_path = os.path.join(project_root, "PALM", "ResizeRight")
    if os.path.exists(resize_right_path) and resize_right_path not in sys.path:
        sys.path.insert(0, resize_right_path)

    try:
        # Import from matlab_tools
        from PALM.matlab_tools import load_dncnn, fspecial, DnCNN

        # Import from utils_palm
        from PALM.utils_palm import (
            estimate_c,
            Link,
            d1, d2, dtd,
            HXconv, BlockMM,
            FSR_xirm_NL,
            Descente_grad_xus_NL,
            FusionPALM,
        )
    except ImportError as e:
        print(f"Error importing PALM modules (required for --palm/--all): {e}")
        print(f"Python path: {sys.path}")

        palm_dir = os.path.join(project_root, "PALM")
        if os.path.exists(palm_dir):
            print("\nContents of PALM directory:")
            for f in os.listdir(palm_dir):
                if f.endswith(".py") or os.path.isdir(os.path.join(palm_dir, f)):
                    print(f"  - {f}")
        raise

    _PALM_IMPORTED = True

# ============================================================================
# Configuration
# ============================================================================

ROOT_DIR = project_root
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")

# Input images
MRI_SOURCE = os.path.join(DATA_DIR, "irm.png")
US_SOURCE = os.path.join(DATA_DIR, "us.png")

# PALM parameters (from palm_main.py)
PALM_PARAMS = {
    'tau1': 1e-12,
    'tau2': 1e-15,
    'tau3': 2e-4,
    'tau4': 1e-4,
    'd': 6
}

# ============================================================================
# Setup Functions
# ============================================================================

def setup_directories():
    dirs = [
        EXPERIMENTS_DIR,
        os.path.join(EXPERIMENTS_DIR, "palm"),
        os.path.join(EXPERIMENTS_DIR, "ddfm"),
        os.path.join(EXPERIMENTS_DIR, "hybrids"),
        os.path.join(EXPERIMENTS_DIR, "summary"),
        os.path.join(EXPERIMENTS_DIR, "temp"),
        "DDFM/output/recon",
        "DDFM/output/progress",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Directories created in {EXPERIMENTS_DIR}")

def check_source_images():
    if not os.path.exists(MRI_SOURCE):
        print(f"ERROR: MRI image not found: {MRI_SOURCE}")
        print(f"   Files in data/: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'data/ not found'}")
        return False
    if not os.path.exists(US_SOURCE):
        print(f"ERROR: US image not found: {US_SOURCE}")
        return False
    print(f"Source images found: {MRI_SOURCE}, {US_SOURCE}")
    return True

# ============================================================================
# PALM Experiments
# ============================================================================

def run_palm(iterations=5, mode='full'):
    _import_palm()
    print(f"\nRunning PALM ({mode}) for {iterations} iterations...")
    
    # Load images
    mri = imageio.imread(MRI_SOURCE).astype(np.float64)
    us = imageio.imread(US_SOURCE).astype(np.float64)
    
    # Normalize to [0, 1]
    mri = mri / 255.0
    us = us / 255.0
    
    print(f"  MRI shape: {mri.shape}, range: [{mri.min():.2f}, {mri.max():.2f}]")
    print(f"  US shape: {us.shape}, range: [{us.min():.2f}, {us.max():.2f}]")
    
    # Estimate polynomial coefficients
    cest, _ = estimate_c(mri, us, d=6)
    c = np.abs(cest)
    print(f"  Polynomial coefficients estimated")
    
    # Prepare inputs for PALM
    ym = mri / mri.max()
    yu = us / us.max()
    xu0 = load_dncnn(yu)  # DnCNN denoising
    print(f"  DnCNN denoising complete")
    
    if mode == 'full':
        # Use FusionPALM from utils_palm
        result = FusionPALM(
            ym,
            xu0,
            c,
            PALM_PARAMS['tau1'],
            PALM_PARAMS['tau2'],
            PALM_PARAMS['tau3'],
            PALM_PARAMS['tau4'],
            PALM_PARAMS['d'],
            iterations
        )
    else:
        print(f"  WARNING: Mode '{mode}' not fully implemented yet")
        print(f"  Using full PALM as fallback")
        result = FusionPALM(
            ym,
            xu0,
            c,
            PALM_PARAMS['tau1'],
            PALM_PARAMS['tau2'],
            PALM_PARAMS['tau3'],
            PALM_PARAMS['tau4'],
            PALM_PARAMS['d'],
            iterations
        )
    
    # Normalize and save
    result = (result - result.min()) / (result.max() - result.min())
    result = (result * 255).astype(np.uint8)
    
    # Save
    mode_suffix = "" if mode == 'full' else f"_{mode}"
    filename = f"palm{iterations}{mode_suffix}.png"
    output_path = os.path.join(EXPERIMENTS_DIR, "palm", filename)
    imageio.imwrite(output_path, result)
    print(f"  Saved to: {output_path}")
    
    return output_path

# ============================================================================
# DDFM Experiments
# ============================================================================

def prepare_for_ddfm(image_path, output_path, target_size=(256, 256)):
    """Resize image for DDFM input"""
    img = imageio.imread(image_path)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    img_resized = resize(img, target_size, preserve_range=True, anti_aliasing=True)
    img_resized = img_resized.astype(np.uint8)
    imageio.imwrite(output_path, img_resized)
    return output_path

def run_ddfm_baseline():
    """Run DDFM baseline (using random initialization)"""
    print("\nRunning DDFM baseline (100 steps)...")
    
    temp_dir = os.path.join(EXPERIMENTS_DIR, "temp")
    
    # Create a neutral gray image as initialization
    gray_init = os.path.join(temp_dir, "gray_init.png")
    gray_img = np.ones((256, 256), dtype=np.uint8) * 128
    imageio.imwrite(gray_init, gray_img)
    
    cmd = [
        "python", "DDFM/run_single_experiment.py",
        "--initial", gray_init,
        "--name", "ddfm_baseline",
        "--steps", "100"
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(f"    {line.strip()}")
    
    # Wait for process to complete
    return_code = process.wait()
    
    if return_code != 0:
        print(f"DDFM failed with return code {return_code}")
        return None
    
    # Check for output
    ddfm_output = "DDFM/output/recon/ddfm_baseline.png"
    max_wait = 300  # 5 minutes timeout
    waited = 0
    
    while not os.path.exists(ddfm_output) and waited < max_wait:
        time.sleep(5)
        waited += 5
        print(f"  Waiting for output... ({waited}s)")
    
    if os.path.exists(ddfm_output):
        final_output = os.path.join(EXPERIMENTS_DIR, "ddfm", "ddfm_baseline.png")
        img = imageio.imread(ddfm_output)
        if img.shape != (600, 600):
            img = resize(img, (600, 600), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        imageio.imwrite(final_output, img)
        print(f"  Saved to: {final_output}")
        return final_output
    else:
        print(f"Timeout waiting for output")
        return None

def run_hybrid(initial_image, exp_name, steps):
    """Run hybrid experiment (PALM + DDFM)"""
    print(f"\nRunning hybrid: {exp_name} (steps={steps})")
    
    temp_dir = os.path.join(EXPERIMENTS_DIR, "temp")
    initial_256 = os.path.join(temp_dir, f"{exp_name}_256.png")
    prepare_for_ddfm(initial_image, initial_256)
    
    cmd = [
        "python", "DDFM/run_single_experiment.py",
        "--initial", initial_256,
        "--name", exp_name,
        "--steps", str(steps)
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    ddfm_output = f"DDFM/output/recon/{exp_name}.png"
    time.sleep(2)
    
    if os.path.exists(ddfm_output):
        final_output = os.path.join(EXPERIMENTS_DIR, "hybrids", f"{exp_name}.png")
        img = imageio.imread(ddfm_output)
        if img.shape != (600, 600):
            img = resize(img, (600, 600), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        imageio.imwrite(final_output, img)
        print(f"  Saved to: {final_output}")
        return final_output
    else:
        print(f"ERROR: Output not found: {ddfm_output}")
        return None

# ============================================================================
# Evaluation
# ============================================================================

def normalize(img):
    img = img.astype(np.float64)
    img = img - img.min()
    img = img / img.max()
    return img

def evaluate(ref, img):
    psnr = peak_signal_noise_ratio(ref, img, data_range=1)
    ssim = structural_similarity(ref, img, data_range=1)
    return psnr, ssim

def run_evaluation():
    """Evaluate all results and create summary CSV"""
    print("\nRunning evaluation...")
    
    mri = normalize(imageio.imread(MRI_SOURCE))
    us = normalize(imageio.imread(US_SOURCE))
    
    target_shape = (600, 600)
    
    if mri.shape != target_shape:
        mri = resize(mri, target_shape, anti_aliasing=True)
    if us.shape != target_shape:
        us = resize(us, target_shape, anti_aliasing=True)
    
    results = []
    
    # PALM baselines
    palm_files = {
        "PALM1": os.path.join(EXPERIMENTS_DIR, "palm", "palm1.png"),
        "PALM5": os.path.join(EXPERIMENTS_DIR, "palm", "palm5.png"),
        "PALM10": os.path.join(EXPERIMENTS_DIR, "palm", "palm10.png"),
    }
    
    for name, path in palm_files.items():
        if os.path.exists(path):
            img = normalize(imageio.imread(path))
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            psnr_mri, ssim_mri = evaluate(mri, img)
            psnr_us, ssim_us = evaluate(us, img)
            results.append({
                "Method": name, "Type": "PALM",
                "PSNR_MRI": psnr_mri, "SSIM_MRI": ssim_mri,
                "PSNR_US": psnr_us, "SSIM_US": ssim_us
            })
    
    # DDFM baseline
    ddfm_path = os.path.join(EXPERIMENTS_DIR, "ddfm", "ddfm_baseline.png")
    if os.path.exists(ddfm_path):
        img = normalize(imageio.imread(ddfm_path))
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        psnr_mri, ssim_mri = evaluate(mri, img)
        psnr_us, ssim_us = evaluate(us, img)
        results.append({
            "Method": "DDFM", "Type": "DDFM",
            "PSNR_MRI": psnr_mri, "SSIM_MRI": ssim_mri,
            "PSNR_US": psnr_us, "SSIM_US": ssim_us
        })
    
    # Hybrids
    hybrid_names = [
        "P1_D10", "P1_D25", "P1_D50", "P1_D75", "P1_D100",
        "P5_D10", "P5_D25", "P5_D50", "P5_D75", "P5_D100"
    ]
    
    for name in hybrid_names:
        path = os.path.join(EXPERIMENTS_DIR, "hybrids", f"{name}.png")
        if os.path.exists(path):
            img = normalize(imageio.imread(path))
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            psnr_mri, ssim_mri = evaluate(mri, img)
            psnr_us, ssim_us = evaluate(us, img)
            results.append({
                "Method": name, "Type": "Hybrid",
                "PSNR_MRI": psnr_mri, "SSIM_MRI": ssim_mri,
                "PSNR_US": psnr_us, "SSIM_US": ssim_us
            })
    
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(EXPERIMENTS_DIR, "summary", "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n  Results saved to: {csv_path}")
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
    
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MRI-US Fusion Experiments")
    
    parser.add_argument('--palm', action='store_true', help='Run PALM baselines (1,5,10)')
    parser.add_argument('--ddfm', action='store_true', help='Run DDFM baseline')
    parser.add_argument('--hybrids', action='store_true', help='Run hybrid experiments')
    parser.add_argument('--option1', action='store_true', help='Run Option 1 (US-only PALM)')
    parser.add_argument('--option2', action='store_true', help='Run Option 2 (MRI-only PALM)')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--all', action='store_true', help='Run everything')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print("MRI-US FUSION EXPERIMENTS")
    print("="*70)
    print(f"Project root: {ROOT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Experiments directory: {EXPERIMENTS_DIR}")
    
    setup_directories()
    
    if not check_source_images():
        return
    
    if args.all or args.palm:
        print("\n" + "="*70)
        print("Running PALM Baselines")
        print("="*70)
        run_palm(1)
        run_palm(5)
        run_palm(10)
    
    if args.all or args.option1:
        print("\n" + "="*70)
        print("Running Option 1: US-only PALM")
        print("="*70)
        print("  WARNING: Option 1 not fully implemented yet")
    
    if args.all or args.option2:
        print("\n" + "="*70)
        print("Running Option 2: MRI-only PALM")
        print("="*70)
        print("  WARNING: Option 2 not fully implemented yet")
    
    if args.all or args.ddfm:
        print("\n" + "="*70)
        print("Running DDFM Baseline")
        print("="*70)
        run_ddfm_baseline()
    
    if args.all or args.hybrids:
        print("\n" + "="*70)
        print("Running Hybrid Experiments")
        print("="*70)
        
        palm1 = os.path.join(EXPERIMENTS_DIR, "palm", "palm1.png")
        palm5 = os.path.join(EXPERIMENTS_DIR, "palm", "palm5.png")
        
        if not os.path.exists(palm1) or not os.path.exists(palm5):
            print("WARNING: PALM results missing. Run --palm first")
        else:
            experiments = [
                (palm1, "P1_D10", 10), (palm1, "P1_D25", 25),
                (palm1, "P1_D50", 50), (palm1, "P1_D75", 75), (palm1, "P1_D100", 100),
                (palm5, "P5_D10", 10), (palm5, "P5_D25", 25),
                (palm5, "P5_D50", 50), (palm5, "P5_D75", 75), (palm5, "P5_D100", 100),
            ]
            
            for init, name, steps in experiments:
                run_hybrid(init, name, steps)
                time.sleep(2)
    
    if args.all or args.evaluate:
        print("\n" + "="*70)
        print("Running Evaluation")
        print("="*70)
        run_evaluation()
    
    print("\nAll requested experiments completed!")

if __name__ == "__main__":
    main()