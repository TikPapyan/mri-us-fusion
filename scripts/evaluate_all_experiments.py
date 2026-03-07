# evaluate_all_experiments.py
import numpy as np
import imageio.v2 as imageio
import pandas as pd
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

def normalize(img):
    img = img.astype(np.float64)
    img = img - img.min()
    img = img / img.max()
    return img

def evaluate(ref, img):
    psnr = peak_signal_noise_ratio(ref, img, data_range=1)
    ssim = structural_similarity(ref, img, data_range=1)
    return psnr, ssim

def main():
    # Paths
    mri_path = "PALM/results/mri.png"
    us_path = "PALM/results/us.png"
    hybrid_dir = "experiments"
    palm_dir = "PALM/results"
    
    # Load reference images
    mri = normalize(imageio.imread(mri_path))
    us = normalize(imageio.imread(us_path))
    
    # Get target shape from PALM1
    palm1 = normalize(imageio.imread(os.path.join(palm_dir, "baseline_palm1.png")))
    target_shape = palm1.shape
    
    # Resize references
    if mri.shape != target_shape:
        mri = resize(mri, target_shape, anti_aliasing=True)
    if us.shape != target_shape:
        us = resize(us, target_shape, anti_aliasing=True)
    
    # Baselines
    baselines = {
        "PALM1": os.path.join(palm_dir, "baseline_palm1.png"),
        "PALM5": os.path.join(palm_dir, "baseline_palm5.png"),
        "PALM10": os.path.join(palm_dir, "palm_10.png"),
        "DDFM": os.path.join(palm_dir, "ddfm_baseline.png")
    }
    
    # Hybrid experiments
    exp_names = [
        "P1_D10", "P1_D25", "P1_D50", "P1_D75", "P1_D100",
        "P5_D10", "P5_D25", "P5_D50", "P5_D75", "P5_D100"
    ]
    
    # Collect all results
    results = []
    
    # Evaluate baselines
    for name, path in baselines.items():
        if os.path.exists(path):
            img = normalize(imageio.imread(path))
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            if img.shape != target_shape:
                img = resize(img, target_shape, anti_aliasing=True)
            
            psnr_mri, ssim_mri = evaluate(mri, img)
            psnr_us, ssim_us = evaluate(us, img)
            
            results.append({
                "Method": name,
                "Type": "Baseline",
                "PALM_iters": 0 if name == "DDFM" else (1 if name == "PALM1" else 5),
                "DDFM_steps": 100 if name == "DDFM" else 0,
                "PSNR_MRI": psnr_mri,
                "SSIM_MRI": ssim_mri,
                "PSNR_US": psnr_us,
                "SSIM_US": ssim_us
            })
    
    # Evaluate hybrid experiments
    for exp_name in exp_names:
        path = os.path.join(hybrid_dir, f"{exp_name}.png")
        if os.path.exists(path):
            img = normalize(imageio.imread(path))
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            if img.shape != target_shape:
                img = resize(img, target_shape, anti_aliasing=True)
            
            psnr_mri, ssim_mri = evaluate(mri, img)
            psnr_us, ssim_us = evaluate(us, img)
            
            # Parse experiment name
            palm_part, ddfm_part = exp_name.split('_')
            palm_iters = 1 if palm_part == "P1" else 5
            ddfm_steps = int(ddfm_part[1:])
            
            results.append({
                "Method": exp_name,
                "Type": "Hybrid",
                "PALM_iters": palm_iters,
                "DDFM_steps": ddfm_steps,
                "PSNR_MRI": psnr_mri,
                "SSIM_MRI": ssim_mri,
                "PSNR_US": psnr_us,
                "SSIM_US": ssim_us
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by PALM_iters and DDFM_steps
    df = df.sort_values(["PALM_iters", "DDFM_steps"])
    
    # Save to CSV
    df.to_csv(os.path.join(hybrid_dir, "summary", "all_results.csv"), index=False)
    
    # Print summary tables
    print("\n" + "="*80)
    print("COMPLETE RESULTS SUMMARY")
    print("="*80)
    
    # Baselines
    print("\n📊 BASELINES:")
    print("-"*60)
    baseline_df = df[df["Type"] == "Baseline"]
    for _, row in baseline_df.iterrows():
        print(f"{row['Method']:<10} | MRI: {row['PSNR_MRI']:>6.2f} dB / {row['SSIM_MRI']:.4f} | US: {row['PSNR_US']:>6.2f} dB / {row['SSIM_US']:.4f}")
    
    # PALM1 variants
    print("\n📊 PALM1 + DDFM VARIANTS:")
    print("-"*60)
    print(f"{'Method':<8} | {'Steps':<5} | {'PSNR_MRI':<10} | {'SSIM_MRI':<9} | {'PSNR_US':<10} | {'SSIM_US':<9}")
    print("-"*60)
    p1_df = df[(df["PALM_iters"] == 1) & (df["Type"] == "Hybrid")]
    for _, row in p1_df.iterrows():
        print(f"{row['Method']:<8} | {row['DDFM_steps']:<5} | {row['PSNR_MRI']:<10.2f} | {row['SSIM_MRI']:<9.4f} | {row['PSNR_US']:<10.2f} | {row['SSIM_US']:<9.4f}")
    
    # PALM5 variants
    print("\n📊 PALM5 + DDFM VARIANTS:")
    print("-"*60)
    print(f"{'Method':<8} | {'Steps':<5} | {'PSNR_MRI':<10} | {'SSIM_MRI':<9} | {'PSNR_US':<10} | {'SSIM_US':<9}")
    print("-"*60)
    p5_df = df[(df["PALM_iters"] == 5) & (df["Type"] == "Hybrid")]
    for _, row in p5_df.iterrows():
        print(f"{row['Method']:<8} | {row['DDFM_steps']:<5} | {row['PSNR_MRI']:<10.2f} | {row['SSIM_MRI']:<9.4f} | {row['PSNR_US']:<10.2f} | {row['SSIM_US']:<9.4f}")
    
    # Find best in each category
    print("\n🏆 BEST RESULTS:")
    print("-"*60)
    
    best_mri_psnr = df.loc[df['PSNR_MRI'].idxmax()]
    print(f"Best MRI PSNR: {best_mri_psnr['Method']} - {best_mri_psnr['PSNR_MRI']:.2f} dB")
    
    best_mri_ssim = df.loc[df['SSIM_MRI'].idxmax()]
    print(f"Best MRI SSIM: {best_mri_ssim['Method']} - {best_mri_ssim['SSIM_MRI']:.4f}")
    
    best_us_psnr = df.loc[df['PSNR_US'].idxmax()]
    print(f"Best US PSNR: {best_us_psnr['Method']} - {best_us_psnr['PSNR_US']:.2f} dB")
    
    best_us_ssim = df.loc[df['SSIM_US'].idxmax()]
    print(f"Best US SSIM: {best_us_ssim['Method']} - {best_us_ssim['SSIM_US']:.4f}")
    
    print(f"\n📁 Full results saved to: {os.path.join(hybrid_dir, 'summary', 'all_results.csv')}")

if __name__ == "__main__":
    main()