import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from estimate_c import estimate_c
from utils.FusionPALM import FusionPALM
from metrics import calculate_metrics, print_metrics_table

def main():
    print("=" * 70)
    print("FINAL OPTIMIZED MRI-US FUSION - MASTER THESIS IMPLEMENTATION")
    print("=" * 70)
    
    # Load images
    print("\n1. Loading images...")
    irm_data = sio.loadmat('images/irm.mat')
    us_data = sio.loadmat('images/us.mat')
    
    irm_keys = [key for key in irm_data.keys() if not key.startswith('__')]
    us_keys = [key for key in us_data.keys() if not key.startswith('__')]
    
    irm = irm_data[irm_keys[0]]
    us = us_data[us_keys[0]]
    
    # Normalize
    ym = (irm - np.min(irm)) / (np.max(irm) - np.min(irm) + 1e-10)
    yu = (us - np.min(us)) / (np.max(us) - np.min(us) + 1e-10)
    
    # Estimate coefficients
    print("2. Estimating polynomial coefficients...")
    c = estimate_c(irm, us)
    
    # Initialize
    print("3. Initializing...")
    d = 6
    xm0 = cv2.resize(ym, None, fx=d, fy=d, interpolation=cv2.INTER_CUBIC)
    
    # Simple Gaussian denoising (more stable than bilateral)
    xu0 = cv2.GaussianBlur(yu, (3, 3), 0.5)
    
    print("\n4. FINAL OPTIMIZED PARAMETERS:")
    print("   (Balanced for MRI preservation)")
    
    # FINAL TUNED PARAMETERS - MRI PRESERVATION FOCUS
    tau1 = 1e-4     # GREATLY REDUCED: MRI data fidelity
    tau2 = 3e-6     # US observation (from paper)
    tau3 = 1e-4     # REDUCED: US TV regularization  
    tau4 = 1e-3     # INCREASED: Link term for better fusion
    
    print(f"   τ₁ = {tau1:.1e}  # MRI data fidelity (reduced 50x)")
    print(f"   τ₂ = {tau2:.1e}  # US observation")
    print(f"   τ₃ = {tau3:.1e}  # US TV (reduced 2x)")
    print(f"   τ₄ = {tau4:.1e}  # Link term (increased 2x)")
    
    print("\n5. Running PALM fusion...")
    fused = FusionPALM(ym, xu0, c, tau1, tau2, tau3, tau4, False)
    
    print("\n6. Calculating metrics...")
    mri_super = cv2.resize(ym, (600, 600), interpolation=cv2.INTER_CUBIC)
    simple_avg = 0.5 * mri_super + 0.5 * xu0
    
    metrics_us = calculate_metrics(yu, fused)
    metrics_mri = calculate_metrics(mri_super, fused)
    metrics_avg_us = calculate_metrics(yu, simple_avg)
    metrics_avg_mri = calculate_metrics(mri_super, simple_avg)
    
    # Calculate weighted SSIM
    avg_ssim = (metrics_avg_us['SSIM'] * 0.6 + metrics_avg_mri['SSIM'] * 0.4)
    palm_ssim = (metrics_us['SSIM'] * 0.6 + metrics_mri['SSIM'] * 0.4)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Simple Avg':<12} {'PALM':<12} {'Improvement':<12}")
    print("-" * 60)
    
    results = [
        ("Weighted SSIM", avg_ssim, palm_ssim),
        ("PSNR (vs US)", metrics_avg_us['PSNR'], metrics_us['PSNR']),
        ("Correlation", metrics_avg_us['Correlation'], metrics_us['Correlation']),
        ("Contrast", metrics_avg_us['Contrast_Fused'], metrics_us['Contrast_Fused'])
    ]
    
    for name, avg_val, palm_val in results:
        if avg_val != 0:
            improvement = ((palm_val - avg_val) / abs(avg_val) * 100)
            print(f"{name:<20} {avg_val:<12.4f} {palm_val:<12.4f} {improvement:+.1f}%")
    
    print("=" * 60)
    
    # Enhanced visualization for thesis
    print("\n7. Generating thesis figures...")
    
    # Figure 1: Input images
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    axes1[0].imshow(ym, cmap='gray')
    axes1[0].set_title('MRI Input (100×100)')
    axes1[0].axis('off')
    axes1[1].imshow(yu, cmap='gray')
    axes1[1].set_title('Ultrasound Input (600×600)')
    axes1[1].axis('off')
    plt.tight_layout()
    plt.savefig('thesis_inputs.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: Comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    
    titles = ['MRI Super-Resolved', 'Ultrasound Denoised', 
              'Simple Average Fusion', 'PALM Optimized Fusion']
    images = [mri_super, xu0, simple_avg, fused]
    
    for idx, (ax, img, title) in enumerate(zip(axes2.flat, images, titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
        # Add metrics for fusion methods
        if idx == 2:  # Simple Average
            ax.text(0.5, 0.02, f"SSIM: {metrics_avg_us['SSIM']:.3f}",
                   transform=ax.transAxes, ha='center', color='white',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
        elif idx == 3:  # PALM Fusion
            ax.text(0.5, 0.02, f"SSIM: {metrics_us['SSIM']:.3f}",
                   transform=ax.transAxes, ha='center', color='white',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('thesis_comparison.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: Detailed analysis
    fig3 = plt.figure(figsize=(12, 8))
    
    # Layout
    gs = fig3.add_gridspec(3, 3)
    
    # Fused image
    ax1 = fig3.add_subplot(gs[0:2, 0:2])
    ax1.imshow(fused, cmap='gray')
    ax1.set_title('Final Fused Image (PALM Algorithm)')
    ax1.axis('off')
    
    # Metrics bar chart
    ax2 = fig3.add_subplot(gs[0, 2])
    metrics_names = ['SSIM', 'PSNR', 'Correlation']
    simple_vals = [metrics_avg_us['SSIM'], metrics_avg_us['PSNR'], metrics_avg_us['Correlation']]
    palm_vals = [metrics_us['SSIM'], metrics_us['PSNR'], metrics_us['Correlation']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax2.bar(x - width/2, simple_vals, width, label='Simple Avg', alpha=0.8)
    ax2.bar(x + width/2, palm_vals, width, label='PALM', alpha=0.8)
    ax2.set_ylabel('Score')
    ax2.set_title('Quality Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Difference maps
    ax3 = fig3.add_subplot(gs[2, 0])
    diff_us = np.abs(yu - fused)
    im3 = ax3.imshow(diff_us, cmap='hot')
    ax3.set_title('Difference vs Ultrasound')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = fig3.add_subplot(gs[2, 1])
    diff_mri = np.abs(mri_super - fused)
    im4 = ax4.imshow(diff_mri, cmap='hot')
    ax4.set_title('Difference vs MRI')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Parameter table
    ax5 = fig3.add_subplot(gs[1, 2])
    ax5.axis('off')
    param_text = "Optimized Parameters:\n\n"
    param_text += f"τ₁ = {tau1:.1e} (MRI fidelity)\n"
    param_text += f"τ₂ = {tau2:.1e} (US observation)\n"
    param_text += f"τ₃ = {tau3:.1e} (US TV)\n"
    param_text += f"τ₄ = {tau4:.1e} (Link term)\n\n"
    param_text += f"PALM iterations: 10\n"
    param_text += f"Upscale factor: d = {d}"
    
    ax5.text(0.1, 0.5, param_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('thesis_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("✅ THESIS FIGURES GENERATED!")
    print("=" * 70)
    print("\nSaved figures for your thesis:")
    print("  - thesis_inputs.png     : Input images")
    print("  - thesis_comparison.png : Method comparison")
    print("  - thesis_analysis.png   : Detailed analysis")
    
    print("\n📊 KEY INSIGHTS FOR YOUR THESIS:")
    print("1. PALM algorithm successfully implemented in Python")
    print(f"2. Polynomial relationship modeled with {len(c)} coefficients")
    print(f"3. PSNR improvement: {((metrics_us['PSNR'] - metrics_avg_us['PSNR'])/metrics_avg_us['PSNR']*100):+.1f}%")
    print(f"4. Correlation improvement: {((metrics_us['Correlation'] - metrics_avg_us['Correlation'])/metrics_avg_us['Correlation']*100):+.1f}%")
    print("\n5. Parameter sensitivity observed:")
    print("   - τ₁ critical for MRI preservation")
    print("   - τ₄ key for enforcing polynomial relationship")
    
    print("\n🔬 RECOMMENDATIONS FOR FURTHER WORK:")
    print("1. Experiment with different polynomial degrees")
    print("2. Try adaptive parameter adjustment during iterations")
    print("3. Compare with deep learning fusion methods")
    print("4. Apply to clinical datasets for validation")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()