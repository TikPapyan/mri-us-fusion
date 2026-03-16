# MRI-US Fusion Project: Complete Context for Cursor

## Project Overview
This project focuses on fusing Magnetic Resonance (MR) and Ultrasound (US) images for endometriosis detection. Endometriosis is a gynecological condition where endometrial tissue grows outside the uterus, affecting fertility and quality of life. Accurate preoperative mapping is crucial for surgical success.

## Why Fusion?
MRI: Good contrast, wide field of view, but low spatial resolution (~1mm)

US: High spatial resolution, but speckle noise, limited field of view

Fusion goal: Combine MRI's anatomical context with US's fine details in a single image

## Literature & References

### Core Papers
[3] El Mansouri et al. (2020) - PALM-based Fusion

Title: "Fusion of Magnetic Resonance and Ultrasound Images for Endometriosis Detection"

Journal: IEEE Transactions on Image Processing

Key idea: Inverse problem formulation with polynomial intensity mapping

Method: Proximal Alternating Linearized Minimization (PALM)

Strengths: Handles different resolutions, noise models, interpretable

Code: Provided by supervisor (Tom Longin's implementation)

[2] DDFM (Zhao et al., 2023) - ICCV Oral

Title: "DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion"

Key idea: Diffusion model + EM algorithm for fusion

Code: https://github.com/Zhaozixiang1228/MMIF-DDFM

Key innovation: Splits conditional generation into unconditional diffusion + likelihood rectification via EM

## Project Objectives

### Main Goal
Develop and evaluate fusion methods combining MRI and US images, comparing:

1. PALM (traditional optimization-based)
2. DDFM (diffusion-based generative)
3. Hybrid approaches (PALM + DDFM with varying steps)

### Professor's Specific Instructions
1. Establish baselines: PALM1, PALM5, DDFM
2. Test hybrids: PALM1/5 + DDFM with steps (10,25,50,75,100)
3. New experiments (from latest email):
- Option 1: Only US estimation from PALM + DDFM
- Option 2: Only MRI estimation from PALM + DDFM
4. Create concise PDF with visual results
5. Make code Google Colab compatible

## Project Structure
```
mri-us-fusion/
├── PALM/                          # Original PALM implementation
│   ├── palm_main.py                # Main PALM algorithm
│   ├── utils_palm.py                # Core PALM functions
│   │   ├── estimate_c()             # Polynomial coefficient estimation
│   │   ├── FusionPALM()             # Main fusion loop
│   │   ├── FSR_xirm_NL()             # MRI super-resolution update
│   │   └── Descente_grad_xus_NL()    # US denoising update
│   ├── matlab_tools.py              # DnCNN denoising, MATLAB-like utils
│   ├── ResizeRight/                  # High-quality resizing library
│   ├── images/                       # Source images (Data1/)
│   └── results/                      # Empty (results stored elsewhere)
│
├── DDFM/                           # DDFM implementation (cloned + modified)
│   ├── guided_diffusion/            # Core diffusion code
│   │   ├── gaussian_diffusion.py     # Diffusion sampler
│   │   ├── EM_onestep.py              # EM algorithm for likelihood rectification
│   │   └── unet.py                    # UNet architecture
│   ├── configs/                      # YAML configs
│   ├── util/                         # Utilities
│   ├── input/                        # Source images (256x256)
│   │   ├── ir/irm.png                 # MRI
│   │   └── vi/us.png                   # US
│   ├── models/                        # Pre-trained diffusion model
│   │   └── 256x256_diffusion_uncond.pt
│   ├── output/                        # DDFM intermediate outputs
│   │   └── recon/                      # Hybrid results
│   │       ├── ddfm_baseline.png
│   │       ├── irm_us_fused.png
│   │       └── [P1/P5]_D[10-100].png  # All hybrid variants
│   ├── preprocess_for_ddfm.py
│   ├── run_single_experiment.py        # Single experiment runner
│   ├── sample.py                       # Original DDFM sampler
│   └── ...
│
├── scripts/                         # Experiment automation
│   ├── run_all_manual.py              # Master script for all experiments
│   ├── prepare_palm_for_ddfm.py        # Resize PALM outputs for DDFM
│   ├── experiment_runner.py             # Basic evaluation
│   └── evaluate_all_experiments.py      # Comprehensive CSV evaluation
│
├── results/                          # FINAL ORGANIZED RESULTS
│   ├── baselines/                     # Baseline fusion results
│   │   ├── baseline_palm1.png
│   │   ├── baseline_palm5.png
│   │   ├── palm_10.png
│   │   └── ddfm_baseline.png
│   ├── hybrids/                        # All 10 hybrid results
│   │   ├── P1_D10.png
│   │   ├── P1_D25.png
│   │   ├── P1_D50.png
│   │   ├── P1_D75.png
│   │   ├── P1_D100.png
│   │   ├── P5_D10.png
│   │   ├── P5_D25.png
│   │   ├── P5_D50.png
│   │   ├── P5_D75.png
│   │   └── P5_D100.png
│   └── summary/                        # Quantitative results
│       └── all_results.csv
│
├── README.md                         # Project documentation
├── requirements.txt                   # Python dependencies
├── Overview.md                        # This context document
└── [.gitignore]                       # (should be added)
```

## Key Algorithms Explained

### 1. PALM (Proximal Alternating Linearized Minimization)
Purpose: Solve the fusion inverse problem by alternating between MRI super-resolution and US denoising.

Mathematical Model:

```
y_m = SCx + n_m      (MRI: blur + downsample + Gaussian noise)
y_u = f(x, ∇x) + n_u (US: polynomial mapping + log-Rayleigh noise)
```

Key Functions:

- estimate_c(): Finds polynomial coefficients linking MRI and US intensities
- FSR_xirm_NL(): MRI update – super-resolution with TV regularization
- Descente_grad_xus_NL(): US update – gradient descent with log-Rayleigh noise model
- Link(): Polynomial mapping: x_u = Σ c_pq * x_m^p * |∇x_m|^q

Convergence: PALM5 ≈ PALM10 (algorithm stabilizes by iteration 5)

### 2. DDFM (Denoising Diffusion Model for Fusion)
Purpose: Use diffusion model's generative prior while enforcing fidelity to source images.

Core Idea: Split conditional generation into:

1. Unconditional Diffusion Sampling (UDS): Predicts clean image f̂₀|ₜ from noisy fₜ
2. EM Module: Rectifies prediction using source images via Expectation-Maximization

Algorithm Steps (per timestep t):

text
1. Predict: f̂₀|ₜ = UDS(fₜ)                    # From diffusion model
2. E-step: Update latent variables (m, n) using Eq. 16
3. M-step: Solve for refined f̂₀|ₜ using Eqs. 25,27,29
4. Update: fₜ₋₁ using fₜ and refined f̂₀|ₜ

#### Key Files:

 - gaussian_diffusion.py: p_sample_loop() – main sampling loop
 - EM_onestep.py: EM_onestep() – single EM iteration
 - condition_methods.py: Wraps EM module for conditioning

### 3. Hybrid Approach (PALM → DDFM)
Idea: Use PALM output as initial guess for DDFM, then run limited diffusion steps.

Implementation:

```
# 1. Run PALM for k iterations
palm_output = FusionPALM(ym, yu, ..., num_iterations=k)

# 2. Resize to 256×256 (DDFM input size)
palm_256 = resize(palm_output, (256,256))

# 3. Feed into DDFM with specified steps
ddfm_output = run_ddfm(
    initial_image=palm_256,
    mri=mri_256,
    us=us_256,
    num_steps=s
)

# 4. Resize back to original size (600×600) for evaluation
final = resize(ddfm_output, (600,600))
```

## Current Results (as of March 2026)

### Baselines
Method	vs MRI (PSNR/SSIM)	vs US (PSNR/SSIM)
PALM1	19.79 dB / 0.521	8.00 dB / 0.110
PALM5	18.89 dB / 0.504	8.18 dB / 0.116
DDFM	21.41 dB / 0.502	8.91 dB / 0.190

### Hybrid Results – Key Findings

Config	MRI PSNR	MRI SSIM	US PSNR	US SSIM	Observation
P5_D10	27.25 dB	0.879	7.58	0.102	Best MRI preservation (+8 dB over baselines)
P1_D50	21.75 dB	0.519	8.81	0.176	Balanced
DDFM	21.41 dB	0.502	8.91 dB	0.190	Best US preservation

### Key Observations

1. Trade-off: Low DDFM steps (10-25) → excellent MRI, poor US; High steps (50-100) → good US, lower MRI
2. Sweet spot: 10-step hybrids give dramatic MRI improvement (+8 dB PSNR)
3. Starting point: PALM1 vs PALM5 makes negligible difference
4. Best overall: P5_D10 for MRI quality, pure DDFM for US quality