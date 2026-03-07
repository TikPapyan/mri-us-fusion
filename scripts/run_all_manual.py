# run_all_manual_fixed.py
import os
import subprocess
import time
import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize

def run_ddfm_experiment(initial_image, exp_name, num_steps):
    """Run a single DDFM experiment"""
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name} (steps={num_steps})")
    print(f"{'='*60}")
    
    # Run DDFM with the fixed script
    cmd = [
        "python", "DDFM/run_single_experiment.py",
        "--initial", initial_image,
        "--name", exp_name,
        "--steps", str(num_steps)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        print(f"Return code: {result.returncode}")
        return None
    
    # DDFM output path
    ddfm_output = f"DDFM/output/recon/{exp_name}.png"
    
    # Wait a moment for file to be written
    time.sleep(1)
    
    # Final output path in experiments
    final_output = f"experiments/{exp_name}.png"
    
    # Copy and resize to 600x600
    if os.path.exists(ddfm_output):
        print(f"Found output: {ddfm_output}")
        img = imageio.imread(ddfm_output)
        print(f"Output shape: {img.shape}")
        
        if img.shape != (600, 600):
            img = resize(img, (600, 600), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        
        imageio.imwrite(final_output, img)
        print(f"✅ Saved to: {final_output}")
        return final_output
    else:
        print(f"❌ Output not found: {ddfm_output}")
        print(f"Contents of DDFM/output/recon/: {os.listdir('DDFM/output/recon/') if os.path.exists('DDFM/output/recon/') else 'directory not found'}")
        return None

def main():
    # Create directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("DDFM/output/recon", exist_ok=True)
    
    # First, prepare PALM inputs
    print("Step 1: Preparing PALM inputs...")
    subprocess.run(["python", "prepare_palm_for_ddfm.py"])
    
    # Define experiments
    experiments = [
        # PALM1 variants
        ("experiments/inputs/palm1_256.png", "P1_D10", 10),
        ("experiments/inputs/palm1_256.png", "P1_D25", 25),
        ("experiments/inputs/palm1_256.png", "P1_D50", 50),
        ("experiments/inputs/palm1_256.png", "P1_D75", 75),
        ("experiments/inputs/palm1_256.png", "P1_D100", 100),
        
        # PALM5 variants
        ("experiments/inputs/palm5_256.png", "P5_D10", 10),
        ("experiments/inputs/palm5_256.png", "P5_D25", 25),
        ("experiments/inputs/palm5_256.png", "P5_D50", 50),
        ("experiments/inputs/palm5_256.png", "P5_D75", 75),
        ("experiments/inputs/palm5_256.png", "P5_D100", 100),
    ]
    
    # Run experiments
    results = []
    successful = 0
    failed = 0
    
    for i, (initial_image, exp_name, steps) in enumerate(experiments):
        print(f"\n--- Progress: {i+1}/{len(experiments)} ---")
        output = run_ddfm_experiment(initial_image, exp_name, steps)
        if output:
            results.append(output)
            successful += 1
        else:
            failed += 1
        time.sleep(2)  # Small pause between experiments
    
    print(f"\n🎉 Experiments completed! Successful: {successful}, Failed: {failed}")
    
    if successful > 0:
        print(f"Results saved in: experiments/")
        
        # Now run evaluation
        print("\n📊 Running evaluation...")
        if os.path.exists("evaluate_all_experiments.py"):
            subprocess.run(["python", "evaluate_all_experiments.py"])
        else:
            print("Warning: evaluate_all_experiments.py not found")

if __name__ == "__main__":
    main()