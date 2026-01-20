import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original, fused, reference=None):
    """
    Calculate image quality metrics
    
    Parameters:
    -----------
    original : numpy array
        Original image (MRI or US)
    fused : numpy array  
        Fused image
    reference : numpy array, optional
        Ground truth reference (if available)
    
    Returns:
    --------
    metrics : dict
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # Ensure same size
    if original.shape != fused.shape:
        fused_resized = cv2.resize(fused, original.shape[::-1])
    else:
        fused_resized = fused
    
    # Normalize to [0, 1] if needed
    if np.max(original) > 1.0:
        original_norm = (original - np.min(original)) / (np.max(original) - np.min(original))
    else:
        original_norm = original.copy()
    
    if np.max(fused_resized) > 1.0:
        fused_norm = (fused_resized - np.min(fused_resized)) / (np.max(fused_resized) - np.min(fused_resized))
    else:
        fused_norm = fused_resized.copy()
    
    # 1. Mean Squared Error (MSE)
    mse = np.mean((original_norm - fused_norm) ** 2)
    metrics['MSE'] = mse
    
    # 2. Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        metrics['PSNR'] = float('inf')
    else:
        metrics['PSNR'] = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # 3. Structural Similarity Index (SSIM)
    try:
        # Ensure images are in [0, 1] range for SSIM
        data_range = 1.0
        ssim_value = ssim(original_norm, fused_norm, 
                         data_range=data_range,
                         channel_axis=None if len(original_norm.shape) == 2 else -1)
        metrics['SSIM'] = ssim_value
    except:
        metrics['SSIM'] = np.nan
    
    # 4. Mean Absolute Error (MAE)
    metrics['MAE'] = np.mean(np.abs(original_norm - fused_norm))
    
    # 5. Correlation Coefficient
    flat_orig = original_norm.flatten()
    flat_fused = fused_norm.flatten()
    correlation = np.corrcoef(flat_orig, flat_fused)[0, 1]
    metrics['Correlation'] = correlation
    
    # 6. Entropy (information content)
    def calculate_entropy(image):
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
        hist = hist[hist > 0] / len(image.flatten())
        return -np.sum(hist * np.log2(hist))
    
    metrics['Entropy_Original'] = calculate_entropy(original_norm)
    metrics['Entropy_Fused'] = calculate_entropy(fused_norm)
    
    # 7. Contrast (standard deviation)
    metrics['Contrast_Original'] = np.std(original_norm)
    metrics['Contrast_Fused'] = np.std(fused_norm)
    
    return metrics

def print_metrics_table(metrics_dict, title="Image Quality Metrics"):
    """
    Print metrics in a formatted table
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            if key in ['PSNR']:
                print(f"{key:20} : {value:8.2f} dB")
            elif key in ['SSIM', 'Correlation']:
                print(f"{key:20} : {value:8.4f}")
            elif key in ['MSE', 'MAE']:
                print(f"{key:20} : {value:8.6f}")
            elif 'Entropy' in key or 'Contrast' in key:
                print(f"{key:20} : {value:8.4f}")
            else:
                print(f"{key:20} : {value:8.4f}")
        else:
            print(f"{key:20} : {value}")
    
    print("=" * 60)

# Test the metrics
if __name__ == '__main__':
    # Create test images
    test_img1 = np.random.rand(100, 100)
    test_img2 = test_img1 + 0.1 * np.random.randn(100, 100)
    
    metrics = calculate_metrics(test_img1, test_img2)
    print_metrics_table(metrics, "Test Metrics")