from functools import partial
import os
import argparse
import yaml
import torch
from DDFM.guided_diffusion.unet import create_model
from DDFM.guided_diffusion.gaussian_diffusion import create_sampler
from DDFM.util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings
warnings.filterwarnings('ignore')

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    if img_BGR is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='DDFM/configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str, default='DDFM/configs/diffusion_config.yaml')                     
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)  
    diffusion_config = load_yaml(args.diffusion_config)
   
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
  
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model)
   
    # Working directory
    test_folder = "input"     
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['recon', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Define your image pairs explicitly - THIS IS THE KEY FIX
    # Format: (ir_filename, vi_filename) where ir = MRI, vi = US
    image_pairs = [
        ('irm.png', 'us.png'),  # MRI in ir/, US in vi/
        # Add more pairs here if you have multiple datasets
    ]

    for ir_name, vi_name in image_pairs:
        # Construct full paths
        ir_path = os.path.join(test_folder, "ir", ir_name)
        vi_path = os.path.join(test_folder, "vi", vi_name)
        
        logger.info(f"Processing pair: IR={ir_name}, VI={vi_name}")
        logger.info(f"  IR path: {ir_path}")
        logger.info(f"  VI path: {vi_path}")
        
        # Read images
        inf_img = image_read(ir_path, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        vis_img = image_read(vi_path, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0

        # Normalize to [-1, 1]
        inf_img = inf_img * 2 - 1
        vis_img = vis_img * 2 - 1

        # Crop to make dimensions divisible by scale
        scale = 32
        h, w = inf_img.shape[2:]
        h = h - h % scale
        w = w - w % scale
        logger.info(f"  Original size: ({inf_img.shape[2]}, {inf_img.shape[3]}), Cropped to: ({h}, {w})")

        # Convert to tensors
        inf_img = torch.FloatTensor(inf_img)[:, :, :h, :w].to(device)
        vis_img = torch.FloatTensor(vis_img)[:, :, :h, :w].to(device)
        
        # Verify shapes match
        assert inf_img.shape == vis_img.shape, f"Shape mismatch: {inf_img.shape} vs {vis_img.shape}"

        # Sampling
        seed = 3407
        torch.manual_seed(seed)
        # Create random noise with 3 channels (RGB)
        x_start = torch.randn((1, 3, h, w), device=device)  

        with torch.no_grad():
            sample = sample_fn(
                x_start=x_start, 
                record=True, 
                I=inf_img,  # MRI as infrared modality
                V=vis_img,  # US as visible modality
                save_root=out_path, 
                img_index=f"{ir_name.split('.')[0]}_{vi_name.split('.')[0]}",  # Combined name
                lamb=0.5,
                rho=0.001
            )

        # Post-process and save
        sample = sample.detach().cpu().squeeze().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        
        # Convert to grayscale using YCrCb
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        
        # Normalize to [0, 1] and convert to uint8
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        sample = (sample * 255).astype(np.uint8)
        
        # Save with descriptive name
        output_filename = f"{ir_name.split('.')[0]}_{vi_name.split('.')[0]}_fused.png"
        output_path = os.path.join(out_path, 'recon', output_filename)
        imsave(output_path, sample)
        logger.info(f"Saved fused image to: {output_path}")
        
        # Also save a copy with simple name for compatibility with your experiment_runner
        simple_path = os.path.join(out_path, 'recon', 'ddfm_baseline.png')
        imsave(simple_path, sample)
        logger.info(f"Also saved as: {simple_path}")

    logger.info("All processing complete!")