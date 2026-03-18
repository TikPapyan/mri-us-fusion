import os
import sys
import argparse
import torch
import yaml
import numpy as np
import imageio.v2 as imageio
import cv2
from functools import partial
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.pytorch_colors import rgb_to_ycbcr, ycbcr_to_rgb

def load_yaml(file_path):
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def image_read(path, mode='GRAY'):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if mode == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)

def run_experiment(initial_image_path, output_name, num_steps, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Running DDFM with {num_steps} steps")
    print(f"Initial image: {initial_image_path}")
    print(f"{'='*60}")
    model_config = load_yaml('DDFM/configs/model_config_imagenet.yaml')
    diffusion_config = load_yaml('DDFM/configs/diffusion_config.yaml')
    diffusion_config['timestep_respacing'] = str(num_steps)
    print(f"Step count set to: {num_steps}")
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model)
    target_size = (256, 256)
    print("Loading source images...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mri_path = os.path.join(project_root, "data", "irm.png")
    us_path = os.path.join(project_root, "data", "us.png")
    if not os.path.exists(mri_path):
        mri_path = os.path.join(project_root, "DDFM", "input", "ir", "irm.png")
    if not os.path.exists(us_path):
        us_path = os.path.join(project_root, "DDFM", "input", "vi", "us.png")
    inf_img = image_read(mri_path)
    vis_img = image_read(us_path)
    from skimage.transform import resize
    inf_img = resize(inf_img, target_size, preserve_range=True, anti_aliasing=True)
    vis_img = resize(vis_img, target_size, preserve_range=True, anti_aliasing=True)
    inf_img = inf_img[np.newaxis, np.newaxis, ...] / 255.0
    vis_img = vis_img[np.newaxis, np.newaxis, ...] / 255.0
    print(f"Loading initial image: {initial_image_path}")
    initial = imageio.imread(initial_image_path)
    if len(initial.shape) == 3:
        initial = np.mean(initial, axis=2)
    if initial.shape != target_size:
        initial = resize(initial, target_size, preserve_range=True, anti_aliasing=True)
    inf_img = inf_img * 2 - 1
    vis_img = vis_img * 2 - 1
    initial = initial / 255.0 * 2 - 1
    print(f"inf_img shape: {inf_img.shape}")
    print(f"vis_img shape: {vis_img.shape}")
    print(f"initial shape: {initial.shape}")
    inf_img = torch.FloatTensor(inf_img).to(device)
    vis_img = torch.FloatTensor(vis_img).to(device)
    initial = torch.FloatTensor(initial).unsqueeze(0).unsqueeze(0).to(device)
    initial = initial.repeat(1, 3, 1, 1)
    x_start = torch.randn((1, 3, 256, 256), device=device)
    print("Starting diffusion sampling...")
    with torch.no_grad():
        sample = sample_fn(
            x_start=x_start,
            record=False,
            I=inf_img,
            V=vis_img,
            save_root='output',
            img_index=output_name,
            lamb=0.5,
            rho=0.001
        )
    print("Sampling complete, post-processing...")
    sample = sample.detach().cpu().squeeze().numpy()
    sample = np.transpose(sample, (1, 2, 0))
    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    sample = (sample - sample.min()) / (sample.max() - sample.min())
    sample = (sample * 255).astype(np.uint8)
    output_path = f'DDFM/output/recon/{output_name}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, sample)
    print(f"Saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', type=str, required=True, help='Path to initial image')
    parser.add_argument('--name', type=str, required=True, help='Output name')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    run_experiment(
        initial_image_path=args.initial,
        output_name=args.name,
        num_steps=args.steps,
        device=device
    )