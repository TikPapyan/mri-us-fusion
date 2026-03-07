"""
Ce fichier regroupe des fonctions de Matlab qui ne trouvent pas leur équivalent en Python, mais dont nous avons besoin
pour le bon fonctionnement du programme.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from skimage.util import view_as_blocks

def blockproc(image, block_size, func):
    h, w = image.shape
    bh, bw = block_size
    assert h % bh == 0 and w % bw == 0, "L'image doit être multiple de la taille des blocs"
    # Découpe en blocs
    blocks = view_as_blocks(image, block_shape=block_size)  # shape (H//bh, W//bw, bh, bw)
    # Applique func à chaque bloc si func accepte un tableau 4D
    processed = func(blocks) if callable(func) else blocks
    # Si func ne traite pas directement un tableau 4D, on doit boucler (voir plus bas)
    out_image = processed.transpose(0, 2, 1, 3).reshape(h, w)
    return out_image

def fspecial(filter_type, *args):
    filter_type = filter_type.lower()
    if filter_type == 'average':
        size = args[0] if args else (3, 3)
        if isinstance(size, int):
            size = (size, size)
        h = np.ones(size) / np.prod(size)
        return h
    elif filter_type == 'gaussian':
        size = args[0] if len(args) >= 1 else (3, 3)
        sigma = args[1] if len(args) >= 2 else 0.5
        if isinstance(size, int):
            size = (size, size)
        m, n = [(ss - 1.) / 2. for ss in size]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    elif filter_type == 'laplacian':
        alpha = args[0] if args else 0.2
        alpha = np.clip(alpha, 0, 1)
        h1 = alpha / (alpha + 1)
        h2 = (1 - alpha) / (alpha + 1)
        h = np.array([[h1, h2, h1],
                      [h2, -4/(alpha + 1), h2],
                      [h1, h2, h1]])
        return h
    elif filter_type == 'log':
        size = args[0] if len(args) >= 1 else (5, 5)
        sigma = args[1] if len(args) >= 2 else 0.5
        if isinstance(size, int):
            size = (size, size)
        m, n = [(ss - 1.) / 2. for ss in size]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = ((x*x + y*y - 2*sigma*sigma) / (sigma**4)) * \
            np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h -= h.mean()
        return h
    elif filter_type == 'sobel':
        h = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])
        return h
    elif filter_type == 'prewitt':
        h = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])
        return h
    elif filter_type == 'motion':
        length = args[0] if len(args) >= 1 else 9
        angle = args[1] if len(args) >= 2 else 0
        # Create a motion blur filter
        h = np.zeros((length, length))
        center = length // 2
        theta = np.deg2rad(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        for i in range(length):
            offset = i - center
            x = int(round(center + offset * cos_theta))
            y = int(round(center + offset * sin_theta))
            if 0 <= x < length and 0 <= y < length:
                h[y, x] = 1
        h /= h.sum()
        return h
    elif filter_type == 'disk':
        radius = args[0] if args else 5
        rad = int(np.ceil(radius))
        y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
        mask = x**2 + y**2 <= radius**2
        h = mask.astype(float)
        h /= h.sum()
        return h
    else:
        raise NotImplementedError("Invalid filter. Choose from 'average', 'gaussian', 'laplacian', 'log', 'sobel', 'prewitt' or 'motion'.")
        
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out
    
def load_dncnn(yu):
    # Modèle DnCNN
    model = DnCNN()
    state_dict = torch.load('models/net.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    image_tensor = torch.from_numpy(yu).unsqueeze(0).unsqueeze(0).float()
    # Débruitage US avec DnCNN
    with torch.no_grad():
        output = model(image_tensor)
    return output.squeeze().cpu().numpy().astype(np.float64)