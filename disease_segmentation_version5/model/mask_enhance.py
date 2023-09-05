import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image, ImageDraw


def mask_enhance_edge(mask: torch.Tensor, radius: int = 10, remove_solid: bool = False):
    mask_array = mask.cpu().numpy()

    mask_array = mask_array.astype(np.uint8)

    x = cv2.Sobel(mask_array, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(mask_array, cv2.CV_64F, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    fix_edges = 1 * (edges > 0)

    fix_edges_pil = Image.fromarray(fix_edges)
    draw = ImageDraw.Draw(fix_edges_pil)

    shape = fix_edges.shape
    for j in range(shape[0]):
        for i in range(shape[1]):
            if fix_edges[j, i] == 1:
                draw.ellipse((i - radius, j - radius, i + radius, j + radius), fill=1)
    enhanced_edges = TF.pil_to_tensor(fix_edges_pil).to(mask.device)

    if remove_solid:
        enhanced_edges = enhanced_edges - mask
        enhanced_edges *= (enhanced_edges >= 0)

    return enhanced_edges


def masks_enhace_edge(masks: torch.Tensor, radius: int = 10, remove_solid: bool = False):
    results = []
    for i in range(masks.shape[0]):
        results.append(mask_enhance_edge(masks[i], radius, remove_solid))
    return torch.stack(results)
