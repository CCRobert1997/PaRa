from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd
import argparse
import lpips
import numpy as np


# Desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # Scale imported image
    transforms.ToTensor()])  # Transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # Fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image - 0.5) * 2
    return image.to(torch.float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LPIPS',
        description='Takes the path to two directories and gives 1-LPIPS score')
    parser.add_argument('--original_path', help='Path to original images directory', type=str, required=True)
    parser.add_argument('--edited_path', help='Path to edited images directory', type=str, required=True)
    
    args = parser.parse_args()

    loss_fn_alex = lpips.LPIPS(net='alex')
    
    original_path = args.original_path
    edited_path = args.edited_path

    valid_extensions = ('.png', '.jpg', '.jpeg')
    original_files = [os.path.join(original_path, name) for name in os.listdir(original_path) if name.lower().endswith(valid_extensions)]
    edited_files = [os.path.join(edited_path, name) for name in os.listdir(edited_path) if name.lower().endswith(valid_extensions)]

    if not original_files:
        raise ValueError("No original images found in the specified directory.")
    if not edited_files:
        raise ValueError("No edited images found in the specified directory.")

    lpips_scores = []
    for original_file in original_files:
        for edited_file in edited_files:
            try:
                original = image_loader(original_file)
                edited = image_loader(edited_file)

                l = loss_fn_alex(original, edited)
                lpips_scores.append(l.item())
            except Exception as e:
                print(f'Error processing files {original_file} and {edited_file}: {e}')
                continue

    if lpips_scores:
        mean_lpips_score = np.mean(lpips_scores)
        one_minus_lpips = 1 - mean_lpips_score

        print(f'Mean LPIPS score: {mean_lpips_score}')
        print(f'1-LPIPS score: {one_minus_lpips}')
    else:
        print("No LPIPS scores were calculated. Please check the input directories and files.")

