import os
import sys
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import concurrent.futures

def calculate_ssim(image1, image2):
    # 将图像转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # 计算SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return score

def calculate_average_ssim(images):
    num_images = len(images)
    total_ssim = 0
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                futures.append(executor.submit(calculate_ssim, images[i], images[j]))
        
        for future in concurrent.futures.as_completed(futures):
            total_ssim += future.result()
    
    average_ssim = total_ssim / (num_images * (num_images - 1) / 2)
    return average_ssim

def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def calculate_average_ssim_for_folder(folder):
    images = read_images_from_folder(folder)
    avg_ssim = calculate_average_ssim(images)
    return avg_ssim

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py folder_name")
        sys.exit(1)
    
    folder_name = sys.argv[1]
    avg_ssim = calculate_average_ssim_for_folder(folder_name)
    print("Average SSIM for images in", folder_name, "folder:", avg_ssim)

