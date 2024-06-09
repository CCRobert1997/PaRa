from PIL import Image
import os
import re
import argparse
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def compute_cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='clipScore',
        description='Generate average CLIP score for images based on cosine similarity'
    )
    parser.add_argument('--im_path', help='Path for images', type=str, required=True)
    parser.add_argument('--prompt', help='Prompt to check CLIP score against', type=str, required=True)
    parser.add_argument('--device', help='CUDA device to run on', type=str, required=False, default='cuda:0')
    
    args = parser.parse_args()

    device = args.device
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_path = args.im_path
    prompt = args.prompt.strip()
    
    images = [os.path.join(image_path, fname) for fname in os.listdir(image_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = sorted_nicely(images)

    # Compute text embedding
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs)

    clip_scores = []

    for image_file in images:
        try:
            image = Image.open(image_file)
            image_inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**image_inputs)
                clip_score = compute_cosine_similarity(image_embedding, text_embedding)
                clip_scores.append(clip_score)
                #print(f"Processed {image_file}: CLIP score = {clip_score}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    if clip_scores:
        mean_clip_score = sum(clip_scores) / len(clip_scores)
        print(f"Average CLIP score: {mean_clip_score}")
    else:
        print("No valid CLIP scores were calculated. Please check the input directory and files.")
