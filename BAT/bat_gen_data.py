from ultralytics import YOLO
import torch
from diffusers import StableDiffusion3Pipeline
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from tqdm import tqdm
from openai import OpenAI, RateLimitError, InternalServerError
import os
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# set device
device = torch.device("cuda")

# # get the OpenAI API key
# with open("mDPO/MMHal-Bench/api.txt", "r") as f:
#     API_KEY = f.read().strip()

# # openai client
# openai_client = OpenAI(api_key=API_KEY)

# load object detection model
model = YOLO("BAT/yolo11x-seg.pt").to(device)  # load an official model

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# load the background images where we will place objects
bg_images = []
for filename in os.listdir('BAT/bg_images/'):
    filepath = os.path.join('BAT/bg_images/', filename)
    try:
        bg_image = Image.open(filepath).convert("RGB")
        bg_images.append(bg_image)
    except:
        print(f"Could not load image at {filepath}")

# list of original images
orig_images = []
# list of images with objects placed on background
edited_images = []

for sample in random.sample(data, 6):
    # image path
    image_path = 'mDPO/data/merged_images/' + sample['img_path']
    # open the image 
    image = Image.open(image_path).convert("RGB")
    # dimensions of image
    W, H = image.size

    # run image segmentation
    result = model(
        source=image_path,
        #conf=CONF,
        #iou=IOU,
        #imgsz=IMGSZ,
        #verbose=False
    )[0]

    if result.masks is None or result.masks.data is None:
        continue
    else:
        # segmentation masks
        # shape: [num_objects, height, width]
        seg_masks = result.masks.data.float().detach().cpu()

        # dimensions of segmentation masks
        h, w = seg_masks.shape[-2], seg_masks.shape[-1]
        if (h, w) != (H, W):
            seg_masks = F.interpolate(seg_masks.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)  # [N,H,W]
        
        # build union of masks
        union_mask = (seg_masks > 0.5).any(dim=0)  # bool [H,W]
        # convert to PIL mask
        pil_mask = Image.fromarray((union_mask.numpy().astype(np.uint8) * 255), mode="L")
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2.0))
        
        # resize background image
        bg_image_rs = bg_images[0].resize((W, H), Image.BICUBIC)

        # put segemented objects in background image
        seg_image = Image.composite(
            image,
            bg_image_rs,
            pil_mask
        )

        orig_images.append(image)
        edited_images.append(seg_image)

# figure for the original image and the edited images
fig, axes = plt.subplots(len(orig_images), 2, figsize=(8, 3*len(orig_images)))
axes = axes.flatten()

for i in range(len(orig_images)):
    # plot the original image
    axes[(i*2)].imshow(orig_images[i])
    axes[(i*2)].set_title("Original Image")
    axes[(i*2)].axis('off')

    # plot the chosen image
    axes[(i*2)+1].imshow(edited_images[i])
    axes[(i*2)+1].set_title("Edited Image")
    axes[(i*2)+1].axis('off')

# save the images
plt.savefig(f'BAT/bat_test_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
