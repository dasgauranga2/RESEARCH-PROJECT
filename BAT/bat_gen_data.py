from ultralytics import YOLO
import torch
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from tqdm import tqdm
#from openai import OpenAI, RateLimitError, InternalServerError
import os
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# set device
device = torch.device("cuda")

# load object detection model
model = YOLO("./BAT/yolo11x-seg.pt").to(device)  # load an official model

# open the training data json file
with open('./mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# load the background images where we will place objects
bg_images = []
for filename in os.listdir('./BAT/bg_images/'):
    filepath = os.path.join('./BAT/bg_images/', filename)
    try:
        bg_image = Image.open(filepath).convert("RGB")
        stem, _ = os.path.splitext(filename)
        bg_images.append((bg_image, stem))
    except:
        print(f"Could not load image at {filepath}")

# function to place segmented objects on background image
def place_objects(obj_masks, orig_image, height, width, background):
    # dimensions of segmentation masks
    h, w = obj_masks.shape[-2], obj_masks.shape[-1]
    if (h, w) != (height, width):
        obj_masks = F.interpolate(obj_masks.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)  # [N,H,W]
    
    # build union of masks
    union_mask = (obj_masks > 0.5).any(dim=0)  # bool [H,W]
    # convert to PIL mask
    pil_mask = Image.fromarray((union_mask.numpy().astype(np.uint8) * 255), mode="L")
    pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    
    # resize background image
    background_resized = background.resize((width, height), Image.BICUBIC)

    # put segemented objects in background image
    edited_image = Image.composite(
        orig_image,
        background_resized,
        pil_mask
    )

    return edited_image

# # list of original images
# orig_images = []
# # list of images with objects placed on different background
# edited_images = []

# list to save the data
save_data = []

for sample in random.sample(data, 500):
    # original image file name
    image_name = sample['img_path']
    # image path
    image_path = './mDPO/data/merged_images/' + image_name
    # open the image 
    image = Image.open(image_path).convert("RGB")
    # dimensions of image
    W, H = image.size

    # run image segmentation
    result = model(
        source=image_path,
        retina_masks=True,
        #conf=CONF,
        #iou=IOU,
        #imgsz=IMGSZ,
        verbose=False
    )[0]

    if result.masks is None or result.masks.data is None:
        continue
    else:
        # segmentation masks
        # shape: [num_objects, height, width]
        seg_masks = result.masks.data.float().detach().cpu()

        # iterate through background image
        for bg_image, bg_stem in bg_images:
            # put segmented objects on background image
            db_image = place_objects(seg_masks, image, H, W, bg_image)

            # separate image name and image format
            splits = image_name.split('.')
            # new image file name
            new_image_name = splits[0] + f'_{bg_stem}' + '.' + splits[1]
            
            #print(new_image_name)
            # save image
            db_image.save('./BAT/eval_images/' + new_image_name)

        # confidence scores
        conf = result.boxes.conf.detach()
        # index of most confident score
        im = int(conf.argmax())
        # class id of object category with most confident score
        cls_id = int(result.boxes.cls[im].item())
        # class name of object category with most confident score
        cls_name = result.names[cls_id]
        #oriscore = float(conf[im].item())
        #print(conf, im, cls_id, cls_name, score)

        original = sample.copy()
        original['most_conf_class'] = cls_name
        save_data.append(original)

        # orig_images.append(image)
        # edited_images.append(seg_image)

# save the generated data in a json file
with open('./BAT/bat_data.json', 'w') as f:
    json.dump(save_data, f, indent=4)

# # figure for the original image and the edited images
# fig, axes = plt.subplots(len(orig_images), 2, figsize=(8, 3*len(orig_images)))
# axes = axes.flatten()

# for i in range(len(orig_images)):
#     # plot the original image
#     axes[(i*2)].imshow(orig_images[i])
#     axes[(i*2)].set_title("Original Image")
#     axes[(i*2)].axis('off')

#     # plot the chosen image
#     axes[(i*2)+1].imshow(edited_images[i])
#     axes[(i*2)+1].set_title("Edited Image")
#     axes[(i*2)+1].axis('off')

# # save the images
# plt.savefig(f'BAT/bat_test_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()