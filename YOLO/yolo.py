from ultralytics import YOLO
import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import json
import random

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
yolo_model = YOLO("YOLO/yolo11x-seg.pt").to(device)

# function to draw segmentation masks on an image
def draw_seg_mask(model, image_path):
    
    # run inference on image
    results = model(image_path)

    # load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image).mul(255).to(torch.uint8)  # shape: (C, H, W)

    # get the segmentation masks
    masks = results[0].masks.data # shape: (N, H, W)

    # dimensions of image
    _, H, W = image_tensor.shape

    # scale the masks to original scale
    masks = F.interpolate(
        masks.unsqueeze(1),
        size=(H,W),
        mode='nearest'
    ).squeeze().bool()

    # draw the segmentation masks
    image_with_masks = draw_segmentation_masks(image_tensor.cpu(),
                                            masks,
                                            alpha=0.5,
                                            colors=['blue']*masks.shape[0])
    return image_with_masks

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# randomly select some images
image_paths = []
for sample in random.sample(data, 6):
    image_paths.append('mDPO/data/merged_images/' + sample['img_path'])

# open the images
images = []
for image_path in image_paths:
    images.append(Image.open(image_path).convert("RGB"))

# draw segmentation masks on the images
corrupted_images = []
for image_path in image_paths:
    corrupted_image = draw_seg_mask(yolo_model, image_path)
    corrupted_images.append(corrupted_image)

# figure for the original image and the corrupted images
fig, axes = plt.subplots(len(images), 2, figsize=(15, 20))
axes = axes.flatten()

for i in range(len(images)):

    # plot the original image
    axes[(i*2)].imshow(images[i])
    axes[(i*2)].set_title("Original Image")
    axes[(i*2)].axis('off')

    # plot the corrupted image
    axes[(i*2)+1].imshow(corrupted_images[i].permute(1, 2, 0))
    axes[(i*2)+1].set_title(f"Custom Corruption")
    axes[(i*2)+1].axis('off')

# save the images
plt.savefig(f'mDPO/results/img_seg_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()