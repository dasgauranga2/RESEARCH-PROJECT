from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import json
import random

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# load the model
sam = sam_model_registry["vit_h"](checkpoint="Segment Anything/sam_vit_h_4b8939.pth").to(device)

# automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# function to draw segmentation mask on the image
def draw_seg_mask(model, numpy_image):

    # generate masks for the image
    # {'segmentation': array([[False, False, False, ..., False, False, False],
    #    [False, False, False, ..., False, False, False],
    #    [False, False, False, ..., False, False, False],
    #    ...,
    #    [False, False, False, ..., False, False, False],
    #    [False, False, False, ..., False, False, False],
    #    [False, False, False, ..., False, False, False]]), 
    #   'area': 1552504, 'bbox': [0, 6, 1699, 1186], 'predicted_iou': 1.0378663539886475, 
    #   'point_coords': [[557.8125, 504.5625]], 'stability_score': 0.9833655953407288, 'crop_box': [0, 0, 1700, 1196]}
    masks = model.generate(numpy_image)

    # select only top masks
    top_masks = masks[:5]
    mask_list = [torch.from_numpy(m['segmentation']) for m in top_masks]

    # convert numpy array to tensor
    image_tensor = F.to_tensor(numpy_image).mul(255).to(torch.uint8)

    # stack masks into (N, H, W)
    mask_tensor = torch.stack(mask_list).bool()

    # draw the masks (random colors)
    image_with_masks = draw_segmentation_masks(image_tensor.cpu(), 
                                               mask_tensor.cpu(), 
                                               alpha=0.5, 
                                               colors=['blue']*5)
    
    return image_with_masks

# randomly select some images
images = []
for sample in random.sample(data, 6):
    images.append(Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB"))

# convert images to numpy arrays
numpy_images = [np.array(image) for image in images]

# figure for the original image and the corrupted images
fig, axes = plt.subplots(len(images), 2, figsize=(15, 20))
axes = axes.flatten()

for i in range(len(images)):
    # get the image with segmentation mask
    segm_image = draw_seg_mask(mask_generator, numpy_images[i])

    # plot the original image
    axes[(i*2)].imshow(images[i])
    axes[(i*2)].set_title("Original Image")
    axes[(i*2)].axis('off')

    # plot the corrupted image
    axes[(i*2)+1].imshow(segm_image.permute(1, 2, 0))
    axes[(i*2)+1].set_title(f"Custom Corruption")
    axes[(i*2)+1].axis('off')

# save the images
plt.savefig(f'mDPO/results/img_seg_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()