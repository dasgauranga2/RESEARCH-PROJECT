from ultralytics import YOLO
import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import json
import random
from tqdm import tqdm
from torchvision import transforms

# set device
device = torch.device("cuda")

# load the model
yolo_model = YOLO("YOLO/yolo11x-seg.pt").to(device)

# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# function to remove right-half of each segmentation mask
def remove_right_half_mask(masks):
    masks_modified = masks.clone()
    print(f"MASK SHAPE: {masks.shape}")

    for i in range(masks.shape[0]):
        mask = masks[i]
        # Get bounding box of the mask
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        if not rows.any() or not cols.any():
            continue  # skip empty mask

        y_min, y_max = rows.nonzero()[0].item(), rows.nonzero()[-1].item()
        x_min, x_max = cols.nonzero()[0].item(), cols.nonzero()[-1].item()

        # Compute vertical middle of the detected region
        x_mid = (x_min + x_max) // 2

        # Zero out the right half of the mask within its own bounding box
        masks_modified[i, y_min:y_max+1, x_mid+1:x_max+1] = False

    return masks_modified

# function to draw segmentation masks on an image
def draw_seg_mask(model, image_path):
    
    # run inference on image
    results = model(image_path)

    # load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image).mul(255).to(torch.uint8)  # shape: (C, H, W)

    # check if no masks were found
    if results[0].masks is None or results[0].masks.data is None:
        print("NO OBJECT DETECTED")
        return image_tensor, image_tensor

    # get the segmentation masks
    masks = results[0].masks.data # shape: (N, H, W)

    # dimensions of image
    _, H, W = image_tensor.shape

    # scale the masks to original scale
    masks = F.interpolate(
        masks.unsqueeze(1),
        size=(H,W),
        mode='nearest'
    ).squeeze(1).bool()

    # modify the masks such that
    # the right-half is removed
    left_masks = remove_right_half_mask(masks)

    # draw the segmentation masks
    image_with_masks = draw_segmentation_masks(image_tensor.cpu(),
                                            masks,
                                            alpha=0.5,
                                            colors=['blue']*masks.shape[0])
    
     # draw the left-half segmentation masks
    image_with_left_imasks = draw_segmentation_masks(image_tensor.cpu(),
                                            left_masks,
                                            alpha=0.5,
                                            colors=['blue']*masks.shape[0])
    
    return image_with_masks, image_with_left_imasks

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# # APPLY IMAGE CORRUPTION TO SOME IMAGES
# # randomly select some images
# image_paths = []
# for sample in random.sample(data, 6):
#     image_paths.append('mDPO/data/merged_images/' + sample['img_path'])

# # open the images
# images = []
# for image_path in image_paths:
#     images.append(Image.open(image_path).convert("RGB"))

# # draw segmentation masks on the images
# corrupted_images1 = []
# corrupted_images2 = []
# for image_path in image_paths:
#     corrupted_images = draw_seg_mask(yolo_model, image_path)

#     corrupted_images1.append(corrupted_images[0])
#     corrupted_images2.append(corrupted_images[1])

# # figure for the original image and the corrupted images
# fig, axes = plt.subplots(len(images), 3, figsize=(10, 20))
# axes = axes.flatten()

# for i in range(len(images)):

#     # plot the original image
#     axes[(i*3)].imshow(images[i])
#     axes[(i*3)].set_title("Original Image")
#     axes[(i*3)].axis('off')

#     # plot the corrupted image
#     axes[(i*3)+1].imshow(corrupted_images1[i].permute(1, 2, 0))
#     axes[(i*3)+1].set_title(f"Custom Corruption")
#     axes[(i*3)+1].axis('off')

#     # plot the corrupted image
#     axes[(i*3)+2].imshow(corrupted_images2[i].permute(1, 2, 0))
#     axes[(i*3)+2].set_title(f"Custom Corruption")
#     axes[(i*3)+2].axis('off')

# # save the images
# plt.savefig(f'mDPO/results/img_seg_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# APPLY IMAGE CORRUPTION TO THE ENTIRE DATASET
# image paths
image_paths = []
# image names
image_names = []
for sample in tqdm(data, desc='Saving image paths and names'):
    image_paths.append('mDPO/data/merged_images/' + sample['img_path'])
    image_names.append(sample['img_path'])

assert len(image_paths)==len(image_names), 'Different lengths'

# draw segmentation masks on the images
corrupted_images1 = []
corrupted_images2 = []
for image_path in tqdm(image_paths, desc='Drawing segmentation masks'):
    corrupted_images = draw_seg_mask(yolo_model, image_path)
    corrupted_images1.append(corrupted_images[0])
    corrupted_images2.append(corrupted_images[1])

assert len(image_paths)==len(corrupted_images1), 'Different lengths of corrupted images1'
assert len(image_paths)==len(corrupted_images2), 'Different lengths of corrupted images2'

# save the images
for i in tqdm(range(len(image_names)), desc='Saving images'):
    pil_image1 = to_pil(corrupted_images1[i].squeeze())
    pil_image2 = to_pil(corrupted_images2[i].squeeze())

    save_path1 = 'mDPO/data/merged_images_corrupted1/' + image_names[i]
    save_path2 = 'mDPO/data/merged_images_corrupted2/' + image_names[i]

    pil_image1.save(save_path1)
    pil_image2.save(save_path2)