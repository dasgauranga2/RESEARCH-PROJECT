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
import torchvision.transforms.v2 as v2
import random

# set device
device = torch.device("cuda")

# load the model
yolo_model = YOLO("YOLO/yolo11x-seg.pt").to(device)

# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# # function to remove right-half of each segmentation mask
# def remove_half_mask(masks):
#     masks_modified1 = masks.clone()
#     masks_modified2 = masks.clone()
#     #print(f"MASK SHAPE: {masks.shape}")

#     for i in range(masks.shape[0]):
#         mask = masks[i]
#         # Get bounding box of the mask
#         rows = torch.any(mask, dim=1)
#         cols = torch.any(mask, dim=0)
#         if not rows.any() or not cols.any():
#             continue  # skip empty mask

#         y_min, y_max = rows.nonzero()[0].item(), rows.nonzero()[-1].item()
#         x_min, x_max = cols.nonzero()[0].item(), cols.nonzero()[-1].item()

#         # Compute vertical middle of the detected region
#         x_mid = (x_min + x_max) // 2

#         # Zero out the right half of the mask within its own bounding box
#         masks_modified1[i, y_min:y_max+1, x_mid+1:x_max+1] = False
#         # Zero out the left half of the mask within its own bounding box
#         masks_modified2[i, y_min:y_max+1, x_min+1:x_mid+1] = False

#     return masks_modified1, masks_modified2

# function to apply elastic warping on an image only on the segmented regions
def apply_elastic_warping(image_tensor, masks):
    # Ensure inputs are on CPU and detached
    image_tensor = image_tensor.detach().cpu()
    masks = masks.detach().cpu()

    # Convert to float and scale to [0,1] for torchvision transforms
    image = image_tensor.float() / 255.0

    # elastic warping parameters
    alpha = random.uniform(300.0, 400.0)
    sigma = random.uniform(5.0, 10.0)
    
    # Elastic transformer object
    transformer = v2.ElasticTransform(alpha=alpha, sigma=sigma)

    # Apply elastic transform to the full image
    warped_image = transformer(image)

    # Initialize final image as a copy of the original
    final_image = image.clone()

    # For each mask: apply only to the masked region
    for i in range(masks.shape[0]):
        mask = masks[i]  # (H, W)
        final_image[:, mask] = warped_image[:, mask]

    # Convert back to uint8
    return (final_image * 255.0).to(torch.uint8)

# function to filter large masks
def filter_large_masks(masks):
    # dimensions of masks
    _, H, W = masks.shape
    # total area of each
    total_area = H * W

    # Compute area for each mask (i.e., number of True pixels)
    mask_areas = masks.flatten(1).sum(dim=1)

    # Keep masks whose area is below threshold
    keep = mask_areas < (0.6 * total_area)

    return masks[keep]

# transform to resize an image
resize_transform = v2.Compose([
    v2.Resize((640, 640)),
])

# function to draw segmentation masks on an image and corrupt the image
def draw_seg_mask(model, image_paths):
    # load the images
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    # resized images
    resized_images = [resize_transform(image) for image in images]
    # convert to image tensors
    image_tensors = [to_tensor(image).mul(255).to(torch.uint8) for image in resized_images]  # shape: (C, H, W)

    # run image-segmentation on batch of images
    results = model(resized_images)
    #results = [model(image_path)[0] for image_path in image_paths]
    
    # list of images with segmentation masks
    segmented_images = []
    # list of images with elastic warping applied on the segmented regions
    corrupted_images = []

    for image_tensor, result in zip(image_tensors, results):
        # check if no masks were found
        if result.masks is None or result.masks.data is None:
            print("NO OBJECT DETECTED")
            segmented_images.append(image_tensor)
            corrupted_images.append(image_tensor)
            continue

        # get the segmentation masks
        masks = result.masks.data.bool() # shape: (N, H, W)

        # # modify the masks such that
        # # the right-half and left-half is removed
        # left_masks, right_masks = remove_half_mask(masks)

        # draw the segmentation masks
        image_with_masks = draw_segmentation_masks(image_tensor.cpu(),
                                                masks,
                                                alpha=0.5,
                                                colors=['blue']*masks.shape[0])
        
        segmented_images.append(image_with_masks)
        
        # filter the masks
        masks = filter_large_masks(masks)

        # corrupt the image
        corrupted_image = apply_elastic_warping(image_tensor, masks)

        corrupted_images.append(corrupted_image)
        
        # # draw the left-half segmentation masks
        # image_with_left_imasks = draw_segmentation_masks(image_tensor.cpu(),
        #                                         left_masks,
        #                                         alpha=0.5,
        #                                         colors=['blue']*masks.shape[0])
        
        #  # draw the left-half segmentation masks
        # image_with_right_imasks = draw_segmentation_masks(image_tensor.cpu(),
        #                                         right_masks,
        #                                         alpha=0.5,
        #                                         colors=['blue']*masks.shape[0])
    

    return segmented_images, corrupted_images

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# APPLY IMAGE CORRUPTION TO SOME IMAGES
# randomly select some images
image_paths = []
for sample in random.sample(data, 4):
    image_paths.append('mDPO/data/merged_images/' + sample['img_path'])

# open the images
images = []
for image_path in image_paths:
    images.append(Image.open(image_path).convert("RGB"))

# draw segmentation masks on the images and corrupt them
seg_images, corrupted_images = draw_seg_mask(yolo_model, image_paths)

# figure for the original image and the corrupted images
fig, axes = plt.subplots(len(images), 3, figsize=(15, 20))
axes = axes.flatten()

for i in range(len(images)):

    # plot the original image
    axes[(i*3)].imshow(images[i])
    axes[(i*3)].set_title("Original Image")
    axes[(i*3)].axis('off')

    # plot the image with segmentation masks
    axes[(i*3)+1].imshow(seg_images[i].permute(1, 2, 0))
    axes[(i*3)+1].set_title(f"Segmentation Masks")
    axes[(i*3)+1].axis('off')

    # plot the corrupted image
    axes[(i*3)+2].imshow(corrupted_images[i].permute(1, 2, 0))
    axes[(i*3)+2].set_title(f"Corrupted Image")
    axes[(i*3)+2].axis('off')

# save the images
plt.savefig(f'mDPO/results/img_seg_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()


# # APPLY IMAGE CORRUPTION TO THE ENTIRE DATASET
# # image paths
# image_paths = []
# # image names
# image_names = []
# for sample in tqdm(data, desc='Saving image paths and names'):
#     image_paths.append('mDPO/data/merged_images/' + sample['img_path'])
#     image_names.append(sample['img_path'])

# assert len(image_paths)==len(image_names), 'Different lengths'

# # get the corrupted images
# corrupted_images = []
# for image_path in image_paths:
#     result_images = draw_seg_mask(yolo_model, image_path)
    
#     corrupted_images.append(result_images[1])

# assert len(image_paths)==len(corrupted_images), 'Different lengths of corrupted images1'

# # save the images
# for i in tqdm(range(len(image_names)), desc='Saving images'):
#     pil_image = to_pil(corrupted_images[i].squeeze())

#     save_path = 'mDPO/data/merged_images_corrupted/' + image_names[i]

#     pil_image.save(save_path)
