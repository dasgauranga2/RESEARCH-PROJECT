import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import random
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale, adjust_hue
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import PiecewiseAffineTransform, warp
from io import BytesIO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
import os
import re

# function to apply elastic warping on an image
def elastic_transform(image, alpha=500, sigma=20):
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # shape: (H, W, C)
    H, W = image_np.shape[:2]

    # Generate displacement fields
    random_state = np.random.RandomState(None)
    dx = ndimage.gaussian_filter((random_state.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create meshgrid
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply displacement to each channel
    distorted = np.zeros_like(image_np)
    for i in range(image_np.shape[2]):
        distorted[..., i] = ndimage.map_coordinates(image_np[..., i], indices, order=1, mode='reflect').reshape(H, W)

    # Convert back to torch tensor
    distorted_tensor = torch.from_numpy(distorted).permute(2, 0, 1).unsqueeze(0).float()

    return distorted_tensor

# function to perform object detection on an image
# and return a binary mask which specifies the region
# where an object is detected
def elastic_object_detection(model, weights, image_tensors):
    # initialize the preprocessor
    preprocess = weights.transforms()

    # apply preprocessing to the image
    batch = [preprocess(image_tensor).to(device) for image_tensor in image_tensors]

    # get the model predictions
    # each prediction is of the format :-
    # {'boxes': tensor([[ 50.0912, 179.3047, 852.6385, 732.4894]], grad_fn=<StackBackward0>),
    # 'labels': tensor([17]),
    # 'scores': tensor([0.9971], grad_fn=<IndexBackward0>)}
    predictions = model(batch)

    # all object bounding box masks
    all_masks = []

    for i in range(len(image_tensors)):
        # get the image dimensions
        H, W = image_tensors[i].shape[1:]
        # create a binary mask that determines the region of detected object
        mask = torch.zeros((H, W), dtype=torch.float32)

        # get the bounding boxes
        boxes = predictions[i]['boxes']
        # get the bounding box confidence scores
        scores = predictions[i]['scores']

        # if not objects are detected
        if len(boxes) == 0:
            # we will apply global warping to the entire image
            mask[:,:] = 1
        else:
            # get the maximum score
            max_score = scores.max()
            # threshold for each bounding box
            keep = [scores >= 0.8*max_score]

            for box in boxes[keep]:
                x1, y1, x2, y2 = box.int()
                mask[y1:y2, x1:x2] = 1

        all_masks.append(mask)

    return all_masks, predictions

# function when given a list of bounding boxes
# will filter out those bounding boxes which are 
# contained within other bounding boxes
def filter_contained_boxes(boxes):
    keep = []
    n = boxes.size(0)

    for i in range(n):
        xi1, yi1, xi2, yi2 = boxes[i]
        contained = False

        for j in range(n):
            if i == j:
                continue

            xj1, yj1, xj2, yj2 = boxes[j]

            # Check if box i is fully inside box j
            if xi1 >= xj1 and yi1 >= yj1 and xi2 <= xj2 and yi2 <= yj2:
                contained = True
                break

        if not contained:
            keep.append(i)

    if len(keep) == 0:
        return boxes[:1]  # fallback: return at least one box
    
    return boxes[keep]

# function to perform object detection on an image
# and draw bounding boxes on the image
def draw_bounding_box(model, weights, image_tensors):
    # initialize the preprocessor
    preprocess = weights.transforms()

    # apply preprocessing to the image
    batch = [preprocess(image_tensor).to(device) for image_tensor in image_tensors]

    # get the model predictions
    # each prediction is of the format :-
    # {'boxes': tensor([[ 50.0912, 179.3047, 852.6385, 732.4894]], grad_fn=<StackBackward0>),
    # 'labels': tensor([17]),
    # 'scores': tensor([0.9971], grad_fn=<IndexBackward0>)}
    predictions = model(batch)

    # images with bounding boxes drawn on them
    modified_images = []
    for i in range(len(image_tensors)):
        # get each image
        modified_image = image_tensors[i].clone()
        # get dimensions of image
        H, W = image_tensors[i].shape[1:]

        # convert image to uint8 in the range [0, 255]
        if modified_image.max() <= 1:
            modified_image = (modified_image*255).to(torch.uint8)
        else:
            modified_image = modified_image.to(torch.uint8)

        # get bounding box locations
        boxes = predictions[i]['boxes']
        # get bounding box scores
        scores = predictions[i]['scores']

        # total area of image
        image_area = H*W
        # calculate area of each box
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # if a bounding box is too large we will filter them out
        # this will be used to remove such bounding boxes
        too_big = box_areas >= 0.8*image_area
        # we remove large bounding boxes only if there are multiple of them
        if len(boxes) > 1:
            boxes = boxes[~too_big]
            scores = scores[~too_big]

        # check if no object is detected
        if len(scores) == 0:
            # create a random bounding box
            x1 = random.randint(0, W // 2)
            y1 = random.randint(0, H // 2)
            x2 = random.randint(x1 + 1, W)
            y2 = random.randint(y1 + 1, H)
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        else:
            max_score = scores.max()
            keep = scores >= 0.85*max_score
            boxes = boxes[keep]
            scores = scores[keep]
        
        # filter out those bounding boxes that are contained within
        # other bounding boxes
        boxes = filter_contained_boxes(boxes)

        # draw the bounding boxes
        modified_image = draw_bounding_boxes(
            modified_image,
            boxes=boxes,
            colors=random.choice(['red', 'blue', 'green', 'yellow']),
            fill=True,
            width=2
        )

        modified_images.append(modified_image)
    
    return modified_images

# function to perform object detection on an image
# and apply elastic warping on the bounding box regions
def apply_elastic_warping(model, weights, image_tensors):
    # initialize the preprocessor
    preprocess = weights.transforms()

    # apply preprocessing to the image
    batch = [preprocess(image_tensor).to(device) for image_tensor in image_tensors]

    # get the model predictions
    # each prediction is of the format :-
    # {'boxes': tensor([[ 50.0912, 179.3047, 852.6385, 732.4894]], grad_fn=<StackBackward0>),
    # 'labels': tensor([17]),
    # 'scores': tensor([0.9971], grad_fn=<IndexBackward0>)}
    predictions = model(batch)

    # images with bounding boxes drawn on them
    obj_det_images = []
    # images with elastic warping on the bounding box regions
    el_warp_images = []
    for i in range(len(image_tensors)):
        # get dimensions of image
        H, W = image_tensors[i].shape[1:]

        # create two copies of image tensor
        # int - [0,255]
        # float - [0,1]
        if image_tensors[i].max() > 1:
            int_image = image_tensors[i].clone().to(torch.uint8)
            float_image = int_image.to(torch.float32) / 255.0
        else:
            float_image = image_tensors[i].clone().to(torch.float32)
            int_image = (float_image * 255).to(torch.uint8)

        # get bounding box locations
        boxes = predictions[i]['boxes']
        # get bounding box scores
        scores = predictions[i]['scores']

        # total area of image
        image_area = H*W
        # calculate area of each box
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # if a bounding box is too large we will filter them out
        # this will be used to remove such bounding boxes
        too_big = box_areas >= 0.8*image_area
        # we remove large bounding boxes only if there are multiple of them
        if len(boxes) > 1:
            boxes = boxes[~too_big]
            scores = scores[~too_big]

        # check if no object is detected
        if len(scores) == 0:
            # create a random bounding box
            x1 = random.randint(0, W // 2)
            y1 = random.randint(0, H // 2)
            x2 = random.randint(x1 + 1, W)
            y2 = random.randint(y1 + 1, H)
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        else:
            max_score = scores.max()
            keep = scores >= 0.85*max_score
            boxes = boxes[keep]
            scores = scores[keep]
        
        # filter out those bounding boxes that are contained within
        # other bounding boxes
        boxes = filter_contained_boxes(boxes)

        # draw the bounding boxes
        obj_det_image = draw_bounding_boxes(
            int_image,
            boxes=boxes,
            colors=random.choice(['red', 'blue', 'green', 'yellow']),
            fill=True,
            width=2
        )
        obj_det_images.append(obj_det_image)

        # iterate through each box
        for box in boxes:
            # get coordinates of each bounding box
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)

            # extract the region of image where bounding box is detected
            roi = float_image[:, y1:y2, x1:x2].unsqueeze(0).float()

            # apply elastic warping on that region
            warped_roi = elastic_transform(roi, alpha=300, sigma=15).squeeze(0)

            # Clamp and replace the region in the original image
            float_image[:, y1:y2, x1:x2] = warped_roi.clamp(0,1)
        
        el_warp_images.append(float_image)
    
    return obj_det_images, el_warp_images

# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# object to convert a PIL image into a Pytorch tensor
to_tensor = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the object detection model
frcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
frcnn_model = fasterrcnn_resnet50_fpn_v2(weights=frcnn_weights).to(device)
frcnn_model.eval()

# open the training data json file
with open('./data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)
print(len(data))
#print(data[0])

# APPLY IMAGE CORRUPTION TO ENTIRE DATASET AND SAVE THEM
# image file names
image_names = [sample['img_path'] for sample in data]
print(len(image_names), len(set(image_names))) # some images are repeated

# list of images
images = []
for image_name in tqdm(image_names, desc='Loading images'):
    images.append(Image.open('./data/merged_images/' + image_name).convert("RGB"))

print(len(images))

# list of image tensors
image_tensors = []
for image in tqdm(images, desc='Converting images to tensors'):
    image_tensors.append(to_tensor(image))

print(len(image_tensors))

# list of image tensors with bounding box on them
custom_corrupted_image_tensors = []
for i in tqdm(range(0, len(image_tensors), 4), desc='Applying object detection'):
    batch_image_tensors = image_tensors[i:i+4]
    #batch_corrupted_image_tensors = draw_bounding_box(frcnn_model, frcnn_weights, batch_image_tensors)
    _, batch_corrupted_image_tensors = apply_elastic_warping(frcnn_model, frcnn_weights, batch_image_tensors)
    custom_corrupted_image_tensors += batch_corrupted_image_tensors

print(len(custom_corrupted_image_tensors))

# save the images
for i in tqdm(range(len(custom_corrupted_image_tensors)), desc='Converting tensors back to images'):
    pil_image = to_pil(custom_corrupted_image_tensors[i].squeeze())
    save_path = './data/merged_images_corrupted/' + image_names[i]
    pil_image.save(save_path)

print(len(os.listdir('./data/merged_images_corrupted/')))

# # APPLY IMAGE CORRUPTION TO SOME IMAGES FROM THE DATASET
# # randomly select some images
# images = []
# for sample in random.sample(data, 6):
#     images.append(Image.open('./data/merged_images/' + sample['img_path']).convert("RGB"))

# # convert the images to tensors
# image_tensors = [to_tensor(image) for image in images]

# # # get the mask which defines the location of the object
# # object_masks, _ = object_detection(frcnn_model, frcnn_weights, image_tensors)

# # apply custom image corruption only on the bounding box
# #custom_corrupted_image_tensors = [elastic_transform(image_tensor, object_mask) for image_tensor, object_mask in zip(image_tensors, object_masks)]
# #custom_corrupted_image_tensors = draw_bounding_box(frcnn_model, frcnn_weights, image_tensors)
# custom_corrupted_image_tensors1, custom_corrupted_image_tensors2 = apply_elastic_warping(frcnn_model, frcnn_weights, image_tensors)

# # convert image tensor back back to PIL Image
# #custom_corrupted_images = [to_pil(custom_corrupted_image_tensor.squeeze()) for custom_corrupted_image_tensor in custom_corrupted_image_tensors]
# custom_corrupted_images1 = [to_pil(custom_corrupted_image_tensor.squeeze()) for custom_corrupted_image_tensor in custom_corrupted_image_tensors1]
# custom_corrupted_images2 = [to_pil(custom_corrupted_image_tensor.squeeze()) for custom_corrupted_image_tensor in custom_corrupted_image_tensors2]

# # figure for the original image and the corrupted images
# fig, axes = plt.subplots(len(images), 3, figsize=(15, 20))
# axes = axes.flatten()

# for i in range(len(images)):
#     # # plot the original image
#     # axes[i*2].imshow(images[i])
#     # axes[i*2].set_title("Original Image")
#     # axes[i*2].axis('off')

#     # # plot the custom corrupted image
#     # axes[(i*2)+1].imshow(custom_corrupted_images[i])
#     # axes[(i*2)+1].set_title(f"Custom Corruption")
#     # axes[(i*2)+1].axis('off')

#     # plot the original image
#     axes[i*3].imshow(images[i])
#     axes[i*3].set_title("Original Image")
#     axes[i*3].axis('off')

#     # plot the custom corrupted image
#     axes[(i*3)+1].imshow(custom_corrupted_images1[i])
#     axes[(i*3)+1].set_title(f"Custom Corruption")
#     axes[(i*3)+1].axis('off')

#     # plot the custom corrupted image
#     axes[(i*3)+2].imshow(custom_corrupted_images2[i])
#     axes[(i*3)+2].set_title(f"Custom Corruption")
#     axes[(i*3)+2].axis('off')

# # save the images
# plt.savefig(f'./results/obj_det_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()