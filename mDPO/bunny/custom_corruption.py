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
from skimage.transform import PiecewiseAffineTransform, warp
from io import BytesIO

# function to apply the mDPO image corruption
def corrupt_image(image):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# function to apply elastic warping on an image
# alpha: magnitude of distortion
# this controls how far the pixels are moved from their original positions
# sigma: smoothness of distortion
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

# function to shear an image
def shear_image(image, shear_deg=40):
    shearing = v2.RandomAffine(degrees=0, shear=shear_deg)
    image = shearing(image.squeeze(0)).unsqueeze(0)
    return image

# function to apply grid distortions on an image
def grid_distortion(image, num_steps=5, distort_limit=0.3):
    #assert image.ndim == 4 and image.shape[0] == 1, "Image must be of shape (1, C, H, W)"
    #B, C, H, W = image.shape
    device = image.device

    # Squeeze image
    image = image.squeeze(0)  # Shape: (C, H, W)
    C, H, W = image.shape

    # Create normalized meshgrid
    xx = torch.linspace(-1, 1, W, device=device)
    yy = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2)  # (H, W, 2)

    # Create distortion grid
    step_x = W // num_steps
    step_y = H // num_steps
    distortion = torch.rand((num_steps + 1, num_steps + 1, 2), device=device) * 2 - 1
    distortion *= distort_limit

    # Upsample distortion to match H x W
    distortion = distortion.permute(2, 0, 1).unsqueeze(0)  # (1, 2, grid_y, grid_x)
    distortion = F.interpolate(distortion, size=(H, W), mode='bicubic', align_corners=True)
    distortion = distortion.squeeze(0).permute(1, 2, 0)  # (H, W, 2)

    # Apply distortion to grid
    distorted_grid = grid + distortion
    distorted_grid = distorted_grid.unsqueeze(0)  # (1, H, W, 2)

    # Grid sample requires (1, C, H, W) and grid in range [-1, 1]
    warped = F.grid_sample(image.unsqueeze(0), distorted_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped

# function to randomly permute the colour channels
def shuffle_color_channels(image):
    # Squeeze batch dimension
    img = image.squeeze(0)  # shape: (C, H, W)
    
    # Generate a random permutation of the channel indices
    perm = torch.randperm(img.size(0))
    
    # Shuffle channels
    img_shuffled = img[perm]
    
    # Add back batch dimension
    return img_shuffled.unsqueeze(0)

# object to convert a PIL image into a Pytorch tensor
to_tensor = transforms.ToTensor()
# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# # open the training data json file
# with open('./data/vlfeedback_llava_10k.json', 'r') as file:
#     data = json.load(file)

# open the image
image = Image.open('./data/test3.png').convert("RGB")

# convert the image to a tensor
image_tensor = to_tensor(image)

# apply mDPO image corruption
mdpo_corrupted_image_tensor = corrupt_image(image_tensor)
# convert image tensor back back to PIL Image
mdpo_corrupted_image = to_pil(mdpo_corrupted_image_tensor.squeeze())

# figure for the original image and the corrupted images
fig, axes = plt.subplots(3, 2, figsize=(8, 10))
axes = axes.flatten()

# plot the original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the mDPO corrupted image
axes[1].imshow(mdpo_corrupted_image)
axes[1].set_title("mDPO Corrupted Image")
axes[1].axis('off')

# elastic warping parameters 
params = []
# generate the elastic warping parameters randomly
for _ in range(4):
    # generate alpha randomly
    alpha = random.uniform(100, 500)
    # generate sigma randomly
    sigma = random.uniform(10, 20)

    params.append((alpha, sigma))

# index to track which plot be used
i = 2

for param in params:
    # apply custom image corruption with different parameters
    custom_corrupted_image_tensor = elastic_transform(image_tensor, param[0], param[1])
    #custom_corrupted_image_tensor = shear_image(image_tensor)
    #custom_corrupted_image_tensor = grid_distortion(image_tensor)
    #custom_corrupted_image_tensor = shuffle_color_channels(image_tensor)
    
    # convert image tensor back back to PIL Image
    custom_corrupted_image = to_pil(custom_corrupted_image_tensor.squeeze())

    # plot the custom corrupted image
    axes[i].imshow(custom_corrupted_image)
    axes[i].set_title(f"Alpha {int(param[0])} Sigma {int(param[1])}")
    axes[i].axis('off')

    i += 1

# save the images
plt.savefig(f'./results/image_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()