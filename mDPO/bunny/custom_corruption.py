import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms
import numpy as np
from scipy import ndimage
import json
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

# function to apply the mDPO image corruption
def corrupt_image(image):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# function to apply elastic warping on an image
def elastic_transform_pil(pil_img, alpha=1000, sigma=20):
    # Convert to numpy
    image = np.array(pil_img)

    # Generate displacement fields
    random_state = np.random.RandomState(None)
    dx = ndimage.gaussian_filter((random_state.rand(*image.shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*image.shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create meshgrid
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Map image through displacement
    distorted = np.zeros_like(image)
    for i in range(image.shape[2]):
        distorted[..., i] = ndimage.map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(image.shape[:2])

    return Image.fromarray(distorted)

# object to convert a PIL image into a Pytorch tensor
to_tensor = transforms.ToTensor()
# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# # open the training data json file
# with open('./data/vlfeedback_llava_10k.json', 'r') as file:
#     data = json.load(file)

# open the image
image = Image.open('./data/test3.png').convert("RGB")

# # get image dimensions
# width, height = image.size

# convert the image to a tensor
image_tensor = to_tensor(image)

# apply image corruption
corrupted_image_tensor = corrupt_image(image_tensor)

# convert image tensor back back to PIL Image
mdpo_corrupted_image = to_pil(corrupted_image_tensor.squeeze())

custom_corrupted_image = elastic_transform_pil(image)

# figure for the original image and the attention maps
fig, axes = plt.subplots(1, 3, figsize=(8, 5))
axes = axes.flatten()

# plot the original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the mDPO corrupted image
axes[1].imshow(mdpo_corrupted_image)
axes[1].set_title("mDPO Corrupted Image")
axes[1].axis('off')

# plot the custom corrupted image
axes[2].imshow(custom_corrupted_image)
axes[2].set_title("Custom Corrupted Image")
axes[2].axis('off')

# save the images
plt.savefig(f'./results/image_corruption.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()