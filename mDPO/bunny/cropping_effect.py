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

# COMPARING THE PIXEL-INTENSITY GRADIENT VARIANCES FOR BOTH ORIGINAL AND CORRUPTED IMAGES
# TO COMPARE THEIR SHARPNESS

# function to apply the mDPO image corruption
def corrupt_image(image):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# object to convert a PIL image into a Pytorch tensor
to_tensor = transforms.ToTensor()
# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# open the training data json file
with open('./data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# function to calculate the sharpness of an image
# using the sobel filter gradient values
def sobel_sharpness(pil_image):
    # convert the image to grayscale
    gray_image = pil_image.convert('L')
    
    # convert the grayscale image to a NumPy array
    image_array = np.array(gray_image, dtype=np.float32)
    
    # compute Sobel gradients along the x and y axes
    sobel_x = ndimage.sobel(image_array, axis=0, mode='reflect')
    sobel_y = ndimage.sobel(image_array, axis=1, mode='reflect')
    
    # compute the gradient magnitude
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    
    # calculate and return the variance of the gradient magnitude
    return np.mean(gradient_magnitude)

# function to crop a center patch of the image
def center_crop_image(pil_image, crop_width=40, crop_height=40):
    # get image size
    width, height = pil_image.size

    # define the crop box
    left = (width - crop_width) // 2
    upper = (height - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height

    # crop the image center
    cropped_image = pil_image.crop((left, upper, right, lower))

    return cropped_image

# list to store the differences between sobel variances of the images
diff = []

for i in range(500):

    # open the image
    image = Image.open('./data/merged_images/' + data[i]['img_path']).convert("RGB")

    # get image dimensions
    width, height = image.size

    # convert the image to a tensor
    image_tensor = to_tensor(image)

    # apply image corruption
    corrupted_image_tensor = corrupt_image(image_tensor)

    # convert image tensor back back to PIL Image
    corrupted_image = to_pil(corrupted_image_tensor.squeeze())

    # get the sobel sharpness of both the images
    orig_pvar = sobel_sharpness(center_crop_image(image))
    corrup_pvar = sobel_sharpness(center_crop_image(corrupted_image))

    diff.append(orig_pvar-corrup_pvar)

#print(image)
#print(mean_diff)
print(f"Sobel Variance: {sum(diff) / len(diff):.2f}")