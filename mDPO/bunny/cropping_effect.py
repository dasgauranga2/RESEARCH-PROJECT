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
# TO MEASURE THEIR SHARPNESS

# function to apply the mDPO cropping
def crop_image(image):
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

# function to calculate the sobel variance in the pixel-intensity differences
# which measures the sharpness in the image
def sobel_variance(pil_image):
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
    return np.var(gradient_magnitude)

# list to store the differences between sobel variances of the images
mean_diff = []

for i in range(100):

    # open the image
    image = Image.open('./data/merged_images/' + data[i]['img_path']).convert("RGB")

    # convert the image to a tensor
    image_tensor = to_tensor(image)

    # apply image corruption
    corrupted_image_tensor = crop_image(image_tensor)

    # convert image tensor back back to PIL Image
    corrupted_image = to_pil(corrupted_image_tensor.squeeze())

    # get the sobel variances of both the images
    orig_pvar = sobel_variance(image)
    corrup_pvar = sobel_variance(corrupted_image)

    mean_diff.append(orig_pvar-corrup_pvar)

#print(image)
print(mean_diff)
print(sum(mean_diff) / len(mean_diff))