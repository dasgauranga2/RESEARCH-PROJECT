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
import matplotlib.pyplot as plt
import json

# DEVISE A NEW METRIC THAT SOMEHOW
# CORRELATES MULTIMODAL LLM PERFORMANCE AND NEGATIVE IMAGE SIMILARITY

# function to create a negative image by 
# randomly cropping out a portion of the image
def random_crop(image):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# function to create a negative image
# which is entirely black
def black_image(image_tensor):
    return torch.zeros_like(image_tensor)

# function to create a negative image by 
# randomly rotating the image
def rotate_image(image):
    rotator = v2.RandomRotation(degrees=(10, 80))  # fixed angle
    return rotator(image.squeeze(0)).unsqueeze(0)

# function to create a negative image by 
# applying the forward difussion process to the image
def forward_diffusion(image, step=500):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    #alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    #one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    #noise_delta = int(step)  # from 0-999
    noisy_image = image.squeeze(0)
    image_tensor_cd = q_x(noisy_image, step)

    return image_tensor_cd.unsqueeze(0)

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
corrupted_image_tensors = [random_crop(image_tensor),
                          black_image(image_tensor),
                          rotate_image(image_tensor),
                          forward_diffusion(image_tensor, 200)]

# convert image tensor back back to PIL Image
corrupted_images = [to_pil(corrupted_image_tensor.squeeze()) for corrupted_image_tensor in corrupted_image_tensors]

# no. of columns
cols = 3
# no. of rows
rows = (len(corrupted_images) // cols) + 1

# figure for the original image and the corrupted images
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
axes = axes.flatten()

# plot the original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the corrupted images
for i in range(len(corrupted_images)):
    axes[i+1].imshow(corrupted_images[i])
    axes[i+1].set_title("Corrupted Image")
    axes[i+1].axis('off')

# turn off remaining plots
for i in range(len(corrupted_images)+1, len(axes)):
    axes[i].axis('off')

# save the images
plt.savefig(f'./results/neg_img_metric.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()