import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from torchvision.models import vit_h_14, ViT_H_14_Weights
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import json

# set device
device = 'cuda'
torch.set_default_device(device)

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
        #print(x_0.device, alphas_t.device, alphas_1_m_t.device, noise.device)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    #noise_delta = int(step)  # from 0-999
    noisy_image = image.squeeze(0)
    image_tensor_cd = q_x(noisy_image, step)

    return image_tensor_cd.unsqueeze(0)

# function that will return the encoder outputs
def model_encoder_outputs(model, weights, img_tensor):
    # set model to evaluation mode
    model.eval()

    # list where encoder outputs will be stored
    encoder_outputs = []

    # image preprocessor
    preprocess = weights.transforms()

    # create batch of data
    batch = preprocess(img_tensor.squeeze().to(device)).unsqueeze(0)

    # forward hook function
    # this will be called during the forward pass of each encoder layer
    # that is hooked to this function
    # module: the Pytorch module to which we will attach this hook function
    # input: inputs to that module
    # output: output of the forward pass
    def save_output(module, input, output):
        # save the module outputs to the list
        encoder_outputs.append(output.detach())

    hooks = []
    # iterate over each encoder layer
    for layer in model.encoder.layers:
        # register the function above
        # as a forward hook
        hook = layer.register_forward_hook(save_output)
        # save the hook if we want to remove it later
        hooks.append(hook)

    # get model outputs
    _ = model(batch)

    # remove each hook
    for hook in hooks:
        hook.remove()

    return encoder_outputs

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
image_tensor = to_tensor(image).to(device)

# apply mDPO image corruption
corrupted_image_tensors = [random_crop(image_tensor),
                          black_image(image_tensor),
                          rotate_image(image_tensor),
                          forward_diffusion(image_tensor, 200)]

# initialize the ViT model
vit_weights = ViT_H_14_Weights.DEFAULT
vit_model = vit_h_14(weights=vit_weights)

# get the encoder outputs for original image
orig_outputs = model_encoder_outputs(vit_model, vit_weights, image_tensor)
# get the encoder outputs for corrupted images
rc_outputs = model_encoder_outputs(vit_model, vit_weights, corrupted_image_tensors[0])
black_outputs = model_encoder_outputs(vit_model, vit_weights, corrupted_image_tensors[1])
rotated_outputs = model_encoder_outputs(vit_model, vit_weights, corrupted_image_tensors[2])

print(len(orig_outputs), orig_outputs[0].shape, orig_outputs[-1].shape)
print(len(rc_outputs), rc_outputs[0].shape, rc_outputs[-1].shape)
print(len(black_outputs), black_outputs[0].shape, black_outputs[-1].shape)
print(len(rotated_outputs), rotated_outputs[0].shape, rotated_outputs[-1].shape)

# # convert image tensor back back to PIL Image
# corrupted_images = [to_pil(corrupted_image_tensor.squeeze()) for corrupted_image_tensor in corrupted_image_tensors]

# # no. of columns
# cols = 3
# # no. of rows
# rows = (len(corrupted_images) // cols) + 1

# # figure for the original image and the corrupted images
# fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
# axes = axes.flatten()

# # plot the original image
# axes[0].imshow(image)
# axes[0].set_title("Original Image")
# axes[0].axis('off')

# # plot the corrupted images
# for i in range(len(corrupted_images)):
#     axes[i+1].imshow(corrupted_images[i])
#     axes[i+1].set_title("Corrupted Image")
#     axes[i+1].axis('off')

# # turn off remaining plots
# for i in range(len(corrupted_images)+1, len(axes)):
#     axes[i].axis('off')

# # save the images
# plt.savefig(f'./results/neg_img_metric.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()