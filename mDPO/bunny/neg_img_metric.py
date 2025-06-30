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
import random
import torch.nn.functional as F
from collections import defaultdict

# set device
device = 'cuda'
torch.set_default_device(device)

# DEVISE A NEW METRIC THAT SOMEHOW
# CORRELATES MULTIMODAL LLM PERFORMANCE AND NEGATIVE IMAGE SIMILARITY

# function to create a negative image by 
# randomly cropping out a portion of the image
def random_crop(image, low=0.01, high=0.1):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(low, high))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# function to create a negative image
# which is entirely black
def black_image(image):
    return torch.zeros_like(image)

# function to create a negative image by 
# randomly rotating the image
def rotate_image(image, low=-40, high=40):
    rotator = v2.RandomRotation(degrees=(low, high))  # fixed angle
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

# function to create a negative image by 
# randomly replacing it with another image
def replace_with_random_images(images):
    new_images = []
    for i, img in enumerate(images):
        # indices excluding the current one
        other_indices = list(range(len(images)))
        other_indices.remove(i)

        # choose a different image
        rand_idx = random.choice(other_indices)
        new_images.append(images[rand_idx])
    return new_images

# function that will return a list of encoder outputs
# for an image tensor
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

# function when given two lists of encoder outputs of two different images
# will calculate the similarity between them
def similarity(enc_out_1, enc_out_2):

    # list of similarity scores
    similarities = []
    for enc1, enc2 in zip(enc_out_1, enc_out_2):
        # get CLS token of first
        enc1_cls = enc1.squeeze()[0,:]
        # get CLS token of second
        enc2_cls = enc2.squeeze()[0,:]

        # calculate cosine similarity between them
        cosine_similarity = F.cosine_similarity(enc1_cls, enc2_cls, dim=0)
        similarities.append(cosine_similarity)

    n = len(enc_out_1)
    mid = n // 2
    
    # lower-layer similarities
    low_similarities = sum(similarities[:4])/len(similarities[:4])
    # mid-layer similarities
    mid_similarities = sum(similarities[mid-2:mid+2])/len(similarities[mid-2:mid+2])
    # higher-layer similarities
    high_similarities = sum(similarities[-4:])/len(similarities[-4:])

    return low_similarities, mid_similarities, high_similarities

# function to calculate the average similarity
# between different images
def avg(scores):
    # average of lower layer scores
    lows = torch.stack([s[0] for s in scores]).mean().item()
    # average of mid layer scores
    mids = torch.stack([s[1] for s in scores]).mean().item()
    # average of higher layer scores
    highs = torch.stack([s[2] for s in scores]).mean().item()
    return lows, mids, highs

# object to convert a PIL image into a Pytorch tensor
to_tensor = transforms.ToTensor()
# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# open the training data json file
with open('./data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)
#print(data[0])

# open the image
#image = Image.open('./data/test3.png').convert("RGB")
# open some images from dataset randomly
images = []
for sample in random.sample(data, 20):
    images.append(Image.open('./data/merged_images/' + sample['img_path']).convert("RGB"))

# convert the images to tensors
image_tensors = [to_tensor(image).to(device) for image in images]

# apply different image corruption techniques
fd_100_image_tensors = [forward_diffusion(image_tensor, 100) for image_tensor in image_tensors]
fd_200_image_tensors = [forward_diffusion(image_tensor, 200) for image_tensor in image_tensors]
rr_40_image_tensors = [rotate_image(image_tensor, -40, 40) for image_tensor in image_tensors]
rc_1_10_image_tensors = [random_crop(image_tensor, 0.01, 0.1) for image_tensor in image_tensors]
mdpo_image_tensors = [random_crop(image_tensor, 0.01, 0.2) for image_tensor in image_tensors]
ri_image_tensors = replace_with_random_images(image_tensors)

# initialize the ViT model
vit_weights = ViT_H_14_Weights.DEFAULT
vit_model = vit_h_14(weights=vit_weights)

# dictionary to store similarity scores
sim_scores = defaultdict(list)

for i in range(len(image_tensors)):
    # get the encoder outputs for the original image
    orig_outputs = model_encoder_outputs(vit_model, vit_weights, image_tensors[i])

    # get the encoder outputs for the corrupted images
    rc_1_10_outputs = model_encoder_outputs(vit_model, vit_weights, rc_1_10_image_tensors[i])
    rr_40_outputs = model_encoder_outputs(vit_model, vit_weights, rr_40_image_tensors[i])
    fd_100_outputs = model_encoder_outputs(vit_model, vit_weights, fd_100_image_tensors[i])
    fd_200_outputs = model_encoder_outputs(vit_model, vit_weights, fd_200_image_tensors[i])
    ri_outputs = model_encoder_outputs(vit_model, vit_weights, ri_image_tensors[i])

    # store the similarity scores for each image
    sim_scores['rc_1_10'].append(similarity(orig_outputs, rc_1_10_outputs))
    sim_scores['rr_40'].append(similarity(orig_outputs, rr_40_outputs))
    sim_scores['fd_100'].append(similarity(orig_outputs, fd_100_outputs))
    sim_scores['fd_200'].append(similarity(orig_outputs, fd_200_outputs))
    sim_scores['ri'].append(similarity(orig_outputs, ri_outputs))

# display the similarity scores
for corr_type, scores in sim_scores.items():
    low_avg, mid_avg, high_avg = avg(scores)
    print(f"Corruption Type: {corr_type:<15} Low-Level: {low_avg:<10.2f} Mid-Level: {mid_avg:<10.2f} High-Level: {high_avg:.2f}")

# print(len(orig_outputs), orig_outputs[0].shape, orig_outputs[-1].shape)
# print(len(rc_outputs), rc_outputs[0].shape, rc_outputs[-1].shape)
# print(len(black_outputs), black_outputs[0].shape, black_outputs[-1].shape)
# print(len(rotated_outputs), rotated_outputs[0].shape, rotated_outputs[-1].shape)

# # convert image tensor back back to PIL Image
# corrupted_images = [to_pil(corrupted_image_tensor.squeeze()) for corrupted_image_tensor in corrupted_image_tensors]

# # no. of columns
# cols = 2
# # no. of rows
# #rows = (len(corrupted_images)*2 // cols) + 1
# rows = len(corrupted_images)

# # figure for the original image and the corrupted images
# fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
# axes = axes.flatten()

# for i in range(len(images)):
#     # plot the original image
#     axes[i*2].imshow(images[i])
#     axes[i*2].set_title("Original Image")
#     axes[i*2].axis('off')

#     # plot the corrupted image
#     axes[(i*2)+1].imshow(corrupted_images[i])
#     axes[(i*2)+1].set_title("Corrupted Image")
#     axes[(i*2)+1].axis('off')

# # # turn off remaining plots
# # for i in range(len(corrupted_images)+1, len(axes)):
# #     axes[i].axis('off')

# # save the images
# plt.savefig(f'./results/neg_img_metric.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()