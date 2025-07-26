import torch
from diffusers import StableDiffusion3Pipeline
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# get the OpenAI API key
with open("mDPO/MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# openai client
openai_client = OpenAI(api_key=API_KEY)

# function to summarize a response text
def summarize(client, response_text):
    prompt = (
        "Using the response text, give a short summary of the image that is described by the text.\n\n"
        f"Response Text: {response_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content

# load the stable diffusion model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# print(f"QUESTION: {data[550]['prompt']}\n")
# print(f"ORIGINAL TEXT: {data[550]['chosen']}\n")
# print(f"SUMMARY TEXT: {summarize(openai_client, data[550]['chosen'])}")

# list of chosen responses
chosen = []
# list of rejected responses
rejected = []
# list of image names
image_names = []
# list of images
images = []

# RANDOMLY SAMPLE SOME IMAGES FROM THE DATASET
# iterate through the data
for sample in random.sample(data, 6):
    chosen.append(summarize(openai_client, sample['chosen']))
    rejected.append(summarize(openai_client, sample['rejected']))
    images.append(Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB"))
    image_names.append(sample['img_path'])

# figure for the original image and the custom images
fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 20))
axes = axes.flatten()

for i in range(len(chosen)):

    # generate the image
    custom_image = pipe(
        chosen[i], # text prompt for generation
        num_inference_steps=28, # no. of denoising steps for finder details
        guidance_scale=7.0, # strength of prompt adherence 
    ).images[0]

    # plot the original image
    axes[(i*2)].imshow(images[i])
    axes[(i*2)].set_title("Original Image")
    axes[(i*2)].axis('off')

    # plot the custom image
    axes[(i*2)+1].imshow(custom_image)
    axes[(i*2)+1].set_title("Generated Image")
    axes[(i*2)+1].axis('off')

# save the images
plt.savefig(f'mDPO/results/sd_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()

# # GENERATE IMAGES FROM THE ENTIRE DATASET
# for sample in tqdm(data, desc='Saving image paths and names'):
#     image_names.append(sample['img_path'])
#     chosen.append(sample['chosen'])
#     rejected.append(sample['rejected'])

# for i in tqdm(range(len(data)), desc='Generating images for the entire dataset'):
#     # generate the image
#     custom_image = pipe(
#         chosen[i], # text prompt for generation
#         negative_prompt=rejected[i], # negative prompt for undesirable features
#         num_inference_steps=28, # no. of denoising steps for finder details
#         guidance_scale=7.0, # strength of prompt adherence 
#     ).images[0]

#     save_path = 'mDPO/data/chosen/' + image_names[i]

#     custom_image.save(save_path)