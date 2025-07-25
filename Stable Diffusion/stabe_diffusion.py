import torch
from diffusers import StableDiffusion3Pipeline
import json

# load the stable diffusion model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

print(data[0])

# generate the image
image = pipe(
    data[0]['chosen'], # text prompt for generation
    negative_prompt=data[0]['rejected'], # negative prompt for undesirable features
    num_inference_steps=28, # no. of denoising steps for finder details
    guidance_scale=7.0, # strength of prompt adherence 
).images[0]

image.save('Stable Diffusion/test.png')