import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.utils import load_image, make_image_grid
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from ultralytics import YOLO
from torchvision import transforms
import numpy as np

# set device
device = torch.device("cuda")

# get the OpenAI API key
with open("mDPO/MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# openai client
openai_client = OpenAI(api_key=API_KEY)

# load the instance segmentation model
yolo_model = YOLO("YOLO/yolo11x-seg.pt").to(device)

# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

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

# function to create a hallucinate version of response text
def hallucinate(client, response_text):
    prompt = (
        "Rewrite the response text by replacing some objects with different ones, while keeping the sentence structure and overall tone similar.\n\n"
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

# function to remove objects from an image
def remove_objects(seg_model, sd_pipe, image_path, image):
    # run image segmentation on image
    results = seg_model(image_path)

    if results[0].masks is None or results[0].masks.data is None:
        print("NO OBJECT DETECTED")
        return None
    else:
        # get all the segmentation masks
        masks = results[0].masks.data.cpu().numpy() # shape: (N, H, W)

        if masks.shape[0] > 1:
            # indices of masks that will be selected
            # randomly select any two masks
            selected_indices = random.sample(range(len(masks)), min(2, len(masks)))

            # combine the selected masks using logical OR
            selected_mask = np.logical_or(masks[selected_indices[0]], masks[selected_indices[1]])
        else:
            selected_mask = masks[0]

        # convert the mask to an image
        mask_image = Image.fromarray((selected_mask.astype("uint8") * 255), mode="L")

        # remove the segmented object
        removed_image = sd_pipe(prompt='Erase the object covered by the mask and make the surrounding area blend naturally', 
                             image=image, 
                             mask_image=mask_image).images[0]

        # open the original image
        return mask_image, removed_image

# load the stable diffusion model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16)
#pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# # index of data point
# i = 480
# chosen_response = data[i]['chosen']
# summarized_chosen_response = summarize(openai_client, chosen_response)
# hallucinated_response = hallucinate(openai_client, summarized_chosen_response)
# print(f"QUESTION: {data[i]['prompt']}\n")
# print(f"ORIGINAL CHOSEN RESPONSE: {chosen_response}\n")
# print(f"SUMMARY CHOSEN RESPONSE: {summarized_chosen_response}")
# print(f"HALLUCINATED RESPONSE: {hallucinated_response}")

# list of chosen responses
chosen = []
# list of rejected responses
rejected = []
# list of image names
image_names = []
# list of image paths
image_paths = []
# list of images
images = []

# RANDOMLY SAMPLE SOME IMAGES FROM THE DATASET
# iterate through the data
for sample in random.sample(data, 6):
    chosen.append(summarize(openai_client, sample['chosen']))
    #rejected.append(summarize(openai_client, sample['rejected']))
    images.append(Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB"))
    image_paths.append('mDPO/data/merged_images/' + sample['img_path'])
    image_names.append(sample['img_path'])

# figure for the original image and the custom images
fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 20))
axes = axes.flatten()

for i in range(len(chosen)):
    # # use the segmentation and generation model to remove objects
    # generated_image = remove_objects(yolo_model, pipe, image_paths[i], images[i])
    # generate the chosen image
    chosen_image = pipe(
        chosen[i], # text prompt for generation
        num_inference_steps=28, # no. of denoising steps for finder details
        guidance_scale=7.0, # strength of prompt adherence 
    ).images[0]

    # generate the hallucinated response
    hallucinated = hallucinate(openai_client, chosen[i])

    # generate the rejected image
    rejected_image = pipe(
        hallucinated, # text prompt for generation
        num_inference_steps=28, # no. of denoising steps for finder details
        guidance_scale=7.0, # strength of prompt adherence 
    ).images[0]

    # plot the original image
    axes[(i*3)].imshow(images[i])
    axes[(i*3)].set_title("Original Image")
    axes[(i*3)].axis('off')

    # plot the chosen image
    axes[(i*3)+1].imshow(chosen_image)
    axes[(i*3)+1].set_title("Chosen Image")
    axes[(i*3)+1].axis('off')

    # plot the rejected image
    axes[(i*3)+2].imshow(rejected_image)
    axes[(i*3)+2].set_title("Hallucinated Image")
    axes[(i*3)+2].axis('off')

# save the images
plt.savefig(f'mDPO/results/sd_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()