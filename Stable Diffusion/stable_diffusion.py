import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from openai import OpenAI, RateLimitError, InternalServerError
from ultralytics import YOLO
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import time

# set device
device = torch.device("cuda")

# get the OpenAI API key
with open("./mDPO/MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# openai client
openai_client = OpenAI(api_key=API_KEY)

# open the text file for logging errors
with open("./Stable Diffusion/log.txt", "w") as f:
    f.write("LOG STARTED\n")
# open the text file in append mode
log_file = open("Stable Diffusion/log.txt", "a")

# # load the instance segmentation model
# yolo_model = YOLO("YOLO/yolo11x-seg.pt").to(device)

# # object to convert a Pytorch tensor into a PIL image
# to_pil = transforms.ToPILImage()

# function to summarize a response text
def summarize(client, response_text):
    prompt = (
        "Summarize only the visual content described in the response text in one paragraph in less than 50 words. "
        "Do not include any assumptions, causes, intentions, or speculative interpretations. "
        "Only describe what is visually observable in the scene.\n\n"
        f"Response Text: {response_text}"
    )
    response = None
    while response is None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )

            return response.choices[0].message.content
        except RateLimitError as error:
            log_file.write(f"{str(error)}\n")
            log_file.flush()
            time.sleep(60)
            continue
        except InternalServerError as error:
            log_file.write(f"{str(error)}\n")
            log_file.flush()
            time.sleep(600)
            continue
        except Exception as error:
            log_file.write(f"{str(error)}\n")
            log_file.flush()
            raise
            
# function to create a hallucinated version of response text
def hallucinate(client, response_text):
    # prompt = (
    #     "Rewrite the response text by replacing the most important object (the main subject of the scene ) with an object a multimodal LLM is most likely to hallucinate, while keeping the sentence structure and overall tone similar.\n\n"
    #     f"Response Text: {response_text}"
    # )
    # prompt = (
    #     "Rewrite the response text by replacing every object with another similar but different object, while keeping the sentence structure and overall tone similar.\n\n"
    #     f"Response Text: {response_text}"
    # )
    # prompt = (
    #     "Rewrite the response text using the following instructions.\n"
    #     "1. Replace every object and its attribute with another similar but different object.\n"
    #     "2. If there is a number replace it with another similar number.\n"
    #     "3. Keep the original response text structure same by simply changing some words and phrases.\n"
    #     f"Response Text: {response_text}"
    # )
    prompt = (
    "Rewrite the given response text to create a hallucinated version of the scene.\n"
        "Follow these exact rules:\n"
        "1. Only replace concrete physical OBJECTS (nouns referring to visible things like 'dog', 'car', 'tree', 'chair', 'table', etc.) "
        "with other visually similar but different objects (e.g., 'dog' → 'cat', 'car' → 'bus').\n"
        "2. Do NOT change background or environmental details such as weather, time of day, lighting, visibility, or setting. "
        "Leave words like 'rainy', 'sunny', 'visible', 'foggy', 'indoor', 'outdoor', etc. untouched.\n"
        "3. Do NOT change verbs, adjectives describing states ('stuck', 'visible', 'broken'), or relational words.\n"
        "4. Preserve all sentence structure, style, and phrasing. Only the object nouns (and directly attached colors or counts) should change.\n"
        "5. Replace every number with a similar but different one (e.g., 3 → 4, 10 → 9).\n"
        "6. Do NOT introduce new objects or remove existing ones. Just swap each object with a plausible alternative.\n"
        "7. Do NOT add explanations, reasoning, or extra sentences.\n\n"
        f"Response Text: {response_text}"
    )

    response = None
    while response is None:
        try:
            response_obj = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )

            return response_obj.choices[0].message.content

            # responses = response_obj.choices[0].message.content.split('__')
            # return [response.strip() for response in responses]
        except RateLimitError as error:
            log_file.write(f"{str(error)}\n")
            log_file.flush()
            time.sleep(60)
            continue
        except InternalServerError as error:
            log_file.write(f"{str(error)}\n")
            log_file.flush()
            time.sleep(600)
            continue
        except Exception as error:
            log_file.write(f"{str(error)}\n")
            log_file.flush()
            raise

# load the stable diffusion model
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3.5-large", 
#     torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

# open the training data json file
with open('./mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# # indexes of data point
# idxs = [0, 50, 100, 200, 500, 1000]
# for i in idxs:
#     chosen_response = data[i]['chosen']
#     summarized_chosen_response = summarize(openai_client, chosen_response)
#     hallucinated_response = hallucinate(openai_client, summarized_chosen_response)
#     #print(f"QUESTION: {data[i]['prompt']}\n")
#     #print(f"ORIGINAL CHOSEN RESPONSE: {chosen_response}\n")
#     print(f"SUMMARIZED CHOSEN RESPONSE: {summarized_chosen_response}\n")
#     print(f"HALLUCINATED RESPONSES: {hallucinated_response}\n\n")

# list of summarized chosen responses
chosen = []
# list of rejected responses
rejected = []
# list of hallucinated responses
hallucinated = []
# list of image names
image_names = []
# # list of image paths
# image_paths = []
# # list of images
# images = []
# list to save the responses
responses_data = []

# # RANDOMLY SAMPLE SOME IMAGES FROM THE DATASET
# # iterate through the data
# for sample in random.sample(data, 4):
#     # summarize the chosen response
#     chosen_response_summarized = summarize(openai_client, sample['chosen'])
#     # generate the hallucinated response
#     hallucinated_response = hallucinate(openai_client, chosen_response_summarized)

#     chosen.append(chosen_response_summarized)
#     hallucinated.append(hallucinated_response)
#     #rejected.append(summarize(openai_client, sample['rejected']))
#     images.append(Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB"))
#     image_paths.append('mDPO/data/merged_images/' + sample['img_path'])
#     image_names.append(sample['img_path'])

# # figure for the original image and the custom images
# fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 12))
# axes = axes.flatten()

# start = time.time()
# for i in range(len(chosen)):
#     # # use the segmentation and generation model to remove objects
#     # generated_image = remove_objects(yolo_model, pipe, image_paths[i], images[i])
#     # generate the chosen image
#     chosen_image = pipe(
#         chosen[i], # text prompt for generation
#         height=640,
#         width=640,
#         num_inference_steps=50, # no. of denoising steps for finer details
#         guidance_scale=10.0, # strength of prompt adherence 
#     ).images[0]

#     # generate the rejected image
#     rejected_image = pipe(
#         hallucinated[i], # text prompt for generation
#         height=640,
#         width=640,
#         num_inference_steps=50, # no. of denoising steps for finer details
#         guidance_scale=10.0, # strength of prompt adherence 
#     ).images[0]

#     # plot the original image
#     axes[(i*3)].imshow(images[i])
#     axes[(i*3)].set_title("Original Image")
#     axes[(i*3)].axis('off')

#     # plot the chosen image
#     axes[(i*3)+1].imshow(chosen_image)
#     axes[(i*3)+1].set_title("Chosen Image")
#     axes[(i*3)+1].axis('off')

#     # plot the rejected image
#     axes[(i*3)+2].imshow(rejected_image)
#     axes[(i*3)+2].set_title("Rejected Image")
#     axes[(i*3)+2].axis('off')

# end = time.time()
# print(f"TIME TAKEN: {(end-start):.2f} seconds")

# # save the images
# plt.savefig(f'./mDPO/results/sd_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()


# USE THE ENTIRE DATASET
# iterate through the data
for sample in tqdm(data, desc='Generating Responses'):
    # summarize the chosen response
    chosen_response_summarized = summarize(openai_client, sample['chosen'])
    # generate the hallucinated response
    hallucinated_response = hallucinate(openai_client, chosen_response_summarized)

    # get the original data point
    original = sample.copy()
    original['chosen_summarized'] = chosen_response_summarized
    original['hallucinated'] = hallucinated_response
    responses_data.append(original)

    chosen.append(chosen_response_summarized)
    hallucinated.append(hallucinated_response)
    #rejected.append(summarize(openai_client, sample['rejected']))
    #images.append(Image.open('./mDPO/data/merged_images/' + sample['img_path']).convert("RGB"))
    #image_paths.append('./mDPO/data/merged_images/' + sample['img_path'])
    image_names.append(sample['img_path'])

# save the generated responses
with open('./mDPO/data/vlfeedback_llava_10k_gen.json', 'w') as f:
    json.dump(responses_data, f, indent=4)

for i in range(len(chosen)):
    # generate the chosen image
    chosen_image = pipe(
        chosen[i], # text prompt for generation
        height=512,
        width=512,
        num_inference_steps=40, # no. of denoising steps for finder details
        guidance_scale=10.0, # strength of prompt adherence 
    ).images[0]
    # chosen image save path
    chosen_save_path = './mDPO/data/chosen/' + image_names[i]
    chosen_image.save(chosen_save_path)

    # generate the rejected images
    rejected_image = pipe(
        hallucinated[i], # text prompt for generation
        height=512,
        width=512,
        num_inference_steps=40, # no. of denoising steps for finder details
        guidance_scale=10.0, # strength of prompt adherence 
    ).images[0]
    # rejected image save path
    rejected_save_path = './mDPO/data/rejected/' + image_names[i]
    rejected_image.save(rejected_save_path)

    if i % 10 == 0:
        print(f"COMPLETED {i+1}/{len(chosen)}")

log_file.close()