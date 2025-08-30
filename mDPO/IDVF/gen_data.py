import torch
from diffusers import StableDiffusion3Pipeline
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from openai import OpenAI, RateLimitError, InternalServerError
import json

# set device
device = torch.device("cuda")

# get the OpenAI API key
with open("MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# openai client
openai_client = OpenAI(api_key=API_KEY)

# # load the instance segmentation model
# yolo_model = YOLO("YOLO/yolo11x-seg.pt").to(device)

# # object to convert a Pytorch tensor into a PIL image
# to_pil = transforms.ToPILImage()

# function to summarize a response text
def summarize(client, response_text):
    prompt = (
        "Summarize only the visual content described in the response text in one paragraph in less than 60 words. "
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
        except Exception as error:
            raise
            
# function to create a hallucinated version of response text
def hallucinate(client, response_text):
    prompt = (
        "Pick the the most important object (the main subject of the scene) in the response text\n"
        "Rewrite the response text by replacing that object with another visually similar object, while keeping the sentence structure and overall tone similar.\n"
        "Separate the important object and rewritten response text using only '__'.\n\n"
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
        except Exception as error:
            raise

# load the stable diffusion model
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers", 
#     torch_dtype=torch.float16)
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3.5-medium", 
#     torch_dtype=torch.bfloat16)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", 
    torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

# open the training data json file
with open('./data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# list to save the data
save_data = []

# # indexes of data point
# idxs = [[i for i, x in enumerate(data) if x["img_path"]=="llava-detail-945.jpg"], 
#         [i for i, x in enumerate(data) if x["img_path"]=="llava-detail-1945.jpg"],
#         [i for i, x in enumerate(data) if x["img_path"]=="llava-detail-1845.jpg"],
#         [i for i, x in enumerate(data) if x["img_path"]=="llava-reasoning-3784.jpg"],
#         [i for i, x in enumerate(data) if x["img_path"]=="llava-detail-157.jpg"]]
# idxs = [idx for idx_list in idxs for idx in idx_list]
for i in [0,10,50,80,100]:
    original = data[i].copy()
    # get the chosen response
    chosen_response = original['chosen']
    # summarize the chosen response
    summarized_chosen_response = summarize(openai_client, chosen_response)
    # create a hallucinated version by replacing one of the object
    hallucinated_output = hallucinate(openai_client, summarized_chosen_response).split('__')

    # get the object that was replaced
    hallucinated_obj = hallucinated_output[0].strip()
    # hallucinated response with the object replaced
    hallucinated_response = hallucinated_output[1].strip()

    # generate the hallucinated image
    hall_image = pipe(
        hallucinated_response, # text prompt for generation
        height=640,
        width=640,
        num_inference_steps=40, # no. of denoising steps for finer details
        guidance_scale=10.0, # strength of prompt adherence 
    ).images[0]
    
    # print(f"SUMMARY CHOSEN RESPONSE: {summarized_chosen_response}\n")
    # print(f"HALLUCINATED OBJECT: {hallucinated_obj}")
    # print(f"HALLUCINATED RESPONSE: {hallucinated_response}\n\n")

    # save the generated data in a json file
    original['chosen_summarized'] = summarized_chosen_response
    original['hallucinated_obj'] = hallucinated_obj
    original['hallucinated_response'] = hallucinated_response
    save_data.append(original)

    # image name
    img_name = original['img_path']
    # save the hallucinated image
    hall_image.save('./IDVF/hall_images/' + img_name)

# save the generated responses
with open('./IDVF/idvf_data.json', 'w') as f:
    json.dump(save_data, f, indent=4)

# # list of summarized chosen responses
# chosen = []
# # list of rejected responses
# rejected = []
# # list of hallucinated responses
# hallucinated = []
# # list of image names
# image_names = []
# # list of image paths
# image_paths = []
# # list of images
# images = []
# # list to save the responses
# responses_data = []

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
#         num_inference_steps=40, # no. of denoising steps for finer details
#         guidance_scale=10.0, # strength of prompt adherence 
#     ).images[0]

#     # generate the rejected image
#     rejected_image = pipe(
#         hallucinated[i], # text prompt for generation
#         height=640,
#         width=640,
#         num_inference_steps=40, # no. of denoising steps for finer details
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
# plt.savefig(f'mDPO/results/sd_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()


# # USE THE ENTIRE DATASET
# # iterate through the data
# for sample in tqdm(data, desc='Generating Responses'):
#     # summarize the chosen response
#     chosen_response_summarized = summarize(openai_client, sample['chosen'])
#     # generate the hallucinated response
#     hallucinated_response = hallucinate(openai_client, chosen_response_summarized)

#     # get the original data point
#     original = sample.copy()
#     original['chosen_summarized'] = chosen_response_summarized
#     original['hallucinated'] = hallucinated_response
#     responses_data.append(original)

#     chosen.append(chosen_response_summarized)
#     hallucinated.append(hallucinated_response)
#     #rejected.append(summarize(openai_client, sample['rejected']))
#     images.append(Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB"))
#     image_paths.append('mDPO/data/merged_images/' + sample['img_path'])
#     image_names.append(sample['img_path'])

# # save the generated responses
# with open('mDPO/data/vlfeedback_llava_10k_gen.json', 'w') as f:
#     json.dump(responses_data, f, indent=4)

# for i in range(len(chosen)):
#     # # use the segmentation and generation model to remove objects
#     # generated_image = remove_objects(yolo_model, pipe, image_paths[i], images[i])
#     # generate the chosen image
#     chosen_image = pipe(
#         chosen[i], # text prompt for generation
#         height=512,
#         width=512,
#         num_inference_steps=40, # no. of denoising steps for finder details
#         guidance_scale=10.0, # strength of prompt adherence 
#     ).images[0]
#     # chosen image save path
#     chosen_save_path = 'mDPO/data/chosen/' + image_names[i]
#     chosen_image.save(chosen_save_path)

#     # generate the rejected images
#     rejected_image = pipe(
#         hallucinated[i], # text prompt for generation
#         height=512,
#         width=512,
#         num_inference_steps=40, # no. of denoising steps for finder details
#         guidance_scale=10.0, # strength of prompt adherence 
#     ).images[0]
#     # rejected image save path
#     rejected_save_path = f'mDPO/data/rejected/' + image_names[i]
#     rejected_image.save(rejected_save_path)

#     if i % 10 == 0:
#         print(f"COMPLETED {i+1}/{len(chosen)}")

# log_file.close()