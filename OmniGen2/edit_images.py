import dotenv
dotenv.load_dotenv(override=True)
import random
from PIL import Image
import torch
import matplotlib.pyplot as plt
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from openai import OpenAI, RateLimitError, InternalServerError
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
import time
import json
import re

# path of model
MODEL_NAME = "OmniGen2/OmniGen2"

# get the OpenAI API key
with open("mDPO/MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# openai client
openai_client = OpenAI(api_key=API_KEY)

# open the text file for logging errors
with open("OmniGen2/log.txt", "w") as f:
    f.write("LOG STARTED\n")
# open the text file in append mode
log_file = open("OmniGen2/log.txt", "a")

# function to load the image editing pipeline
def load_pipeline(accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    pipeline = OmniGen2Pipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            MODEL_NAME,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    # optimization options 
    # see Github repo for explanations
    # pipeline.enable_taylorseer = True
    # pipeline.transformer.enable_teacache = True
    # pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    # scheduler options: euler, dpmsolver++
    scheduler_type = "dpmsolver++"

    if scheduler_type == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    #pipeline.enable_sequential_cpu_offload()
    #pipeline.enable_model_cpu_offload()
    pipeline = pipeline.to(accelerator.device)

    return pipeline

def run(accelerator, pipeline, instruction, negative_prompt, input_image):
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(0)

    result = pipeline(
        prompt=instruction,
        input_images=[input_image],
        width=640, # output image width
        height=640, # output image height
        num_inference_steps=40, # no. of inference steps
        max_sequence_length=1024,
        text_guidance_scale=5.0, # controls how strictly the output adheres to the text prompt
        image_guidance_scale=2.0, # controls how much the final image should resemble the input reference image
        cfg_range=(0.0, 1.0), # range of CFG
        negative_prompt=negative_prompt,
        num_images_per_prompt=1, # no. of images per prompt
        generator=generator,
        output_type="pil",
    )
    return result.images[0]

# function to extract all objects from the response text
def extract_objects(client, response_text):
    # prompt = (
    #     "Rewrite the response text by replacing every object with an object that is visually similar but different, while keeping the sentence structure and overall tone similar.\n"
    #     f"Response Text: {response_text}"
    # )
    # prompt = (
    #     "Rewrite the text by replacing each VISUAL OBJECT CATEGORY with a visually similar but different category, "
    #     "while keeping everything else (structure, tone, relations, counts, adjectives like color/size, locations) the same.\n"
    #     "Rules:\n"
    #     "1) Replace only the object CATEGORY (the noun head). Do not change verbs, prepositions, numbers, or non-visual nouns.\n"
    #     "2) Preserve modifiers (color, size, pose, count) unless the modifier is part of the category name being changed.\n"
    #     "3) If the object is a lexicalized multi-word name (e.g., 'traffic light', 'fire hydrant', 'hot dog', 'baseball bat') or a species term (e.g., 'polar bear'),\n"
    #     "   treat the whole term as the category and replace it with a similar category of the same type.\n"
    #     "4) Do not add or remove objects; keep a 1:1 correspondence with the originals.\n"
    #     "5) Keep the same number of sentences and punctuation. Output only the rewritten text.\n"
    #     f"Text: {response_text}"
    # )
    prompt = f"""
        Task: From the text, extract UNIQUE VISUAL NOUN HEADS (physical object categories) and propose a visually similar but different replacement for each.

        STRICT RULES
        1) Output exactly TWO lines and nothing else.
        2) Line 1 = comma-separated list of unique visual noun heads (lowercase, singular), in order of first mention, max 8 items.
        3) Line 2 = comma-separated replacements, 1:1 aligned with Line 1, same object type (animal↔animal, vehicle↔vehicle, furniture↔furniture), not identical or hypernym/synonym.
        4) Inclusion test (ALL must be true): the item is a concrete, countable, physical object with clear boundaries that can appear in a single photograph (e.g., animal, person, plant, food, tool, appliance, furniture, clothing, vehicle, electronic device, instrument).
        5) Exclude NON-VISUAL categories such as: activities/events/fields (“game”, “campaign”, “hobby”, “sport”, “education”), organizations/services/apps/websites, abstract concepts (“strategy”, “policy”, “idea”), collectives/containers (“group”, “population”, “bunch”, “set”, “pair”), places/scenes (“park”, “street”, “kitchen”), materials/fluids (“water”, “sand”, “air”), weather/time words (“rain”, “morning”), and attributes or actions.
        6) For “X of Y” or collectives (e.g., “bunches of bananas”, “population of polar bears”), KEEP ONLY Y → “banana”, “bear”.
        7) Keep lexicalized multi-word object names as one head if standard (e.g., “traffic light”, “fire hydrant”, “baseball bat”, “cell phone”).

        Text:
        {response_text}
    """
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

# function to build the prompt from the replaced text
def build_omnigen_prompt(text) -> str:
    try:
        # get the lines
        lines = text.split('\n')
        # remove empty strings
        lines = [line for line in lines if line] 

        # objects in the original text
        objs1 = [x.strip() for x in lines[0].split(',')]
        # objects in the rewritten text
        objs2 = [x.strip() for x in lines[1].split(',')]

        # dictionary that maps objects in the original text to objects in the rewritten text
        map1 = {}
        # dictionary that maps objects in the rewritten text to objects in the original text
        map2 = {}

        for i in range(min(len(objs1), len(objs2))):
            o1 = objs1[i]
            o2 = objs2[i]

            # avoid duplicates
            if o1 in map1 or o2 in map2:
                continue

            # avoid replacing same objects
            if o1 == o2:
                continue

            map1[o1] = o2
            map2[o2] = o1

        pairs = []
        for k,v in map1.items():
            pairs.append((k,v))

        result = "; ".join(f"Replace the {o} with {r}" for o, r in pairs) + "."

        return result
    except Exception as error:
        log_file.write(f"PROBLEM WITH REPLACED TEXT: {text}\n")
        log_file.write(f"{str(error)}\n")
        log_file.flush()
        return "Remove all foreground objects and fill in the background."
        #print(f"REPLACED TEXT: {text}")
        #raise

# available data types: 'fp32', 'fp16', 'bf16'
data_type = 'bf16'

# initialize accelerator
accelerator = Accelerator(mixed_precision=data_type if data_type != 'fp32' else 'no')

# set weight dtype
weight_dtype = torch.float32
if data_type == 'fp16':
    weight_dtype = torch.float16
elif data_type == 'bf16':
    weight_dtype = torch.bfloat16

# load image editing pipeline
pipeline = load_pipeline(accelerator, weight_dtype)

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# list of original images
images = []
# list of edited images
edited_images = []
# list of prompts
prompts = []

# # RANDOMLY SAMPLE SOME IMAGES FROM THE DATASET
# #start = time.time()
# # iterate through the data
# for sample in random.sample(data, 8):
#     # chosen response
#     chosen = sample['chosen']
#     # original image
#     image = Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB")

#     # extract all the objects from the chosen response
#     replaced_text = extract_objects(openai_client, chosen)
#     #print(f"CHOSEN: {chosen}\n")
#     print(f"REPLACED TEXT: {replaced_text}")

#     # prompt for image editing model
#     prompt = build_omnigen_prompt(replaced_text)
#     print(f"PROMPT: {prompt}\n")

#     # negative prompt text
#     # tells the model what you don't want to see in the image
#     negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

#     # edit the image
#     edited_image = run(accelerator, pipeline, prompt, negative_prompt, image)

#     images.append(image)
#     edited_images.append(edited_image)
#     prompts.append(prompt)

# # figure for the original image and the edited images
# fig, axes = plt.subplots(len(images), 2, figsize=(10, 12))
# axes = axes.flatten()

# for i in range(len(images)):

#     # plot the original image
#     axes[(i*2)].imshow(images[i])
#     axes[(i*2)].set_title("Original Image")
#     axes[(i*2)].axis('off')

#     # plot the chosen image
#     axes[(i*2)+1].imshow(edited_images[i])
#     axes[(i*2)+1].set_title("Edited Image")
#     axes[(i*2)+1].axis('off')

# # save the images
# plt.savefig(f'mDPO/results/ie_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# USE THE ENTIRE DATASET
# iterate through the data
for sample in data:
    # chosen response
    chosen = sample['chosen']
    # original image name
    image_name = sample['img_path']
    # original image
    image = Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB")

    # extract all the objects from the chosen response
    replaced_text = extract_objects(openai_client, chosen)
    #print(f"CHOSEN: {chosen}\n")
    #print(f"REPLACED TEXT: {replaced_text}")

    # prompt for image editing model
    prompt = build_omnigen_prompt(replaced_text)
    #print(f"PROMPT: {prompt}\n")

    # negative prompt text
    # tells the model what you don't want to see in the image
    negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

    # edit the image
    edited_image = run(accelerator, pipeline, prompt, negative_prompt, image)
    edited_image_save_path = f'mDPO/data/rejected/' + image_name
    edited_image.save(edited_image_save_path)

# end = time.time()
# print(f"TIME TAKEN: {(end-start):.2f} seconds")

log_file.close()