import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes
import os
import json
import random
import matplotlib.pyplot as plt
from openai import OpenAI, RateLimitError, InternalServerError
from ultralytics import YOLO
from diffusers import StableDiffusionGLIGENPipeline
import time

#os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# get the OpenAI API key
with open("mDPO/MMHal-Bench/api.txt", "r") as f:
    API_KEY = f.read().strip()

# openai client
openai_client = OpenAI(api_key=API_KEY)

# open the text file for logging errors
with open("GLIGEN/log.txt", "w") as f:
    f.write("LOG STARTED\n")
# open the text file in append mode
log_file = open("GLIGEN/log.txt", "a")

# load the object detection model
model = YOLO("GLIGEN/yolov8x-oiv7.pt")  # e.g., replace with yolov8n-oiv7.pt if you have that pretrained

# load the image editing pipeline
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-inpainting-text-box",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
#pipe.enable_vae_tiling() 

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# function to normalize a bounding box
def normalize_box(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, x2 / w, y2 / h]

# function to create prompt for GLIGEN from the chosen response
def gligen_prompt(client, response_text):
    prompt = (
        "From the response text write a single-sentence scene caption (<=25 words) that ONLY describes the global background, "
        "environment, lighting, time of day, weather, camera/style, and mood for the desired final image. "
        "Do NOT mention specific foreground objects, categories, counts, or actions.\n"
        f"Response text: {response_text}"
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

# function to suggest replacements for each object
def replace(client, object_text):
    prompt = (
        "Given an object name, propose few alternative objects that are visually similar but clearly different. "
        "Select them ONLY from the LVIS dataset categories. "
        "Prioritize nouns with similar shape, size, and geometry so they can plausibly occupy the same bounding box. "
        "Avoid synonyms of the original object, brand names, or abstract concepts. "
        "Return ONLY a single comma-separated list of distinct, singular LVIS category names. "
        f"Object: {object_text}"
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

# list to store original images
orig_images = []
# list to store images with bounding boxes
bb_images = []
# list to store edited images
ed_images = []
# # path to your font file
# font_path = "GLIGEN/roboto.ttf"

# RANDOMLY SAMPLE SOME IMAGES FROM THE DATASET
# iterate through the data
for sample in random.sample(data, 8):
    # load the image
    image = Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB")
    # get the chosen response
    chosen = sample['chosen']

    # get the prompt for GLIGEN
    prompt = gligen_prompt(openai_client, chosen)

    # print(f"CHOSEN: {chosen}")
    # print(f"PROMPT: {prompt}\n")
    
    # get width and height of image
    width, height = image.size

    # run inference
    results = model.predict(source=image, imgsz=640)[0]

    # list of bounding box locations
    boxes = []
    # list of bounding box labels
    labels = []
    # list of object categories
    class_names = []
    # list of objects that we will replace with
    class_repls = []

    # iterate through each detected result
    for box, class_idx, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):

        # get coordinates of bounding box
        x1, y1, x2, y2 = box.tolist()

        # object category
        class_name = model.names[int(class_idx.item())].lower()

        # if human is detected suggest a statue
        if class_name in {"person", "man", "woman", "boy", "girl"}:
            repl_obj = random.choice(["statue", "sculpture"])
        elif class_name == "clothing": # skip clothing
            continue
        elif class_name.startswith("human"): # skip human parts
            continue
        else:
            # get suggestions for the detected object
            rep_suggs = replace(openai_client, class_name)

            # convert to list
            rep_suggs_list = rep_suggs.split(',')

            # randomly select one object
            repl_obj = random.choice(rep_suggs_list).strip().lower()


        # print(f"CLASS: {class_name}")
        # print(f"REPLACEMENT: {repl_obj}\n")

        # confidence score
        conf = float(score.item())

        class_names.append(class_name)
        class_repls.append(repl_obj)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        labels.append(f"{class_name} {conf:.2f}")

    if boxes:
        # draw the bounding box on the image
        bb_image = draw_bounding_boxes(
            image=pil_to_tensor(image.copy()),                      # uint8 CHW
            boxes=torch.tensor(boxes, dtype=torch.int64),                    # int pixel coords
            labels=labels,                    # optional text per box
            colors="red",                          # or list of colors
            width=3,                               # line thickness
            font_size=20,                           # needs a TrueType font available
            #font=font_path
        )
        bb_image = to_pil_image(bb_image)

        # normalize the bounding box locations
        gligen_boxes = [normalize_box(box, width, height) for box in boxes]

        # phrases for each corresponding bounding box to replace with
        gligen_phrases = class_repls

        # run image editing
        edited_image = pipe(
            prompt=prompt,
            gligen_phrases=gligen_phrases,
            gligen_boxes=gligen_boxes,
            gligen_inpaint_image=image,            # edit your original image
            gligen_scheduled_sampling_beta=0.3,    # see note below
            num_inference_steps=40,
            guidance_scale=7.5,
            output_type="pil"
        ).images[0]
        print(f"DETECTED OBJECTS: {class_names}")
        print(f"REPLACED OBJECTS: {class_repls}\n")

        orig_images.append(image)
        bb_images.append(bb_image)
        ed_images.append(edited_image)
    else:
        print("NO OBJECTS DETECTED")

# figure for the original image and the custom images
fig, axes = plt.subplots(len(orig_images), 3, figsize=(10, 12))
axes = axes.flatten()

for i in range(len(orig_images)):
    # plot the original image
    axes[(i*3)].imshow(orig_images[i])
    axes[(i*3)].set_title("Original Image")
    axes[(i*3)].axis('off')

    # plot the image with bounding box
    axes[(i*3)+1].imshow(bb_images[i])
    axes[(i*3)+1].set_title("Bounding Box Image")
    axes[(i*3)+1].axis('off')

    # plot the edited image
    axes[(i*3)+2].imshow(ed_images[i])
    axes[(i*3)+2].set_title("Edited Image")
    axes[(i*3)+2].axis('off')

# save the images
plt.savefig(f'mDPO/results/ie_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()