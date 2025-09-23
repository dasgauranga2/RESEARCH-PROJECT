import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
import json
from tqdm import tqdm
import os
import glob
import re

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# load the base model
model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = './mDPO/checkpoint/mdpo_bunny'
#checkpoint_path = './mDPO/checkpoint/mdpo_bunny_sd'
#checkpoint_path = './mDPO/checkpoint/mdpo_bunny_sd_all'
#heckpoint_path = './mDPO/checkpoint/mdpo_bunny_sd_all_attr'
# determine if LoRA adapter weights should be used
use_lora = True
# checkpoint name
checkpoint_name = checkpoint_path.split('/')[-1]

if use_lora:
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path
    )

    model = model.merge_and_unload()

# set model to evaluation mode
model.eval()

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# open the evaluation data json file
with open('./BAT/bat_data.json', 'r') as file:
    data = json.load(file)

# function to check for all variants of an image file name
# if original image file name is test.jpg
# it will search for test_snow.jpg, test_grass.jpg, ...
def find_img_variants(img_name):
    base = os.path.basename(img_name)
    root, ext = os.path.splitext(base)
    pattern = os.path.join('./BAT/eval_images/', f"{root}_*{ext}")
    paths = sorted(glob.glob(pattern))
    if not paths:
        single = os.path.join('./BAT/eval_images/', base)
        if os.path.exists(single):
            paths = [single]
    return paths

# function to check if the model's generated response is incorrect
def score_response(text):
    lower_text = text.strip().lower()

    if 'no' in lower_text:
        return 1
    else:
        return 0

# dictionary to store results for each background image
results = {}

for sample in tqdm(data, desc='Evaluating responses'):
#for sample in data[:4]:
    # original image name
    image_name = sample['img_path']
    # most confident object in the image
    most_conf_obj = sample['most_conf_class']

    # get paths of all variants of original image that was saved
    all_image_paths = find_img_variants(image_name)

    for image_path in all_image_paths:
        # query
        query = f'Answer the question using only YES or NO. Is there {most_conf_obj} in the image?'
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

        # load the image
        image = Image.open(image_path).convert("RGB")
        #image.save(f'./BAT/test_{total}.jpg')
        image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
        #print(image_tensor.shape)
        # load the background image type
        bg_type = image_path.split('_')[-1]
        if '.' in bg_type:
            bg_type = bg_type.split('.')[0]
        if bg_type not in results:
            results[bg_type] = {'incorrect': 0, 'total': 0}

        # generate the model outputs
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]

        # get the generated text
        response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

        # update the results
        results[bg_type]['total'] += 1
        results[bg_type]['incorrect'] += score_response(response)

print(f"Results for {checkpoint_name}")
for bg_type, bg_type_results in results.items():
    print(f"{bg_type} Error Rate: {(bg_type_results['incorrect']/bg_type_results['total'])*100:.2f}%")