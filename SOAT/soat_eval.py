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
#checkpoint_path = './mDPO/checkpoint/mdpo_bunny'
checkpoint_path = './mDPO/checkpoint/mdpo_bunny_sd_all'
# determine if LoRA adapter weights should be used
use_lora = True

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
with open('./SOAT/soat_data.json', 'r') as file:
    data = json.load(file)

# function to evaluate the model's response and give a score 0 or 1
def score_response(response_text):
    lower_text = response_text.strip().lower()

    if 'yes' in lower_text:
        return 1
    else:
        return 0

incorrect = 0
total = 0

for sample in tqdm(data, desc='Evaluating responses'):
    # visually similar object image name
    vso_image_name = sample['search_img_path']
    # most important object name
    mio = sample['most_important_object']

    # query
    query = f'Answer the question using only YES or NO. Is there {mio} in the image?'
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

    # load the image
    image = Image.open('./SOAT/eval_images/' + vso_image_name)
    #image.save(f'./BAT/test_{total}.jpg')
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
    #print(image_tensor.shape)

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
    #print(response)

    total += 1
    incorrect += score_response(response)

#print(f"Correct: {correct}")
#print(f"Total: {total}")
print(f"Error Rate: {(incorrect/total)*100:.2f}%")