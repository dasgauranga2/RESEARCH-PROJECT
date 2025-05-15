import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
import json
from tqdm import tqdm
import requests
from io import BytesIO

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# open the file with queries and image paths
with open('./MMHal-Bench/response_template.json') as file:
    queries = json.load(file)

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
checkpoint_path = './checkpoint/mdpo_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

if use_lora:
    # apply LoRA adapter weights
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path
    )

    model = model.merge_and_unload()

    print("LoRA adapter weights applied")

# set model to evaluation mode
model.eval()

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# # function to load an image
# def load_image(image_file):
#     if image_file.startswith('http') or image_file.startswith('https'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')
#     return image

# print(queries)
# print(len(queries))

with torch.no_grad():
    for data in tqdm(queries, desc="Generating responses on MMHal-Bench"):

        # prompt text
        prompt = data['question']
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

        # load the image
        #image = Image.open('./AMBER/data/image/' + data['image']).convert('RGB')
        image_src = data['image_src']
        if image_src.startswith('http') or image_src.startswith('https'):
            response = requests.get(image_src)
            try:
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except:
                continue
        else:
            image = Image.open('./MMHal-Bench/images/' + image_src).convert('RGB')
        image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

        # generate the model outputs
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=150,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]

        # get the generated text
        response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

        data['model_answer'] = response

# save the result
with open("./MMHal-Bench/mdpo_results.json", "w") as file:
    json.dump(queries, file, indent=4)