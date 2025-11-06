from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from peft import PeftModel
import requests
from PIL import Image
import transformers
from io import BytesIO
from tqdm import tqdm

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# load the json data with questions
hall_data = json.load(open('./HallusionBench/HallusionBench.json', 'r'))
# load the sample result json data
result_sample = json.load(open('./HallusionBench/HallusionBench_result_sample.json', 'r'))
# print(hall_data[0])
# print(result_sample[0])
# print('\n\n')
# print(hall_data[220])
# print(result_sample[220])

# function to load an image locally
def load_image(image_src):
    idx = None
    # get the path of image by removing unnecessary parts
    if 'VS' in image_src:
        idx = image_src.find('VS')
    if 'VD' in image_src:
        idx = image_src.find('VD')

    image_path = image_src[idx:]

    # actual path of image
    image_path = './HallusionBench/hallusion_bench/' + image_path

    image = Image.open(image_path).convert('RGB')
    
    return image

# which model to use
model_name = 'mdpo_bunny'
# path of saved checkpoint
checkpoint_path = f'./checkpoint/{model_name}'

# load the base model
if '8b' in checkpoint_path:
    model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-Llama-3-8B-V',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        'BAAI/Bunny-v1_0-3B',
        torch_dtype=torch.float16, # float32 for cpu
        device_map='auto',
        trust_remote_code=True)

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

# function to generate a response
def generate_response(image, question, eval_model):
    if image:
        # prompt text
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"
        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

        # load the image
        image_tensor = eval_model.process_images([image], eval_model.config).to(dtype=eval_model.dtype, device=device)

        # generate the model outputs
        output_ids = eval_model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=150,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]
    else:
        # prompt text
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {question} ASSISTANT:"
        text_chunks = tokenizer(text).input_ids

        input_ids = torch.tensor(text_chunks, dtype=torch.long).unsqueeze(0).to(device)

        # generate the model outputs
        output_ids = eval_model.generate(
            input_ids,
            max_new_tokens=150,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]

    # get the generated text
    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    return response

# list to store the json results
results = []

#print(json_data[0])
for data in tqdm(hall_data, desc="Generating responses"):
    # question
    question = data['question']

    # check if visual input is needed
    if data['visual_input'] in {'1', '2'}:
        # load the image
        image = load_image(data['filename'])
    else:
        image = None

    # get the model response
    response = generate_response(image, question, model)

    # save response
    original = data.copy()
    original['model_prediction'] = response
    results.append(original)

with open(f'./HallusionBench/responses/{model_name}_results.json', 'w') as f:
    json.dump(results, f, indent=4)