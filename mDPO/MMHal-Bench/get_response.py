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

# function to load an image locally
def load_image(image_src):
    # extract the image name
    image_name = image_src.split('/')[-1]

    # path of image
    image_path = './MMHal-Bench/images/' + image_name

    image = Image.open(image_path).convert('RGB')
    
    return image

# build the model using your own code
model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# which model to use
model_name = 'mdpo_bunny'
# path of saved checkpoint
checkpoint_path = f'./checkpoint/{model_name}'
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

json_data = json.load(open('./MMHal-Bench/response_template.json', 'r'))

# function to generate a response
def generate_response(image, question, eval_model):
    # prompt text
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

    # load the image
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

    # generate the model outputs
    output_ids = eval_model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=150,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    # get the generated text
    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    return response

#print(json_data[0])
for idx, line in tqdm(enumerate(json_data), desc='Generating responses'):
    image_src = line['image_src']
    question = line['question']

    # image = load_image(image_src)
    # response = generate_response(image, question, model)
    try:
        image = load_image(image_src)
        response = generate_response(image, question, model)
    except:
        response = "ERROR: Image could not be loaded"
        print(f"Image {idx} could not be loaded")
    
    # print(idx, response)
    line['model_answer'] = response

with open(f'./MMHal-Bench/responses/{model_name}_results.json', 'w') as f:
    json.dump(json_data, f, indent=4)