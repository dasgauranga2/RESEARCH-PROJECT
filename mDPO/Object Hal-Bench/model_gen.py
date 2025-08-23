import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
import json
import os, io, requests
from tqdm import tqdm

# split from COCO 14 dataset
# that will be used to evaluate the model
# each data point has 'image_id' and 'question'
eval_data = []
with open("Object Hal-Bench/obj_halbench_300_with_image.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            eval_data.append(json.loads(line))

# print(data[10]['image_id'])
# print(data[10]['question'])

# from both the annotations file
# build a dictionary
# of image id and image file name
id2file = {}
for split in ("val", "train"):
    p = os.path.join("Object Hal-Bench/annotations", f"instances_{split}2014.json")
    with open(p, "r", encoding="utf-8") as f:
        ann_data = json.load(f)
    for img in ann_data["images"]:
        id2file[int(img["id"])] = (split, img["file_name"])

# function to load an image from image id
def load_coco_image(image_id: int, annot_dir="annotations", cache_dir="coco_cache") -> Image.Image:

    try:
        split, fname = id2file[int(image_id)]
    except KeyError:
        raise KeyError(f"image_id {image_id} not found in COCO2014 annotations at {annot_dir}")

    split_dir = "val2014" if split == "val" else "train2014"
    os.makedirs(os.path.join(cache_dir, split_dir), exist_ok=True)
    local_path = os.path.join(cache_dir, split_dir, fname)

    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGB")

    url = f"http://images.cocodataset.org/{split_dir}/{fname}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    with open(local_path, "wb") as f: f.write(r.content)
    return Image.open(io.BytesIO(r.content)).convert("RGB")

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

# check point name
checkpoint_name = 'mdpo_bunny'
# path of saved checkpoint
checkpoint_path = f'./checkpoint/{checkpoint_name}'
# # determine if LoRA adapter weights should be used
# use_lora = True

# if use_lora:
#     model = PeftModel.from_pretrained(
#         model,
#         checkpoint_path
#     )

#     model = model.merge_and_unload()

# set model to evaluation mode
model.eval()

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# list of responses
responses = []

for sample in tqdm(eval_data, desc='Generating Answers'):
    # get the image id
    image_id  = sample['image_id']
    # get the question
    query = sample['question']

    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

    # load the image
    image = load_coco_image(image_id, 'Object Hal-Bench/annotations/')
    # load the image tensor
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

    # generate the model outputs
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    # get the model response
    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    responses.append(response)

# open file where results will be saved
with open(f'Object Hal-Bench/responses/{checkpoint_name}_results.jsonl', 'w', encoding='utf-8') as file:
    for i in range(len(responses)):
        response_sample = {
            'image_id': int(eval_data[i]['image_id']),
            'question': eval_data[i]['question'],
            'answer': responses[i]
        }

        file.write(json.dumps(response_sample, ensure_ascii=False) + '\n')