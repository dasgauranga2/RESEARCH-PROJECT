import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2, ToTensor, ToPILImage
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from tqdm import tqdm
from collections import Counter
import pickle
import random
import requests
from io import BytesIO

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# GET THE LOCALIZATION HEADS
# WHICH SHOWS WHERE THE MULTIMODAL LLM IS LOOKING AT THE IMAGE WHEN ANSWERING A QUESTION

# BUNNY PROCESSES THE INPUT IMAGE IN THE FOLLOWING STEPS
# 1. THE INPUT IMAGE IS RESIZED TO 384 x 384
# 2. THE IMAGE IS SPLIT INTO PATCHES OF SIZE 14 x 14 WHICH MEANS THERE ARE 27*27=729 PATCHES
# 3. THE PATCHES ARE GIVEN TO THE ViT TO PRODUCE PATCH EMBEDDINGS (VISION ENCODER)
# 4. THE PATCHES EMBEDDINGS ARE GIVEN TO A 2-LAYER MLP THAT TRANSFORMS EACH PATCH EMBEDDING INTO THE LLM'S INPUT EMBEDDING SPACE (CROSS-MODALITY PROJECTOR)
# NOTE: IN BUNNY THE THE CROSS-MODALITY PROJECTOR RECEIVES AS INPUT 729 TOKENS AND OUTPUTS 729 TOKENS

# variable to decide which checkpoint to use
model_name = 'mdpo_bunny'
#model_name = 'mdpo_bunny_cni'

# load the reference model
model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = f'./checkpoint/{model_name}'

# load the vision tower
model.get_vision_tower().load_model()
vision_tower = model.get_vision_tower()
num_patches  = vision_tower.num_patches
patch_grid_N = int(num_patches ** 0.5)

# print(num_patches)
# print(patch_grid_N)

# determine if LoRA adapter weights should be used
if model_name is None:
    use_lora = False
else:
    use_lora = True

if use_lora:

    model = PeftModel.from_pretrained(
        model,
        checkpoint_path
    )

    model = model.merge_and_unload()

# set model to evaluation mode
model.eval()

#print(model.get_vision_tower().is_loaded)

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# processes a single data point
def prepare_inputs(prompt, tokenizer):
    # dictionary to store the inputs
    batch = {}

    prompt_tokens = {}
    # prompt token ids
    prompt_tokens["input_ids"] = tokenizer_image_token(prompt, tokenizer)
    # the attention mask helps the model differentiate between actual input tokens and padding tokens
    prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

    # get the end-of-sequence token id
    eos_token_id = tokenizer.eos_token_id

    # the steps below adjust the attention mask by setting the position corresponding to exisiting EOS tokens to 0
    # this  ensures that the model does not attend to any tokens that come after the EOS token
    eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
    # attention mask these indices to eos_token_id
    new_attention_mask = [
        0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
    ]
    prompt_tokens["attention_mask"] = new_attention_mask

    for k, toks in {
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    # #print('./data/merged_images/' + img_path)
    # image = Image.open(img_path)
    # # process the image into a tensor
    # image_tensor = crop_image(model.process_images([image], model.config)).to(dtype=model.dtype)
    # batch["image"] = image_tensor

    # the final result will be of this format
    #     batch = {
    #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
    #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
    # }

    return batch

# # user query
# query = "How many traffic lights are there in the image?"
# # image path
# image_path = './data/test/count1.jpg'

# function to load the image from refcoco dataset file name
def load_image(filename):
    if 'train2014' in filename:
        folder = 'train2014'
    elif 'val2014' in filename:
        folder = 'val2014'
    else:
        raise ValueError(f"Unknown COCO split in filename: {filename}")

    # image format
    image_format = filename.split('.')[-1]

    # build image name
    parts = filename.split('_')
    if parts[-1].split('.')[0].isdigit():
        parts = parts[:-1]
    image_name = '_'.join(parts) + '.' + image_format

    # image url
    image_url = f'http://images.cocodataset.org/{folder}/{image_name}'

    request = requests.get(image_url, timeout=20)
    request.raise_for_status()

    return Image.open(BytesIO(request.content)).convert("RGB")

# load the refcoco dataset
with open('./data/refcoco/refs(unc).p', "rb") as f:
    ref_data = pickle.load(f)

# print(len(ref_data))
# print(ref_data[0].keys())

# # Show 3 random samples
# for sample in random.sample(ref_data, 3):
#     print("\n---")
#     print("Image filename:", sample["file_name"])
#     print("Image ID:", sample["image_id"])
#     print("Split:", sample["split"])
#     print("Sentences:", [s["sent"] for s in sample["sentences"]])
#     print("Annotation ID:", sample["ann_id"])

# # open the training data json file
# with open('./data/vlfeedback_llava_10k.json', 'r') as file:
#     json_data = json.load(file)

# ##################################################################################################
# # CALCULATING ATTENTION SUMS
# ##################################################################################################

# dictionary to store attention sum values for each attention head
attn_sum_vals = {}

# function to calculate the attention sum of a head 
# corresponding to the image tokens
def attention_sums(img_attn_scores):
    return img_attn_scores.sum(dim=1)

# function to calculate the attention sum of each head 
# corresponding to the image tokens
@torch.no_grad()
def all_attention_sums(input_text, img_tensor, tokenizer, model):
    # get the inputs tokens for the model
    data = prepare_inputs(input_text, tokenizer)

    # get the model outputs
    outputs = model(
        input_ids=torch.tensor(data["prompt_input_ids"], dtype=torch.long, device=device).unsqueeze(0),  # add batch dimension
        attention_mask=torch.tensor(data["prompt_attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        images=img_tensor.to(device),
        labels=None,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True
    )

    # number of layers 
    num_layers = len(outputs.attentions)

    # index position where the first image token is located
    image_token_pos = data["prompt_input_ids"].index(-200)

    for layer in range(2, num_layers):
        # we pass the model the input tokens
        # we extract the attention scores corresponding to the last input token
        # for a particular layer
        ans_img_attn_scores = outputs.attentions[layer].squeeze()[:, -1, image_token_pos:image_token_pos+num_patches] # (heads, 729)

        # calculate the attention sum for each head
        ans_img_attn_sums = attention_sums(ans_img_attn_scores) # (heads,)

        for i, ans_img_attn_sum in enumerate(ans_img_attn_sums):

            # index of each attention head
            key = (layer, i)
            if key not in attn_sum_vals:
                attn_sum_vals[key] = [ans_img_attn_sum.item()]
            else:
                attn_sum_vals[key].append(ans_img_attn_sum.item())

for data in tqdm(random.sample(ref_data, 1000), desc='Calculating attention sums or entropies'):
    # get the image from the file name
    image = load_image(data['file_name'])
    # get the referring expression
    ref_sent = random.choice(data['sentences'])['sent']
    # input text for bunny
    input_text = f"<image>\n{ref_sent}"

    #orig_image = corrupt_image(Image.open(image_path))
    # process the image into a tensor
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
    # # resize the image
    # orig_image_resized = orig_image.convert('RGB').resize((384, 384))
    # # crop the upper-left portion of the image
    # crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
    # orig_image_cropped = orig_image_resized.crop(crop_box)

    # calculate attention scores for all heads for the data point
    all_attention_sums(input_text, image_tensor, tokenizer, model)

# 1. PLOT THE AVERAGE ATTENTION SUMS FOR EACH HEAD
# calculate average attention sum for each head
for head in attn_sum_vals:
    average = sum(attn_sum_vals[head])/len(attn_sum_vals[head])
    attn_sum_vals[head] = average

# attention sum values in sorted order
attn_sum_sorted = list(attn_sum_vals.values())
attn_sum_sorted.sort()

# plot the attention sum values
plt.figure(figsize=(8, 4))
plt.plot(range(len(attn_sum_sorted)), attn_sum_sorted)
plt.title(f'{model_name} Attention Sums Curve')
plt.xlabel('Head')
plt.ylabel('Average Attention Sum')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./results/avg_attn_sums.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()

# ##################################################################################################
# # CALCULATING SPATIAL ENTROPY
# ##################################################################################################


# # list to store which heads with lowest entropies were selected
# sel_entropy = []

# # function to calculate the spatial entropy
# # from a spatial attention map
# def spatial_entropy(attn_map):
#     # calculate the mean threshold
#     mean_val = np.mean(attn_map)
#     # compute the binarized attention map
#     binarized_attn_map = np.where(attn_map > mean_val, 1, 0)

#     # keep track of cells visited for dfs
#     visited = np.full_like(binarized_attn_map, False, dtype=bool)
#     # lengths of connected components
#     connected_components = {}
#     # index of each connected component
#     cci = 0
#     # total number of connected component cells
#     total_cc = 0

#     # function to perform dfs to find number of connected components
#     def dfs(i, j, idx):
#         if i < 0 or i >= len(visited) or j < 0 or j >= len(visited[0]):
#             return
#         if visited[i][j]: # if cell is visited
#             return
#         if binarized_attn_map[i][j] == 0: # if cell is 0
#             return

#         visited[i][j] = True
#         nonlocal total_cc
#         total_cc += 1
#         if idx not in connected_components:
#             connected_components[idx] = 1
#         else:
#             connected_components[idx] += 1

#         dfs(i, j+1, idx)
#         dfs(i, j-1, idx)
#         dfs(i+1, j, idx)
#         dfs(i-1, j, idx)
#         dfs(i+1, j+1, idx)
#         dfs(i-1, j+1, idx)
#         dfs(i+1, j-1, idx)
#         dfs(i-1, j-1, idx)
    
#     # perform dfs to identify size of each connected component
#     for i in range(len(visited)):
#         for j in range(len(visited[0])):
#             if binarized_attn_map[i][j] == 1 and not visited[i][j]:
#                 dfs(i, j, cci)
#                 cci += 1
    
#     # calculate the spatial entropy
#     entropy = 0
#     for ccs in connected_components.values():
#         # probability of connected components
#         pcn = ccs / total_cc

#         entropy = entropy - (pcn * np.log(pcn))
    
#     return entropy

# # function to calculate the spatial attention maps for each head
# def spatial_attention_map(prompt, img_tensor, tokenizer, model):
#     # list to store attention maps for each head
#     all_sam = []

#     # prompt text with <image> token
#     #prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"

#     # get the inputs for the model
#     data = prepare_inputs(prompt, tokenizer)

#     # get the model outputs
#     outputs = model(
#         input_ids=torch.tensor(data["prompt_input_ids"], dtype=torch.long, device=device).unsqueeze(0),  # add batch dimension
#         attention_mask=torch.tensor(data["prompt_attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
#         images=img_tensor.to(device),
#         labels=None,
#         use_cache=False,
#         output_attentions=True,
#         output_hidden_states=False,
#         return_dict=True
#     )

#     # get the attention scores from a particular layer
#     #att_scores = outputs.attentions[layer_index].squeeze() # (heads, 781, 781)

#     # number of layers 
#     num_layers = len(outputs.attentions)

#     # index position where the first image token is located
#     image_token_pos = data["prompt_input_ids"].index(-200)

#     for layer in range(2, num_layers):
#         # when predicting the first answer token
#         # extract the attention scores to the image tokens
#         # for a particular layer
#         ans_img_attn_scores = outputs.attentions[layer].squeeze()[:, -1, image_token_pos:image_token_pos+729] # (heads, 729)

#         # calculate the attention sum for each head
#         ans_img_attn_sums = attention_sums(ans_img_attn_scores) # (heads,)

#         for i, (ans_img_attn_score, ans_img_attn_sum) in enumerate(zip(ans_img_attn_scores, ans_img_attn_sums)):
#             # reshape to 27 x 27 (no. of patches) to get the spatial attention map
#             spatial_attn_map = ans_img_attn_score.reshape(27, 27).cpu().detach().numpy()

#             # calculate the spatial entropy of each attention map
#             entropy = spatial_entropy(spatial_attn_map)

#             # index of each attention head
#             key = (layer, i)
#             if key not in attn_sum_vals:
#                 attn_sum_vals[key] = [ans_img_attn_sum.item()]
#             else:
#                 attn_sum_vals[key].append(ans_img_attn_sum.item())

#             all_sam.append((key, ans_img_attn_sum, entropy))

#     # threshold to filter out the attention maps with low average attention sum
#     # first comment out the function code below then plot the average attention sum
#     # and then find the point of maximum curvature
#     aas_threshold = 0.6
#     # keep only those attention maps which have high attention map scores
#     all_sam = [attn_map for attn_map in all_sam if attn_map[1] > aas_threshold]

#     # sort according to entropy
#     all_sam.sort(key=lambda x:x[2])
#     # add the attention heads with the lowest entropies
#     global sel_entropy
#     sel_entropy = sel_entropy + all_sam[:10]

# for data in tqdm(random.sample(json_data, 500), desc='Calculating attention sums or entropies'):
#     # path of image
#     image_path = './data/merged_images/' + data['img_path']
#     # prompt text
#     prompt_text = data['prompt']

#     # reopen the original image
#     orig_image = Image.open(image_path)
#     #orig_image = corrupt_image(Image.open(image_path))
#     # process the image into a tensor
#     image_tensor = model.process_images([orig_image], model.config).to(dtype=model.dtype)
#     # resize the image
#     orig_image_resized = orig_image.convert('RGB').resize((384, 384))
#     # crop the upper-left portion of the image
#     crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
#     orig_image_cropped = orig_image_resized.crop(crop_box)

#     # generate spatial attention map of mdpo
#     spatial_attention_map(prompt_text, image_tensor, tokenizer, model)

# # 1. PLOT THE AVERAGE ATTENTION SUMS FOR EACH HEAD
# # calculate average attention sum for each head
# for head in attn_sum_vals:
#     average = sum(attn_sum_vals[head])/len(attn_sum_vals[head])
#     attn_sum_vals[head] = average

# # attention sum values in sorted order
# attn_sum_sorted = list(attn_sum_vals.values())
# attn_sum_sorted.sort()

# # plot the attention sum values
# plt.figure(figsize=(8, 4))
# plt.plot(range(len(attn_sum_sorted)), attn_sum_sorted)
# plt.xlabel('Head')
# plt.ylabel('Average Attention Sum')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f'./results/avg_attn_sums.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# # 2. PLOT THE SELECTION FREQUENCY FOR EACH HEAD
# # count frequency of each head
# head_counts = Counter([head_info[0] for head_info in sel_entropy])
# # get the most frequent heads
# freq_head_counts = head_counts.most_common(10)
# # plot the selection frequency
# plt.figure(figsize=(8, 4))
# plt.bar([f"L{fhc[0][0]}-H{fhc[0][1]}" for fhc in freq_head_counts], [fhc[1] for fhc in freq_head_counts])
# plt.xticks(rotation=45)
# plt.xlabel("Attention Head (Layer,Head)")
# plt.ylabel("Selection Frequency")
# plt.title(f"{model_name} Selected Frequencies")
# plt.tight_layout()
# plt.savefig(f'./results/sel_freq.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()