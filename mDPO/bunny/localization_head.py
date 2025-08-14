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
import math

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

# # dictionary to store attention sum values for each attention head
# attn_sum_vals = {}

# # function to calculate the attention sum of a head 
# # corresponding to the image tokens
# def attention_sums(img_attn_scores):
#     return img_attn_scores.sum(dim=1)

# # function to calculate the attention sum of each head 
# # corresponding to the image tokens
# @torch.no_grad()
# def all_attention_sums(input_text, img_tensor, tokenizer, model):
#     # get the inputs tokens for the model
#     data = prepare_inputs(input_text, tokenizer)

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

#     # number of layers 
#     num_layers = len(outputs.attentions)

#     # index position where the first image token is located
#     image_token_pos = data["prompt_input_ids"].index(-200)

#     for layer in range(2, num_layers):
#         # we pass the model the input tokens
#         # we extract the attention scores corresponding to the last input token
#         # for a particular layer
#         ans_img_attn_scores = outputs.attentions[layer].squeeze()[:, -1, image_token_pos:image_token_pos+num_patches] # (heads, 729)

#         # calculate the attention sum for each head
#         ans_img_attn_sums = attention_sums(ans_img_attn_scores) # (heads,)

#         for i, ans_img_attn_sum in enumerate(ans_img_attn_sums):

#             # index of each attention head
#             key = (layer, i)
#             if key not in attn_sum_vals:
#                 attn_sum_vals[key] = [ans_img_attn_sum.item()]
#             else:
#                 attn_sum_vals[key].append(ans_img_attn_sum.item())

# for data in tqdm(random.sample(ref_data, 1000), desc='Calculating attention sums'):
#     # get the image from the file name
#     image = load_image(data['file_name'])
#     # get the referring expression
#     ref_sent = random.choice(data['sentences'])['sent']
#     # input text for bunny
#     input_text = f"<image>\n{ref_sent}"

#     #orig_image = corrupt_image(Image.open(image_path))
#     # process the image into a tensor
#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
#     # # resize the image
#     # orig_image_resized = orig_image.convert('RGB').resize((384, 384))
#     # # crop the upper-left portion of the image
#     # crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
#     # orig_image_cropped = orig_image_resized.crop(crop_box)

#     # calculate attention scores for all heads for the data point
#     all_attention_sums(input_text, image_tensor, tokenizer, model)

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
# plt.title(f'{model_name} Attention Sums Curve')
# plt.xlabel('Head')
# plt.ylabel('Average Attention Sum')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f'./results/avg_attn_sums.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# ##################################################################################################
# # CALCULATING SPATIAL ENTROPY
# ##################################################################################################

# # attention sum threshold calculated from above
# attn_sum_threshold = 0.7
# # select only those heads with attention sum score above threshold
# allowed_heads = {head_idx for head_idx, attn_sum in attn_sum_vals.items() if attn_sum >= attn_sum_threshold}

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

# # function to calculate the spatial entropy of each head 
# @torch.no_grad()
# def all_spatial_entropies(input_text, img_tensor, tokenizer, model):
#     # list to store all attention maps
#     # with attention with attention sum scores above the threshold
#     attn_maps = []

#     # get the inputs for the model
#     data = prepare_inputs(input_text, tokenizer)

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

#     # number of layers 
#     num_layers = len(outputs.attentions)

#     # index position where the first image token is located
#     image_token_pos = data["prompt_input_ids"].index(-200)

#     for layer in range(2, num_layers):
#         # we pass the model the input tokens
#         # we extract the attention scores corresponding to the last input token
#         # for a particular layer
#         ans_img_attn_scores = outputs.attentions[layer].squeeze()[:, -1, image_token_pos:image_token_pos+num_patches] # (heads, 729)

#         for i, ans_img_attn_score in enumerate(ans_img_attn_scores):
#             # index of each attention head
#             key = (layer, i)
            
#             if key not in allowed_heads:
#                 continue

#             # reshape to 27 x 27 (no. of patches) to get the spatial attention map
#             spatial_attn_map = ans_img_attn_score.reshape(patch_grid_N, patch_grid_N).cpu().detach().numpy()

#             # calculate the spatial entropy of each attention map
#             entropy = spatial_entropy(spatial_attn_map)

#             attn_maps.append((key, entropy))

#     # sort according to entropy
#     attn_maps.sort(key=lambda x:x[1])

#     # add the attention heads with the lowest entropies
#     global sel_entropy
#     sel_entropy = sel_entropy + attn_maps[:10]

# for data in tqdm(random.sample(ref_data, 1000), desc='Calculating spatial entropies'):
#     # get the image from the file name
#     image = load_image(data['file_name'])
#     # get the referring expression
#     ref_sent = random.choice(data['sentences'])['sent']
#     # input text for bunny
#     input_text = f"<image>\n{ref_sent}"

#     # # reopen the original image
#     # orig_image = Image.open(image_path)
#     #orig_image = corrupt_image(Image.open(image_path))
#     # process the image into a tensor
#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
#     # # resize the image
#     # orig_image_resized = orig_image.convert('RGB').resize((384, 384))
#     # # crop the upper-left portion of the image
#     # crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
#     # orig_image_cropped = orig_image_resized.crop(crop_box)

#     # calculate spatial entropies for all heads for the data point
#     all_spatial_entropies(input_text, image_tensor, tokenizer, model)

# # 2. PLOT THE SELECTION FREQUENCY FOR EACH HEAD
# # count frequency of each head
# head_counts = Counter([head_info[0] for head_info in sel_entropy])
# # get the most frequent heads
# freq_head_counts = head_counts.most_common(10)
# # plot the selection frequency
# plt.figure(figsize=(8, 4))
# plt.bar([f"L{fhc[0][0]}-H{fhc[0][1]}" for fhc in freq_head_counts], [fhc[1] for fhc in freq_head_counts])
# plt.xticks(rotation=45)
# plt.title(f"{model_name} Histogram of Spatial Entropy")
# plt.xlabel("Attention Head (Layer,Head)")
# plt.ylabel("Selection Frequency")
# plt.title(f"{model_name} Selected Frequencies")
# plt.tight_layout()
# plt.savefig(f'./results/sel_freq.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()


##################################################################################################
# DISPLAY LOCALIZATION HEADS ATTENTION MAPS
##################################################################################################

LOCALIZATION_HEADS = [(31, 22), (22, 13), (29, 18), (19, 3), (29, 2), (31, 28), (25, 18), (12, 25), (31, 21), (13, 7)]

# function to get the attention maps corresponding to the localization heads 
@torch.no_grad()
def attention_maps(input_text, img_tensor, tokenizer, model):
    # list to store all attention maps
    attn_maps = []

    # get the inputs for the model
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

        for i, ans_img_attn_score in enumerate(ans_img_attn_scores):
            # index of each attention head
            key = (layer, i)
            
            if key not in LOCALIZATION_HEADS:
                continue

            # reshape to 27 x 27 (no. of patches) to get the spatial attention map
            spatial_attn_map = ans_img_attn_score.reshape(patch_grid_N, patch_grid_N).cpu().detach().numpy()

            attn_maps.append((key, spatial_attn_map))

    return attn_maps

# get kernel to perform Gaussian smoothing
def _gaussian3x3():
    k = np.array([[1., 2., 1.],
                  [2., 4., 2.],
                  [1., 2., 1.]], dtype=np.float32)
    k /= k.sum()
    return k

# perform 2d convolution
def _conv2d_same(img, kernel):
    assert img.ndim == 2 and kernel.shape == (3,3)
    H, W = img.shape
    pad = 1
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    out = np.empty_like(img, dtype=np.float32)
    for y in range(H):
        ys = y
        for x in range(W):
            xs = x
            window = padded[ys:ys+3, xs:xs+3]
            out[y, x] = np.sum(window * kernel, dtype=np.float32)
    return out

# function to combine the attention maps of the localization heads
def combine_attention_maps(attn_maps,
                           smooth: bool = True,
                           normalize: bool = True,
                           return_mask: bool = False,
                           return_bbox: bool = False,
                           threshold: str = "mean"):

    # Stack -> (N, P, P)
    stacked = np.stack(attn_maps, axis=0).astype(np.float32)

    # 1) smooth per map (paper)
    if smooth:
        k = _gaussian3x3()
        for i in range(stacked.shape[0]):
            stacked[i] = _conv2d_same(stacked[i], k)

    # 2) element-wise SUM (paper)
    combined = np.sum(stacked, axis=0)

    # 3) (optional) normalize to [0,1] for visualization
    if normalize:
        mn, mx = float(combined.min()), float(combined.max())
        if mx > mn:
            combined = (combined - mn) / (mx - mn)

    # Optionally produce mask and bbox (paper: mean threshold + largest rectangle)
    if not (return_mask or return_bbox):
        return combined

    if threshold.lower() != "mean":
        raise ValueError("Only 'mean' threshold is supported to match the paper.")

    thr = float(combined.mean())
    mask = (combined > thr).astype(np.uint8)  # (P,P) in {0,1}

    bbox = None
    if return_bbox:
        ys, xs = np.where(mask > 0)
        if ys.size > 0:
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            bbox = (x1, y1, x2, y2)

    if return_mask and return_bbox:
        return combined, mask, bbox
    elif return_mask:
        return combined, mask
    elif return_bbox:
        return combined, bbox
    else:
        return combined

# get the image from the file name
orig_image = Image.open('./data/test3.png')
# get the referring expression
ref_sent = 'orange cat'
# input text for bunny
input_text = f"<image>\n{ref_sent}"

# process the image into a tensor
image_tensor = model.process_images([orig_image], model.config).to(dtype=model.dtype)

# resize the image
orig_image_resized = orig_image.convert('RGB').resize((384, 384))
# crop the upper-left portion of the image
crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
orig_image_cropped = orig_image_resized.crop(crop_box)

# get the attention maps of the localization heads
result = attention_maps(input_text, image_tensor, tokenizer, model)

# get the combined attention map
comb_attn_map = combine_attention_maps([attn_map[1] for attn_map in result])

total = len(result)+2
cols = 5
rows = math.ceil(total/cols)

# figure for the original image and the attention maps
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
axes = axes.flatten()

# plot the original image
axes[0].imshow(orig_image_cropped)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the combined attention map image
axes[1].imshow(comb_attn_map)
axes[1].set_title("Combined Attention Map")
axes[1].axis('off')

# plot the attention maps
for j in range(len(result)):
    axes[j+2].imshow(result[j][1], cmap='viridis')
    #axes[i+2].set_title(f"Layer: {spt_attn_maps[j][1]}\nHead: {spt_attn_maps[j][2]}")
    axes[j+2].set_title(f"Layer: {result[j][0][0]}\nHead: {result[j][0][1]}")
    axes[j+2].axis('off')

# turn off remaining plots
for j in range(len(result)+2, len(axes)):
    axes[j].axis('off')

plt.suptitle(f"Input Text: {ref_sent}\nModel Name: {model_name}", fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'./results/spatial_attn_map.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()