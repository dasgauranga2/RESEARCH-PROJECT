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
from sklearn.neighbors import KNeighborsRegressor

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# CALCULATE THE ATTENTION MAP
# WHICH SHOWS WHERE THE MULTIMODAL LLM IS LOOKING AT THE IMAGE WHEN ANSWERING A QUESTION

# BUNNY PROCESSES THE INPUT IMAGE IN THE FOLLOWING STEPS
# 1. THE INPUT IMAGE IS RESIZED TO 384 x 384
# 2. THE IMAGE IS SPLIT INTO PATCHES OF SIZE 14 x 14 WHICH MEANS THERE ARE 27*27=729 PATCHES
# 3. THE PATCHES ARE GIVEN TO THE ViT TO PRODUCE PATCH EMBEDDINGS (VISION ENCODER)
# 4. THE PATCHES EMBEDDINGS ARE GIVEN TO A 2-LAYER MLP THAT TRANSFORMS EACH PATCH EMBEDDING INTO THE LLM'S INPUT EMBEDDING SPACE (CROSS-MODALITY PROJECTOR)
# NOTE: IN BUNNY THE THE CROSS-MODALITY PROJECTOR RECEIVES AS INPUT 729 TOKENS AND OUTPUTS 729 TOKENS

# variable to decide which checkpoints to use
model_names = [
    'mdpo_bunny',
    'mdpo_bunny_cni',
    'mdpo_bunny_dci',
]
#model_name = 'mdpo_bunny'
#model_name = 'mdpo_bunny_dci'
#model_name = 'mdpo_bunny_cni'

# load the reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# function to load a model from the checkpoint name
def load_model(name):
    # load the reference model
    base_model = AutoModelForCausalLM.from_pretrained(
        'BAAI/Bunny-v1_0-3B',
        torch_dtype=torch.float16, # float32 for cpu
        device_map='auto',
        trust_remote_code=True)
    
    # path of saved checkpoint
    checkpoint_path = f'./checkpoint/{name}'

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path
    )

    model = model.merge_and_unload()

    # load the vision tower
    model.get_vision_tower().load_model()

    # set model to evaluation mode
    model.eval()

    return model

# # path of saved checkpoint
# checkpoint_path = f'./checkpoint/{model_name}'

# vision_tower = model.get_vision_tower()
# num_patches  = vision_tower.num_patches
# patch_grid_N = int(num_patches ** 0.5)

# print(num_patches)
# print(patch_grid_N)

# # determine if LoRA adapter weights should be used
# if model_name is None:
#     use_lora = False
# else:
#     use_lora = True

# if use_lora:

#     model = PeftModel.from_pretrained(
#         model,
#         checkpoint_path
#     )

#     model = model.merge_and_unload()

# # load the vision tower
# model.get_vision_tower().load_model()

# # set model to evaluation mode
# model.eval()

#print(model.get_vision_tower().is_loaded)

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    './checkpoint/mdpo_bunny',
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

# user query
query = "How many bicycles are there in the image?"
# image path
image_path = './data/test/count2.jpg'

# # user query
# query = "How many zebras are there in the image?"
# # image path
# image_path = './data/test/count3.jpg'

# # user query
# query = "How many players are there in the image?"
# # image path
# image_path = './data/test/count4.jpg'

# # user query
# query = "How many chairs are there in the image?"
# # image path
# image_path = './data/test/count5.jpg'

# # user query
# query = "How many chairs are there in the image?"
# # image path
# image_path = './data/test/count6.jpg'

# # user query
# query = "How many teddy bears are there in the image?"
# # image path
# image_path = './data/test/count8.jpg'

# # user query
# query = "How many traffic signs in the image?"
# # image path
# image_path = './data/test/count9.jpg'

# function to generate the model's response
def generate_response(question, img_tensor, tokenizer, model):
    # prompt text
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

    # # load the image
    # #image = Image.open('./AMBER/data/image/' + data['image']).convert('RGB')
    # image = Image.open(path).convert('RGB')
    # image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

    # generate the model outputs
    output_ids = model.generate(
        input_ids,
        images=img_tensor,
        max_new_tokens=150,
        use_cache=True,
        repetition_penalty=1.0 # increase this to avoid chattering
    )[0]

    # get the generated text
    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    return response

# function to calculate the attention sum of each head 
# corresponding to the image tokens
def attention_sums(img_attn_scores):
    return img_attn_scores.sum(dim=1)

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

# function to calculate the spatial attention maps for each head
def spatial_attention_map(question, img_tensor, tokenizer, model):
    # list to store attention maps for each head
    all_sam = []

    # prompt text with <image> token
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"

    # get the inputs for the model
    data = prepare_inputs(prompt, tokenizer)

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

    # # index position where the first image token is located
    # image_token_pos = data["prompt_input_ids"].index(-200)

    # for lh in localization_heads:
    #     # when predicting the first answer token
    #     # extract the attention scores to the image tokens
    #     # for a particular layer
    #     ans_img_attn_scores = outputs.attentions[lh[0]].squeeze()[:, -1, image_token_pos:image_token_pos+729] # (heads, 729)

    #     # exrtract the attention scores for a particular head
    #     ans_img_attn_score = ans_img_attn_scores[lh[1]]

    #     # reshape to 27 x 27 (no. of patches) to get the spatial attention map
    #     spatial_attn_map = ans_img_attn_score.reshape(27, 27).cpu().detach().numpy()

    #     all_sam.append((spatial_attn_map, lh[0], lh[1]))
    
    # return all_sam

    # number of layers
    num_layers = len(outputs.attentions)

    # index position where the first image token is located
    image_token_pos = data["prompt_input_ids"].index(-200)

    for layer in range(num_layers):
        # when predicting the first answer token
        # extract the attention scores to the image tokens
        # for a particular layer
        ans_img_attn_scores = outputs.attentions[layer].squeeze()[:, -1, image_token_pos:image_token_pos+729] # (heads, 729)

        # calculate the attention sum for each head
        ans_img_attn_sums = attention_sums(ans_img_attn_scores) # (heads,)

        for i, (ans_img_attn_score, ans_img_attn_sum) in enumerate(zip(ans_img_attn_scores, ans_img_attn_sums)):
            # reshape to 27 x 27 (no. of patches) to get the spatial attention map
            spatial_attn_map = ans_img_attn_score.reshape(27, 27).cpu().detach().numpy()

            # # calculate the spatial entropy of each attention map
            # entropy = spatial_entropy(spatial_attn_map)

            all_sam.append((spatial_attn_map, ans_img_attn_sum, layer, i))
    
    all_sam.sort(key=lambda x:x[1], reverse=True)

    # # keep only those attention maps which have high attention map scores
    # all_sam = [attn_map for attn_map in all_sam if attn_map[1] > 0.2]

    # # sort the attention maps by spatial entropy
    # all_sam.sort(key=lambda x:x[2])
    # # take the top-10
    # all_sam = all_sam[:10]

    return all_sam[:20]

# reopen the original image
orig_image = Image.open(image_path)
#orig_image = corrupt_image(Image.open(image_path))
# process the image into a tensor
image_tensor = ref_model.process_images([orig_image], ref_model.config).to(dtype=ref_model.dtype)
# resize the image
orig_image_resized = orig_image.convert('RGB').resize((384, 384))
# crop the upper-left portion of the image
crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
orig_image_cropped = orig_image_resized.crop(crop_box)

# # mDPO localization heads
# mdpo_lh = [(20,22), (22,13), (31,22), (14,20), (18,8), (25,18), (31,0), (21,14), (27,1), (22,11)]

for i, model_name in enumerate(model_names):
    # load the model from the checkpoint name
    model = load_model(model_name)

    # generate spatial attention map
    spt_attn_maps = spatial_attention_map(query, image_tensor, tokenizer, model)

    # generate the model's response
    response = generate_response(query, image_tensor, tokenizer, model)

    cols = 5
    rows = (len(spt_attn_maps)+cols) // cols

    # figure for the original image and the attention maps
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    # plot the original image
    axes[0].imshow(orig_image_cropped)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # plot the attention maps
    for j in range(len(spt_attn_maps)):
        axes[j+1].imshow(spt_attn_maps[j][0], cmap='viridis')
        #axes[i+1].set_title(f"Layer: {spt_attn_maps[j][1]}\nHead: {spt_attn_maps[j][2]}")
        axes[j+1].set_title(f"Layer: {spt_attn_maps[j][2]}\nHead: {spt_attn_maps[j][3]}")
        axes[j+1].axis('off')

    # turn off remaining plots
    for j in range(len(spt_attn_maps)+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Query: {query}\nModel Name: {model_name}\nAnswer: {response}", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'./results/spatial_attn_map{i+1}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()