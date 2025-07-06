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

# CALCULATE THE RELATIVE ATTENTION MAP
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

# vision_tower = model.get_vision_tower()
# num_patches  = vision_tower.num_patches
# patch_grid_N = int(num_patches ** 0.5)

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

# load the vision tower
model.get_vision_tower().load_model()

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

# user query
query = "How many traffic lights are there in the image?"
# image path
image_path = './data/test/count1.jpg'

# # user query
# query = "How many bicycles are there in the image?"
# # image path
# image_path = './data/test/count2.jpg'

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

# function to calculate the attention sum of each head 
# corresponding to the image tokens
def attention_sums(img_attn_scores):
    return img_attn_scores.sum(dim=1)

# function to calculate the spatial attention maps for each head
def spatial_attention_map(question, img_tensor, tokenizer, model, layer_index = -1):
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

    # get the attention scores from a particular layer
    att_scores = outputs.attentions[layer_index].squeeze() # (heads, 781, 781)

    # index position where the first image token is located
    image_token_pos = data["prompt_input_ids"].index(-200)

    # when predicting the first answer token
    # extract the attention scores to the image tokens
    ans_img_attn_scores = att_scores[:, -1, image_token_pos:image_token_pos+729] # (heads, 729)

    # calculate the attention sum for each head
    ans_img_attn_sums = attention_sums(ans_img_attn_scores) # (heads,)


    for ans_img_attn_score, ans_img_attn_sum in zip(ans_img_attn_scores, ans_img_attn_sums):
        # reshape to 27 x 27 (no. of patches) to get the spatial attention map
        spatial_attn_map = ans_img_attn_score.reshape(27, 27).cpu().detach().numpy()

        all_sam.append((spatial_attn_map, ans_img_attn_sum))

    # sort by attention sum
    all_sam.sort(key=lambda x:x[1], reverse=True)

    return all_sam

# transformer layer index
layer_index = -1

# reopen the original image
orig_image = Image.open(image_path)
#orig_image = corrupt_image(Image.open(image_path))
# process the image into a tensor
image_tensor = model.process_images([orig_image], model.config).to(dtype=model.dtype)
# resize the image
orig_image_resized = orig_image.convert('RGB').resize((384, 384))
# crop the upper-left portion of the image
crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
orig_image_cropped = orig_image_resized.crop(crop_box)

# generate spatial attention map of mdpo
spt_attn_maps = spatial_attention_map(query, image_tensor, tokenizer, model, layer_index)

cols = 4
rows = (len(spt_attn_maps)+cols) // cols

# figure for the original image and the attention maps
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
axes = axes.flatten()

# plot the original image
axes[0].imshow(orig_image_cropped)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the attention maps
for i in range(len(spt_attn_maps)):
    axes[i+1].imshow(spt_attn_maps[i][0], cmap='viridis')
    axes[i+1].set_title(f"Head Attention Sum: {spt_attn_maps[i][1]:.2f}")
    axes[i+1].axis('off')

# turn off remaining plots
for i in range(len(spt_attn_maps)+1, len(axes)):
    axes[i].axis('off')

plt.suptitle(f"Query: {query}\nModel Name: {model_name}", fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'./results/spatial_attn_map.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()