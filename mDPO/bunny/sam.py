import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# CALCULATE THE SPATIAL ATTENTION MAP
# WHICH SHOWS WHERE THE LLM IS LOOKING AT THE IMAGE WHEN ANSWERING A QUESTION

# load the reference model
# BUNNY PROCESSES THE INPUT IMAGE IN THE FOLLOWING STEPS
# 1. THE INPUT IMAGE IS RESIZED TO 384 x 384
# 2. THE IMAGE IS SPLIT INTO PATCHES OF SIZE 14 x 14 WHICH MEANS THERE ARE 27*27=729 PATCHES
# 3. THE PATCHES ARE GIVEN TO THE ViT TO PRODUCE PATCH EMBEDDINGS (VISION ENCODER)
# 4. THE PATCHES EMBEDDINGS ARE GIVEN TO A 2-LAYER MLP THAT TRANSFORMS EACH PATCH EMBEDDING INTO THE LLM'S INPUT EMBEDDING SPACE (CROSS-MODALITY PROJECTOR)
# NOTE: IN BUNNY THE THE CROSS-MODALITY PROJECTOR RECEIVES AS INPUT 729 TOKENS AND OUTPUTS 576 TOKENS
reference_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# # load the mdpo model
# mdpo_model = AutoModelForCausalLM.from_pretrained(
#     'BAAI/Bunny-v1_0-3B',
#     torch_dtype=torch.float16, # float32 for cpu
#     device_map='auto',
#     trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = './checkpoint/mdpo_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

# if use_lora:
#     mdpo_model = PeftModel.from_pretrained(
#         mdpo_model,
#         checkpoint_path
#     )

#     mdpo_model = mdpo_model.merge_and_unload()

# load the vision towers
reference_model.get_vision_tower().load_model()
#mdpo_model.get_vision_tower().load_model()

# set model to evaluation mode
reference_model.eval()
#mdpo_model.eval()

#print(mdpo_model.get_vision_tower().is_loaded)
print(reference_model.get_vision_tower().is_loaded)

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# # function to crop an image
# def crop_image(image):
#     resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
#     image = resize_cropper(image.squeeze(0)).unsqueeze(0)
#     return image

# processes a single data point
def prepare_inputs(prompt, response, img_path, tokenizer, model):
    # dictionary to store the inputs
    batch = {}

    # tokenize the response texts
    response_tokens = tokenizer(response, add_special_tokens=False)

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

    # do the same for the response
    eos_indices_response = [i for i, x in enumerate(response_tokens["input_ids"]) if x == eos_token_id]
    new_attention_mask_c = [
        0 if i in eos_indices_response else p for i, p in enumerate(response_tokens["attention_mask"])
    ]
    response_tokens["attention_mask"] = new_attention_mask_c

    # add EOS token to end of responses
    response_tokens["input_ids"].append(tokenizer.eos_token_id)
    response_tokens["attention_mask"].append(1)

    # concatenate the prompt and response tokens
    response_sequence_tokens = {k: prompt_tokens[k] + response_tokens[k] for k in response_tokens}
    # lables are created from the above tokens such that
    # tokens corresponding to prompt tokens are masked 
    response_sequence_tokens["labels"] = response_sequence_tokens["input_ids"][:]
    response_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )

    for k, toks in {
        "response": response_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    #print('./data/merged_images/' + img_path)
    image = Image.open(img_path)
    # process the image into a tensor
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
    batch["image"] = image_tensor

    # the final result will be of this format
    #     batch = {
    #     "response_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + response
    #     "response_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
    #     "response_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
    #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
    #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
    #     "image": <tensor representation of the image>  # image tensor of shape (1,C,W,H)
    # }

    return batch

# prompt text with <image> token
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nDescribe the image. ASSISTANT:"
# response text
response = "There is an orange cat in a jungle."
# image path
image_path = './data/test3.png'

# get the inputs for the model
data = prepare_inputs(prompt, response, image_path, tokenizer, reference_model)
#print(data)

# get the model outputs
outputs = reference_model(
    input_ids=torch.tensor(data["response_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
    attention_mask=torch.tensor(data["response_attention_mask"], dtype=torch.long).unsqueeze(0),
    images=data["image"],
    labels=None,
    use_cache=False,
    output_attentions=True,
    output_hidden_states=False,
    return_dict=True
)

#logits = outputs.logits.squeeze()

# get the attention scores from the last layer
att_scores = outputs.attentions[-1].squeeze() # (heads, 781, 781)

# index position of first answer token
first_answer_token_idx = len(data["prompt_input_ids"]) + 728

# index position where the first image token is located
image_token_pos = data["prompt_input_ids"].index(-200)

# extract attention scores for the first answer token to the image tokens
ans_img_attn_scores = att_scores[:, first_answer_token_idx, image_token_pos:image_token_pos+729] # (heads, 729)

# average over attention heads
avg_attn = ans_img_attn_scores.mean(dim=0)  # (729,)

# reshape to 27 x 27 (no. of patches) to get the spatial attention map
spatial_attn_map = avg_attn.reshape(27, 27).cpu().detach().numpy()

# reopen the image and then resize it
orig_image_resized = Image.open(image_path).convert('RGB').resize((384, 384))

# figure with two columns for the original image and the spatial attention map
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# plot the original image
axes[0].imshow(orig_image_resized)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the spatial attention map
axes[1].imshow(spatial_attn_map, cmap='hot')
axes[1].set_title("Spatial Attention Map")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('./results/spatial_attn_map.png', bbox_inches='tight', pad_inches=0)
plt.close()