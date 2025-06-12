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
# WHICH SHOWS WHERE THE LLM IS LOOKING AT THE IMAGE WHEN ANSWERING A QUESTION

# BUNNY PROCESSES THE INPUT IMAGE IN THE FOLLOWING STEPS
# 1. THE INPUT IMAGE IS RESIZED TO 384 x 384
# 2. THE IMAGE IS SPLIT INTO PATCHES OF SIZE 14 x 14 WHICH MEANS THERE ARE 27*27=729 PATCHES
# 3. THE PATCHES ARE GIVEN TO THE ViT TO PRODUCE PATCH EMBEDDINGS (VISION ENCODER)
# 4. THE PATCHES EMBEDDINGS ARE GIVEN TO A 2-LAYER MLP THAT TRANSFORMS EACH PATCH EMBEDDING INTO THE LLM'S INPUT EMBEDDING SPACE (CROSS-MODALITY PROJECTOR)
# NOTE: IN BUNNY THE THE CROSS-MODALITY PROJECTOR RECEIVES AS INPUT 729 TOKENS AND OUTPUTS 729 TOKENS

# variable to decide which model to use
# 'Bunny', 'DPO', 'mDPO'
which_model = 'mDPO'

# load the reference model
model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# vision_tower = model.get_vision_tower()
# num_patches  = vision_tower.num_patches
# patch_grid_N = int(num_patches ** 0.5)

# print(num_patches)
# print(patch_grid_N)

# path of saved checkpoint
if which_model == 'DPO':
    checkpoint_path = './checkpoint/dpo_bunny'
else:
    checkpoint_path = './checkpoint/mdpo_bunny'

# determine if LoRA adapter weights should be used
if which_model == 'DPO' or which_model == 'mDPO':
    use_lora = True
elif which_model == 'Bunny':
    use_lora = False

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

# function to crop an image
def crop_image(image):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# processes a single data point
def prepare_inputs(prompt, response, tokenizer):
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

    # #print('./data/merged_images/' + img_path)
    # image = Image.open(img_path)
    # # process the image into a tensor
    # image_tensor = crop_image(model.process_images([image], model.config)).to(dtype=model.dtype)
    # batch["image"] = image_tensor

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

# user query
query = "How many traffic lights are there in the image?"
# image path
image_path = './data/test/count1.jpg'

# # user query
# query = "What colour are the traffic lights on the left?"
# # image path
# image_path = './data/test/count1.jpg'

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
# query = "What is the colour of the chairs in the image?"
# # response text
# response = "There are four chairs."
# # image path
# image_path = './data/test/count6.jpg'

# # user query
# query = "How many teddy bears are there in the image?"
# # response text
# response = "There are seven teddy bears."
# # image path
# image_path = './data/test/count8.jpg'

# # user query
# query = "How many traffic signs in the image?"
# # image path
# image_path = './data/test/count9.jpg'

# # user query
# query = "How many cars are there in the image?"
# # response text
# response = "There are two cars."
# # image path
# image_path = './data/test/count10.jpg'

# # user query
# query = "How many people are there in the image?"
# # response text
# response = "There are four people in the image."
# # image path
# image_path = './data/test/count11.jpg'

# # user query
# query = "How many oysters can you see in the photo?"
# # response text
# response = "There are five oysters visible in the photo."
# # image path
# image_path = './data/test/count12.jpg'

# # user query
# query = "How many horses are there in the photo?"
# # response text
# response = "There are three horses in the photo."
# # image path
# image_path = './data/test/count13.jpg'

# # user query
# query = "How many forks can you see?"
# # response text
# response = "There are two forks."
# # image path
# image_path = './data/test/count14.jpg'

# queries = [
#     ("How many traffic lights are there in the image?", './data/test/count1.jpg'),
#     ("What colour are the traffic lights on the left?", './data/test/count1.jpg'),
#     ("What colour are the traffic lights on the right?", './data/test/count1.jpg'),
#     ("How many bicycles are there in the image?", './data/test/count2.jpg'),
#     ("How many zebras are there in the image?", './data/test/count3.jpg'),
#     ("How many players are there in the image?", './data/test/count4.jpg'),
#     ("How many chairs are there in the image?", './data/test/count5.jpg'),
#     ("How many chairs are there in the image?", './data/test/count6.jpg'),
#     # ("How many traffic signs in the image?", './data/test/count9.jpg')
# ]

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

# function to calculate the generic attention map for a generic query
def generic_attention_map(answer, img_tensor, tokenizer, model, layer_index = -1):
    # generic query
    generic_query = "Write a general description of the image."

    # prompt text with <image> token
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{generic_query} ASSISTANT:"

    # get the inputs for the model
    data = prepare_inputs(prompt, answer, tokenizer)

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

    # get the attention scores from the last layer
    att_scores = outputs.attentions[layer_index].squeeze() # (heads, 781, 781)

    assert data["prompt_input_ids"].index(-200)==data["response_input_ids"].index(-200)

    # index position where the first image token is located
    image_token_pos = data["prompt_input_ids"].index(-200)

    # extract attention scores when predicting the first answer token using the last prompt token
    ans_img_attn_scores = att_scores[:, -1, image_token_pos:image_token_pos+729] # (heads, 729)

    # average over attention heads
    avg_attn = ans_img_attn_scores.mean(dim=0)  # (729,)

    # reshape to 27 x 27 (no. of patches) to get the spatial attention map
    spatial_attn_map = avg_attn.reshape(27, 27).cpu().detach().numpy()

    return spatial_attn_map

# function to calculate the spatial attention map for each answer token
def spatial_attention_map(question, answer, img_tensor, tokenizer, model, layer_index = -1):
    # list to store the spatial attention maps
    all_sam = []

    # prompt text with <image> token
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"

    # get the inputs for the model
    data = prepare_inputs(prompt, answer, tokenizer)

    # get the model outputs
    outputs = model(
        input_ids=torch.tensor(data["response_input_ids"], dtype=torch.long, device=device).unsqueeze(0),  # add batch dimension
        attention_mask=torch.tensor(data["response_attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        images=img_tensor.to(device),
        labels=None,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True
    )

    # get the attention scores from the last layer
    att_scores = outputs.attentions[layer_index].squeeze() # (heads, 781, 781)

    assert data["prompt_input_ids"].index(-200)==data["response_input_ids"].index(-200)

    # index position where the first image token is located
    image_token_pos = data["response_input_ids"].index(-200)

    # index position where the first answer token is located
    first_ans_idx = len(data["prompt_input_ids"])+728

    for i in range(first_ans_idx, att_scores.shape[1]):
        # extract attention scores when predicting i-th answer token
        ans_img_attn_scores = att_scores[:, i-1, image_token_pos:image_token_pos+729] # (heads, 729)

        # average over attention heads
        avg_attn = ans_img_attn_scores.mean(dim=0)  # (729,)

        # # normalize values
        # avg_attn = avg_attn / avg_attn.sum()

        # reshape to 27 x 27 (no. of patches) to get the spatial attention map
        spatial_attn_map = avg_attn.reshape(27, 27).cpu().detach().numpy()

        # get the i-th answer token
        ans_token = tokenizer.decode(data["response_input_ids"][i-728])

        all_sam.append((spatial_attn_map,ans_token))

    return all_sam

# function to apply image corruption
def corrupt_image(pil_image):
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    image_tensor = to_tensor(pil_image)
    corrupted_image_tensor = crop_image(image_tensor)

    return to_pil(corrupted_image_tensor.squeeze())

# function that will build a model to predict entries
# where generic is zero and actual is non-zero
def predict_inf(rel_attn, spt_attn, gen_attn):
    # flatten the attention maps
    rel_attn = rel_attn.flatten()
    spt_attn = spt_attn.flatten()
    gen_attn = gen_attn.flatten()

    # get those entries when generic is greater than zero
    mask = gen_attn > 1e-10

    # build the dataset
    x = np.stack([gen_attn[mask], spt_attn[mask]], axis=1)
    y = rel_attn[mask]

    # regression model
    regressor = KNeighborsRegressor(
        n_neighbors=5,
        weights='uniform'
    )
    regressor.fit(x, y)

    return regressor

# transformer layer index
layer_index = -1

# reopen the original image
#orig_image = Image.open(image_path)
orig_image = corrupt_image(Image.open(image_path))
# process the image into a tensor
image_tensor = model.process_images([orig_image], model.config).to(dtype=model.dtype)
# resize the image
orig_image_resized = orig_image.convert('RGB').resize((384, 384))
# crop the upper-left portion of the image
crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
orig_image_cropped = orig_image_resized.crop(crop_box)

# generate mdpo's response
response_text = generate_response(query, image_tensor, tokenizer, model)

# generate spatial attention maps of mdpo for every answer token
spt_attn_maps = spatial_attention_map(query, response_text, image_tensor, tokenizer, model, layer_index)

# generate generic attention map using the generic query
gen_attn_map = generic_attention_map(response_text, image_tensor, tokenizer, model, layer_index)

cols = 4
rows = (len(spt_attn_maps)+cols) // cols

# figure for the original image and the attention maps
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
axes = axes.flatten()

# plot the original image
axes[0].imshow(orig_image_cropped)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the relative attention maps
for i in range(len(spt_attn_maps)):
    # get the spatial attention map for the answer token
    spt_attn_map = spt_attn_maps[i][0]
    # numpy array to store the relative attention map
    rel_attn_map = np.zeros_like(spt_attn_map)
    # divide the actual attention map for the answer token by generic attention map
    np.divide(
        spt_attn_map,
        gen_attn_map,
        out=rel_attn_map,
        where=(gen_attn_map > 1e-10) # prevent divide-by-zero error
    )
    # there may still be division overflow problems if denominator is too small and numerator is too large
    # find places in relative attention map with finite values
    finite_mask = np.isfinite(rel_attn_map)
    # find the finite maximum value
    finite_max = rel_attn_map[finite_mask].max()
    # set infinite values to finite maximum value
    rel_attn_map[np.isinf(rel_attn_map)] = finite_max

    # ############
    # eps = 1e-10
    # zero_mask = mdpo_gen_attn_map <= eps
    # non_zero_mask = mdpo_gen_attn_map > eps

    # ne = "No Entries"
    # print(f"Generic Zero - Actual Attention Mean: {np.mean(mdpo_spt_attn_map[ zero_mask ]):e}")
    # print(f"Generic Non-Zero - Actual Attention Mean: {np.mean(mdpo_spt_attn_map[non_zero_mask]):e}\n")

    # ###################################################################################################################
    # # 1. use a regression model to predict those entries when generic is zero and actual is non-zero
    # # build the regression model using the attention maps
    # regressor = predict_inf(rel_attn_map, mdpo_spt_attn_map, mdpo_gen_attn_map)

    # assert rel_attn_map.shape==mdpo_spt_attn_map.shape and rel_attn_map.shape==mdpo_gen_attn_map.shape

    # # get dimensions of relative attention map
    # height, width = rel_attn_map.shape

    # for j in range(height):
    #     for k in range(width):
    #         gen = mdpo_gen_attn_map[j,k]
    #         act = mdpo_spt_attn_map[j,k]

    #         # if generic is zero and actual is non-zero use the regression model to predict the values
    #         if gen <= 1e-10 and act > 0:
    #             x = np.array([[gen, act]])
    #             pred = regressor.predict(x)[0]
    #             rel_attn_map[j,k] = pred
    # ###################################################################################################################

    # ###################################################################################################################
    # # 2. fill those entries when generic is zero and actual is non-zero with half of maximum relative attention
    # # find the maximum relative attention
    # max_rel_attn = np.max(rel_attn_map)

    # # find those cases when generic is zero but actual attention is non-zero
    # mask = (gen_attn_map <= 1e-10) & (spt_attn_map > 0)

    # rel_attn_map[mask] = max_rel_attn/2
    # ###################################################################################################################

    # ###################################################################################################################
    # # 3. fill those entries when generic is zero and actual is non-zero with actual attention divided by minimum such attention and then multiply with maximum relative attention
    # # find the maximum relative attention
    # max_rel_attn = np.max(rel_attn_map)

    # # find those cases when generic is zero but actual attention is non-zero
    # mask = (gen_attn_map <= 1e-10) & (spt_attn_map > 0)

    # # find the minimum actual attention
    # min_act_attn = np.min(spt_attn_map[mask])

    # rel_attn_map[mask] = (spt_attn_map[mask]/min_act_attn)*max_rel_attn
    # ###################################################################################################################

    axes[i+1].imshow(rel_attn_map, cmap='viridis')
    axes[i+1].set_title(f"Token: {spt_attn_maps[i][1]}")
    axes[i+1].axis('off')

# turn off remaining plots
for i in range(len(spt_attn_maps)+1, len(axes)):
    axes[i].axis('off')

plt.suptitle(f"Query: {query}\n{which_model} Answer: {response_text}", fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'./results/spatial_attn_map.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()