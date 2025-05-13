import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
import json
import re
import time
import spacy

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# PROBLEM: Existing hallucinations in DPA still exist if rejected phrases don't contain the hallucinated object
# HYPOTHESIS: If we have an image of cat, the chosen response contains 'cat' and the model is hallucinating
#             and predicts the token 'dog'. We want the rejected response to contain 'dog' instead of another token.
# EXPERIMENT: Take an image, a query and a partial response and compute the probabilities of the next response token
#             for both the reference and DPA model

# load the base model
model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# load the vision tower
model.get_vision_tower().load_model()

# set model to evaluation mode
model.eval()

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    model_max_length=2048,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True)

# set the padding token
tokenizer.pad_token_id = tokenizer.eos_token_id

# function to prepare inputs
def prepare_inputs(prompt, response, img_path, model, tokenizer):
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

    # do the same for chosen and rejected responses
    eos_indices_response = [i for i, x in enumerate(response_tokens["input_ids"]) if x == eos_token_id]
    new_attention_mask_c = [
        0 if i in eos_indices_response else p for i, p in enumerate(response_tokens["attention_mask"])
    ]
    response_tokens["attention_mask"] = new_attention_mask_c


    # add EOS token to end of responses
    response_tokens["input_ids"].append(tokenizer.eos_token_id)
    response_tokens["attention_mask"].append(1)

    # # determine the longer of the chosen and rejected response
    # longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # # if combined sequence is too long, truncate the prompt
    # if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
    #     if self.truncation_mode == "keep_start":
    #         prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
    #     elif self.truncation_mode == "keep_end":
    #         prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
    #     else:
    #         raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

    # # if that's still too long, truncate the response
    # if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
    #     chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
    #     rejected_tokens = {
    #         k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
    #     }

    # concatenate the prompt and response tokens
    response_sequence_tokens = {k: prompt_tokens[k] + response_tokens[k] for k in response_tokens}
    
    # labels are created from the above tokens such that
    # tokens corresponding to prompt tokens are masked 
    response_sequence_tokens["labels"] = response_sequence_tokens["input_ids"][:]
    response_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [tokenizer.pad_token_id] * len(
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
    #     "response_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + chosen response
    #     "response_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
    #     "response_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
    #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
    #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
    #     "image": <tensor representation of the image>  # image tensor of shape (1, channels, height, width)
    # }

    return batch

img_path = "./data/merged_images/llava-reasoning-3730.jpg"
prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nHow might the atmosphere of this photo appear to be, and what is one possible reason these people have gathered outside? ASSISTANT:"
response = "The atmosphere of the photo appears to be rainy and gloomy, as the people are holding umbrellas to protect themselves from the rain. One possible reason for these people gathering outside is that they are waiting for a bus or a taxi, as there are cars and a bus visible in the image. The presence of handbags and backpacks among the people suggests that they might be commuters or travelers, and the rain might have caused some delays or inconvenience in their plans."

# prepare the input data
data = prepare_inputs(prompt, response, img_path, model, tokenizer)

# get the model outputs
outputs = model(
    input_ids=torch.tensor(data["response_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
    attention_mask=torch.tensor(data["response_attention_mask"], dtype=torch.long).unsqueeze(0),
    images=data["image"],
    labels=None,
    use_cache=False,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True
)

# get the model logits
logits = outputs.logits.squeeze()

nlp = spacy.load("en_core_web_sm")

# function to check if a word is an object
def is_object(word):
    doc = nlp(word)
    return doc[0].pos_ in ["NOUN", "PROPN"]

# start checking in reverse order
for i in range(1, len(data["response_labels"])+1):
    # get the response token id
    token_id = data["response_labels"][-i]

    # skip padding tokens
    if token_id == tokenizer.pad_token_id:
        continue

    # decode the token
    token = tokenizer.decode(token_id).strip().lower()

    # check if token is an object
    if is_object(token):
        # get the response token's corresponding position logits for the entire vocabulary
        # we get the logits for one step before
        token_logits = logits[-i-1,:]

        print(token)

        # get the indices of the top-10 logits
        _ , top_indices = torch.topk(token_logits, 10)

        for ind in top_indices:
            # decode each top token
            print(tokenizer.decode(ind).strip().lower())

    print("\n\n")