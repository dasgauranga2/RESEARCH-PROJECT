import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
import math
import json

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

# path of saved checkpoint
checkpoint_path = './checkpoint/mdpo_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

if use_lora:
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path
    )

    model = model.merge_and_unload()

# set model to evaluation mode
model.eval()

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# function to prepare the inputs for hellinger metrics
def prepare_inputs(
        prompt, # prompt text
        response, # response text
        img_path, # image path
        tokenizer, 
        model
    ):
        # dictionary to store the inputs
        batch = {}

        # tokenize the response texts
        response_tokens = tokenizer(response, add_special_tokens=False)

        # prompt text by default contains the <image> token
        # replace that to produce imageless conditioned response
        prompt_imageless = prompt.replace('<image>','')

        prompt_tokens = {}
        # prompt token ids
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, tokenizer)
        # the attention mask helps the model differentiate between actual input tokens and padding tokens
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        # repeat for imageless
        prompt_imageless_tokens = {}
        prompt_imageless_tokens["input_ids"] = tokenizer_image_token(prompt_imageless, tokenizer)
        prompt_imageless_tokens["attention_mask"] = [1 for _ in range(len(prompt_imageless_tokens["input_ids"]))]

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

        # repeat for imageless
        eos_indices_prompt_imageless = [i for i, x in enumerate(prompt_imageless_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_imageless = [
            0 if i in eos_indices_prompt_imageless else p for i, p in enumerate(prompt_imageless_tokens["attention_mask"])
        ]
        prompt_imageless_tokens["attention_mask"] = new_attention_mask_imageless

        eos_indices_response = [i for i, x in enumerate(response_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_response = [
            0 if i in eos_indices_response else p for i, p in enumerate(response_tokens["attention_mask"])
        ]
        response_tokens["attention_mask"] = new_attention_mask_response

        # add EOS token to end of responses
        response_tokens["input_ids"].append(tokenizer.eos_token_id)
        response_tokens["attention_mask"].append(1)

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
        conditioned_tokens = {k: prompt_tokens[k] + response_tokens[k] for k in response_tokens}
        unconditioned_tokens = {k: prompt_imageless_tokens[k] + response_tokens[k] for k in response_tokens}
        # lables are created from the above tokens such that
        # tokens corresponding to prompt tokens are masked 
        conditioned_tokens["labels"] = conditioned_tokens["input_ids"][:]
        conditioned_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])

        unconditioned_tokens["labels"] = unconditioned_tokens["input_ids"][:]
        unconditioned_tokens["labels"][: len(prompt_imageless_tokens["input_ids"])] = [-100] * len(prompt_imageless_tokens["input_ids"])

        batch["conditioned_input_ids"] = conditioned_tokens["input_ids"]
        batch["conditioned_attention_mask"] = conditioned_tokens["attention_mask"]
        batch["conditioned_labels"] = conditioned_tokens["labels"]

        batch["unconditioned_input_ids"] = unconditioned_tokens["input_ids"]
        batch["unconditioned_attention_mask"] = unconditioned_tokens["attention_mask"]
        batch["unconditioned_labels"] = unconditioned_tokens["labels"]

        image = Image.open(img_path).convert('RGB')
        # process the image into a tensor
        image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
        batch["image"] = image_tensor

        # the final result will be of this format
        #     batch = {
        #     "conditioned_input_ids" # input token IDs for the prompt + image + response
        #     "conditioned_attention_mask" # attention mask
        #     "conditioned_labels" # labels for the conditioned_input_ids with prompt tokens masked
        #     "unconditioned_input_ids" # input token IDs for the prompt + response
        #     "unconditioned_attention_mask" # attention mask
        #     "unconditioned_labels" # labels for the unconditioned_input_ids with prompt tokens masked
        #     "image": <tensor representation of the image>  # image tensor
        # }

        return batch

# open the file with queries and image paths
with open('./AMBER/data/query/query_generative.json') as file:
    queries = json.load(file)

# open the file with responses
with open("./AMBER/mdpo_results.json", "r") as file:
    responses = json.load(file)

assert len(queries)==len(responses)

# total response tokens seen so far
total_count = 0
# number of surprise tokens seen so far
surprise_count = 0
# sum of hellinger distance values
hellinger_sum = 0

for i, (query_data, response_data) in enumerate(zip(queries, responses)):

    assert query_data['id']==response_data['id']

    # prompt text with <image> token
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {query_data['query']}\n<image> ASSISTANT:"
    # response text
    response = response_data['response']
    # image path
    image_path = './AMBER/data/image/' + query_data['image']

    # get the inputs for the model
    data = prepare_inputs(prompt, response, image_path, tokenizer, model)

    # list to compute Hellinger distance for each token
    hellinger_distance = []

    with torch.no_grad():
        # feedforward the conditioned inputs with image
        outputs = model(
            input_ids=torch.tensor(data["conditioned_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
            attention_mask=torch.tensor(data["conditioned_attention_mask"], dtype=torch.long).unsqueeze(0),
            images=data["image"].unsqueeze(0),  # model expects batch size
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # get the conditioned probabilities
        logits = outputs.logits.squeeze()
        probs = torch.softmax(logits, dim=-1)

        # feedforward the unconditioned inputs without the image
        outputs_imageless = model(
            input_ids=torch.tensor(data["unconditioned_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
            attention_mask=torch.tensor(data["unconditioned_attention_mask"], dtype=torch.long).unsqueeze(0),
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # get the unconditioned probabilities
        logits_imageless = outputs_imageless.logits.squeeze()
        probs_imageless = torch.softmax(logits_imageless, dim=-1)

        # get the response length
        response_length = (torch.tensor(data["conditioned_labels"]) != -100).sum().item()
        response_imageless_length = (torch.tensor(data["unconditioned_labels"]) != -100).sum().item()
        assert response_length == response_imageless_length

        # slice only the probabilities corresponding to response tokens
        # since, when calculating Prompt Dependency Measure using Hellinger distance
        # we only calculate probabilities of response tokens
        probs_response = probs[-(response_length+1):]  # conditioned
        probs_imageless_response = probs_imageless[-(response_length+1):]  # unconditioned

        # get the response token ids
        response_token_ids = [token_id for token_id in data["conditioned_labels"] if token_id != -100]

        # decode each token
        response_tokens = [tokenizer.decode([token_id]) for token_id in response_token_ids]

        # pre-compute square root of 2
        sqrt2 = math.sqrt(2) 

        for j in range(response_length):
            # conditioned probability
            probs_t = probs_response[j]
            # unconditioned probability
            probs_imageless_t = probs_imageless_response[j]

            # calculate the Hellinger distance
            H = torch.sqrt(torch.sum((torch.sqrt(probs_t) - torch.sqrt(probs_imageless_t))**2)) / sqrt2

            # get the corresponding token
            token = response_tokens[j]

            hellinger_distance.append((token, H.item()))

            hellinger_sum += H
            if H > 0.5:
                surprise_count += 1
        
        total_count += response_length

    if i > 0 and i%100 == 0:
         print(f"Surprise Tokens: {(surprise_count/total_count)*100:.2f}%\tAverage Hellinger Distance: {hellinger_sum/total_count:.2f}")