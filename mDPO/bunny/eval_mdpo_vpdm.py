import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
import math

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

        #print('./data/merged_images/' + img_path)
        image = Image.open(img_path)
        # process the image into a tensor
        image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
        batch["image"] = image_tensor

        return batch

# prompt text
prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Describe the image?\n<image> ASSISTANT:"
response = "The image shows a mouse with a cable connected to it."
# load the image
image_path = './data/test2.png'

data = prepare_inputs(prompt, response, image_path, tokenizer, model)

with torch.no_grad():
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

    logits = outputs.logits.squeeze()
    probs = torch.softmax(logits, dim=-1)

    outputs_imageless = model(
        input_ids=torch.tensor(data["unconditioned_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
        attention_mask=torch.tensor(data["unconditioned_attention_mask"], dtype=torch.long).unsqueeze(0),
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )

    logits_imageless = outputs_imageless.logits.squeeze()
    probs_imageless = torch.softmax(logits_imageless, dim=-1)

    # Step 1: Get response length (already done)
    response_length = (torch.tensor(data["conditioned_labels"]) != -100).sum().item()
    response_imageless_length = (torch.tensor(data["unconditioned_labels"]) != -100).sum().item()

    assert response_length == response_imageless_length

    # Step 2: Slice last `response_length` probabilities
    probs_response = probs[-(response_length+1):]  # conditioned
    probs_imageless_response = probs_imageless[-(response_length+1):]  # unconditioned

    # Step 3: Get response token IDs
    response_token_ids = [token_id for token_id in data["conditioned_labels"] if token_id != -100]

    # Step 4: Decode each token
    response_tokens = [tokenizer.decode([token_id]) for token_id in response_token_ids]

    # Step 5: Compute Hellinger distances (shifted correctly)
    hellinger_distance = []

    sqrt2 = math.sqrt(2)  # precompute sqrt(2)

    # we skip the first probs because it predicts the 1st token, and we cannot align it cleanly
    for i in range(len(probs_response) - 1):
        probs_t = probs_response[i]
        probs_imageless_t = probs_imageless_response[i]

        # Hellinger distance
        H = torch.sqrt(torch.sum((torch.sqrt(probs_t) - torch.sqrt(probs_imageless_t))**2)) / sqrt2

        # Correct token matching: Hellinger at step i → token at i+1
        token = response_tokens[i]  # shift by +1

        hellinger_distance.append((token, H.item()))
    
    print(hellinger_distance)