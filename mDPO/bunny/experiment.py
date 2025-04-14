import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token

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

# processes a single data point
def prepare_inputs(prompt, chosen, rejected, img_path, tokenizer, model):
    # dictionary to store the inputs
    batch = {}

    # tokenize the response texts
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)

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
    eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
    new_attention_mask_c = [
        0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
    ]
    chosen_tokens["attention_mask"] = new_attention_mask_c

    eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
    new_attention_mask_r = [
        0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
    ]
    rejected_tokens["attention_mask"] = new_attention_mask_r

    # add EOS token to end of responses
    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    # determine the longer of the chosen and rejected response
    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

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
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    # lables are created from the above tokens such that
    # tokens corresponding to prompt tokens are masked 
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
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
    #     "chosen_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + chosen response
    #     "chosen_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
    #     "chosen_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
    #     "rejected_input_ids": [101, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # input token IDs for the prompt + rejected response
    #     "rejected_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
    #     "rejected_labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 3446, 3430, 1012, 102, 1],  # labels for the prompt + rejected response with prompt part masked
    #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
    #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
    #     "image": <tensor representation of the image>  # image tensor
    # }

    return batch

# prompt text with <image> token
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Describe the image\n<image> ASSISTANT:"
# chosen response text
chosen = "There is an orange cat sitting in the sofa"
# rejected response text
rejected = "The orange cat is sleeping"
# image path
image_path = './data/test3.png'

# get the inputs for the model
data = prepare_inputs(prompt, chosen, rejected, image_path, tokenizer, model)
#print(data)

with torch.no_grad():
    # feedforward the chosen inputs
    chosen_outputs = model(
        input_ids=torch.tensor(data["chosen_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
        attention_mask=torch.tensor(data["chosen_attention_mask"], dtype=torch.long).unsqueeze(0),
        images=data["image"].unsqueeze(0),  # model expects batch size
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True
    )

    # get the chosen logits
    chosen_logits = chosen_outputs.logits.squeeze()

    # decode the chosen sequence tokens
    chosen_tokens = [tokenizer.decode([token_id]) for token_id in data["chosen_input_ids"]]

    # the chosen logits and chosen_tokens lengths won't match
    # since the chosen logits have extra image tokens inserted between them
    # but, both sequences end with response tokens so reverse them both
    chosen_tokens.reverse()
    chosen_logits = chosen_logits[:-1]
    
    print(chosen_logits.shape, len(data["chosen_input_ids"]))

    # # feedforward the rejected inputs
    # rejected_outputs = model(
    #     input_ids=torch.tensor(data["rejected_input_ids"], dtype=torch.long).unsqueeze(0),  # add batch dimension
    #     attention_mask=torch.tensor(data["rejected_attention_mask"], dtype=torch.long).unsqueeze(0),
    #     images=data["image"].unsqueeze(0),  # model expects batch size
    #     labels=None,
    #     use_cache=False,
    #     output_attentions=False,
    #     output_hidden_states=False,
    #     return_dict=True
    # )

    # # get the rejected logits
    # rejected_logits = rejected_outputs.logits.squeeze()