import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
import torch.nn.functional as F

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# load the reference model
reference_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# load the mdpo model
mdpo_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = './checkpoint/mdpo_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

if use_lora:
    mdpo_model = PeftModel.from_pretrained(
        mdpo_model,
        checkpoint_path
    )

    mdpo_model = mdpo_model.merge_and_unload()

# load the vision towers
reference_model.get_vision_tower().load_model()
mdpo_model.get_vision_tower().load_model()

# set model to evaluation mode
reference_model.eval()
mdpo_model.eval()

print(mdpo_model.get_vision_tower().is_loaded)
print(reference_model.get_vision_tower().is_loaded)

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
    #     "image": <tensor representation of the image>  # image tensor of shape (1,C,W,H)
    # }

    return batch

# prompt text with <image> token
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Describe the image\n<image> ASSISTANT:"
# chosen response text
chosen = "There is an orange cat sitting on the sofa looking directly at the camera"
# rejected response text
rejected = "The orange cat is sleeping on the table"
# image path
image_path = './data/test3.png'

# get the inputs for the model
data = prepare_inputs(prompt, chosen, rejected, image_path, tokenizer, mdpo_model)
#print(data)

# function to calculate the log-probability of the response
def compute_response_log_probs(model, input_ids, attention_mask, labels, image):

    with torch.no_grad():
        # feedforward the inputs
        model_outputs = model(
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),  # add batch dimension
            attention_mask=torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
            images=image,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        # get the logits
        model_logits = model_outputs.logits.squeeze()

        # get the response token ids
        response_token_ids = [token_id for token_id in labels if token_id != -100]

        # response length
        response_length = len(response_token_ids)

        # get the logits corresponding to response tokens
        response_logits = model_logits[-(response_length+1):-1,:]

        # get the response token ids tensor
        target_ids = torch.tensor(response_token_ids, dtype=torch.long)

        # compute log-probabilities
        response_log_probs = F.log_softmax(response_logits, dim=-1)
        # get log-probabilities corresponding to response tokens
        response_token_log_probs = response_log_probs[range(response_length), target_ids]

        # total log-probability
        total_response_log_prob = response_token_log_probs.sum()

        return total_response_log_prob

#print(data['image'].shape)
# get the corruped image
corrupted_image = crop_image(data['image'])

reference_log_probs = compute_response_log_probs(reference_model, data['chosen_input_ids'], data['chosen_attention_mask'], data['chosen_labels'], data['image'])
mdpo_log_probs = compute_response_log_probs(mdpo_model, data['chosen_input_ids'], data['chosen_attention_mask'], data['chosen_labels'], data['image'])
reference_corrupted_log_probs = compute_response_log_probs(reference_model, data['chosen_input_ids'], data['chosen_attention_mask'], data['chosen_labels'], corrupted_image)
mdpo_corrupted_log_probs = compute_response_log_probs(mdpo_model, data['chosen_input_ids'], data['chosen_attention_mask'], data['chosen_labels'], corrupted_image)

print(f"Reference Log-Probs: {reference_log_probs:.4f}")
print(f"mDPO Log-Probs: {mdpo_log_probs:.4f}")
print(f"Corrupted Reference Log-Probs: {reference_corrupted_log_probs:.4f}")
print(f"Corrupted mDPO Log-Probs: {mdpo_corrupted_log_probs:.4f}")