import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# PROBLEM: DPA produces negative text which may not have logical sense
# EXPERIMENT: Take an image, a query and a partial response and compute the probabilities of the next response token
#             for both the reference and DPA model

# load the base model
reference_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# load the base model
dpa_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = './checkpoint/dpa_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

if use_lora:
    dpa_model = PeftModel.from_pretrained(
        dpa_model,
        checkpoint_path
    )

    dpa_model = dpa_model.merge_and_unload()

# load the vision towers
reference_model.get_vision_tower().load_model()
dpa_model.get_vision_tower().load_model()

# set model to evaluation mode
reference_model.eval()
dpa_model.eval()

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
def prepare_inputs(prompt, partial_response, img_path, tokenizer, model):
    # dictionary to store the inputs
    batch = {}

    # tokenize the response texts
    response_tokens = tokenizer(partial_response, add_special_tokens=False)
    #rejected_tokens = tokenizer(rejected, add_special_tokens=False)

    prompt_tokens = {}
    # prompt token ids
    prompt_tokens["input_ids"] = tokenizer_image_token(prompt, tokenizer)
    # the attention mask helps the model differentiate between actual input tokens and padding tokens
    prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

    # get the end-of-sequence token id
    eos_token_id = tokenizer.eos_token_id

    # the steps below adjust the attention mask by setting the position corresponding to exisiting EOS tokens to 0
    # this ensures that the model does not attend to any tokens that come after the EOS token
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

# query text
query = "Write an informative summary of the picture."
# prompt text with <image> token
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
# partial response text
response = "A person is sitting in a chair in a park. There is a suitcase next to them. In the background there is a large"
# image path
image_path = './data/vg/VG_100K/2348476.jpg'

# get the inputs for the model
data = prepare_inputs(prompt, response, image_path, tokenizer, reference_model)
#print(data)

# function to get the top token probabilities
def top_token_probs(model, input_ids, attention_mask, image, tokenizer, top=5):
    # list to store probabilities of each token
    probabilities = []

    # result string
    result = ""

    with torch.no_grad():
        # feedforward the inputs
        outputs = model(
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
        logits = outputs.logits.squeeze()

        # get logits corresponding to the last token
        last_token_logits = logits[-1,:]
        
        # get probabilities
        probs = torch.softmax(last_token_logits, dim=-1)

        # get probabilities of each token
        for i in range(len(probs)):
            probabilities.append((tokenizer.decode([i], skip_special_tokens=True), probs[i].item()))

        # sort list of probabilities
        probabilities.sort(key=lambda x:x[1], reverse=True) 

        # print the top probabilities
        for tok, prob in probabilities[:top]:
            result = result + f"{tok:<10}" + f"{prob:6.4f}" + "\n"
            #print(f"{tok} {prob:.4f}")
    
    return result

#top_token_probs(model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer)
print("Reference model Probs")
print(top_token_probs(reference_model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer))
print("DPA model Probs")
print(top_token_probs(dpa_model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer))