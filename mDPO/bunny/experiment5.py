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

# PROBLEM: Negative text mining causes the model to dissociate from false negative phrases (phrases which are semantically correct)​
# HYPOTHESIS: The negative phrases are created by changing some of the phrases in the chosen response but, 
#             it is not checked if that negative phrase still correct describes the corresponding phrase in the chosen response​
# EXPERIMENT: Take an image, a query and a partial response and compute the probability of the next response token
#             for such a token that is similar to the token in the chosen response but still correct
#             for both the reference and DPA model and compare them 

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
    #     "response_input_ids": [101, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # input token IDs for the prompt + response
    #     "response_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask
    #     "response__labels": [-100, -100, -100, -100, -100, -100, -100, 2023, 2003, 1996, 2419, 3430, 1012, 102, 1],  # labels for the prompt + chosen response with prompt part masked
    #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
    #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
    #     "image": <tensor representation of the image>  # image tensor
    # }

    return batch

# # query text
# query = "Relay a brief, clear account of the picture shown."
# # prompt text with <image> token
# prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
# # partial response text
# response = "Standing next to the gray truck is a"
# # image path
# image_path = './data/vg/VG_100K/2316038.jpg'
# # token whose logits we want
# target_token = 'person'

# # query text
# query = "Summarize the visual content of the image."
# # prompt text with <image> token
# prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
# # partial response text
# response = "In the picture there are three"
# # image path
# image_path = './data/vg/VG_100K/2354829.jpg'
# # token whose logits we want
# target_token = 'children'

# # query text
# query = "Narrate the contents of the image with precision."
# # prompt text with <image> token
# prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
# # partial response text
# response = "Standing near a store is a man and a"
# # image path
# image_path = './data/vg/VG_100K/2344962.jpg'
# # token whose logits we want
# target_token = 'lady'

# query text
query = "Present a compact description of the photo."
# prompt text with <image> token
prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
# partial response text
response = "Tennis is being played by a"
# image path
image_path = './data/vg/VG_100K/2362406.jpg'
# token whose logits we want
target_token = 'guy'

# get the inputs for the model
data = prepare_inputs(prompt, response, image_path, tokenizer, reference_model)
#print(data)

# function when given a partial response
# gets the probabilities corresponding to the last token
# which is the probability of the next token
# and then pick the target token's probability which is given as an argument to the function
def last_token_prob(model, input_ids, attention_mask, image, tokenizer, token=None):

    if token is None:
        raise Exception("No token is given")

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
        last_token_probs = torch.softmax(last_token_logits, dim=-1)

        # get the token ids of the token whose logits we want
        token_ids = tokenizer(token, add_special_tokens=False)['input_ids']
        # # check if the token has not been split into multiple tokens
        # if len(token_ids) > 1:
        #     raise Exception("Target token is split into multiple tokens")
        
        # get the token id of the token we want the logits for
        token_id = token_ids[0]
        
        return last_token_probs[token_id].item()

# target token prob from reference model
ref_last_prob = last_token_prob(reference_model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer, target_token)
# target token prob from dpa model
dpa_last_prob = last_token_prob(dpa_model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer, target_token)

print(f"Reference model {target_token} probability: {ref_last_prob:.2e}")
print(f"DPA model {target_token} probability: {dpa_last_prob:.2e}")