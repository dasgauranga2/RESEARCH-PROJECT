import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
import json

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# PROBLEM: Existing hallucinations in DPA still exist if negative phrases don't contain the hallucinated object
# HYPOTHESIS: If we have an image of cat, the chosen response contains 'cat' and the model is hallucinating
#             and predicts the token 'dog'. We want the rejected response to contain 'dog' instead of another token.
# EXPERIMENT: Take an image, a query and a partial response and compute the probabilities of the next response token
#             for both the reference and DPA model

# load the base model
reference_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# # load the base model
# dpa_model = AutoModelForCausalLM.from_pretrained(
#     'BAAI/Bunny-v1_0-3B',
#     torch_dtype=torch.float16, # float32 for cpu
#     device_map='auto',
#     trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = './checkpoint/dpa_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

# if use_lora:
#     dpa_model = PeftModel.from_pretrained(
#         dpa_model,
#         checkpoint_path
#     )

#     dpa_model = dpa_model.merge_and_unload()

# load the vision towers
reference_model.get_vision_tower().load_model()
#dpa_model.get_vision_tower().load_model()

# set model to evaluation mode
reference_model.eval()
#dpa_model.eval()

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
# query = "Give a short and clear explanation of the subsequent image."
# # prompt text with <image> token
# prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
# # partial response text
# response = "A blue and white semi-truck with an empty flatbed trailer is parked in a parking lot next to a river. In the background, there is a yellow"
# # image path
# image_path = './data/vg/VG_100K_2/2396016.jpg'

# # get the inputs for the model
# data = prepare_inputs(prompt, response, image_path, tokenizer, reference_model)
# #print(data)

# function that will be given a partial response
# and we get the last token's corresponding top output probabilities
# these are the probabilities of the next token
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
    
    return probabilities[:top]

# #top_token_probs(model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer)
# print("Reference model Probs")
# print(top_token_probs(reference_model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer))
# print("DPA model Probs")
# print(top_token_probs(dpa_model, data["response_input_ids"], data["response_attention_mask"], data["image"], tokenizer))

# function when given a masked answer
# will produce all possible input-output pairs
# such that output will be a phrase inside the maksed tokens
# and input will be all preceeding text
def generate_phrase_inp_out_pairs(masked_answer):
    result = []
    current = ""
    i = 0

    while i < len(masked_answer):
        if masked_answer[i:i+6] == '<MASK>':
            #print(f"Input:-->{current.strip()}<--")

            j = i+7
            target = ""
            while masked_answer[j:j+7] != '</MASK>':
                target = target + masked_answer[j]
                j += 1

            #print(f"Target:-->{target.strip()}<--\n")
            result.append((current.strip(), target.strip()))
            current = current + target.strip()

            i = j+7
            continue
        else:
            current = current+masked_answer[i]

        i += 1
    
    return result

# open the file with queries and image paths
with open('./data/dpa_data.json') as file:
    data = json.load(file)
#print(data[0])
#print(data[1]['correct_answer_masked'])

# sub-optimal hallucination count
soh_count = 0

for i in range(len(data)):

    # # correct_answer_masked = "A <MASK> man </MASK> is lying on a <MASK> bed </MASK> in a hotel room. There is a <MASK> black </MASK> backpack, a <MASK> brown </MASK> suitcase, and a <MASK> white </MASK> telephone on the <MASK> bed </MASK>. The <MASK> window </MASK> has <MASK> light </MASK> <MASK> pink </MASK> curtains and <MASK> white </MASK> blinds."
    # # hallucinated_answer_masked = "A <MASK> dog </MASK> is resting on a <MASK> couch </MASK> in a hotel room. There is a <MASK> blue </MASK> backpack, a <MASK> red </MASK> suitcase, and a <MASK> black </MASK> telephone on the <MASK> couch </MASK>. The <MASK> door </MASK> has <MASK> dark </MASK> <MASK> blue </MASK> curtains and <MASK> black </MASK> blinds."
    correct_answer_masked = data[i]['correct_answer_masked']
    hallucinated_answer_masked = data[i]['hallucinated_answer_masked']

    correct_pairs = generate_phrase_inp_out_pairs(correct_answer_masked)
    hallucinated_pairs = generate_phrase_inp_out_pairs(hallucinated_answer_masked)

    # query text
    query = data[i]['question'].replace('<image>','').replace('\n','')
    # prompt text with <image> token
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{query} ASSISTANT:"
    # # partial response text
    # response = "A person is sitting in a chair in a park. There is a suitcase next to him. In the background there is a large"
    # image path
    image_path = './data/' + data[i]['image']

    for (inp, correct_out), (_, hallucinated_out) in zip(correct_pairs, hallucinated_pairs):
        # get the inputs for the model
        input_data = prepare_inputs(prompt, inp, image_path, tokenizer, reference_model)

        ttp = top_token_probs(reference_model, input_data["response_input_ids"], input_data["response_attention_mask"], input_data["image"], tokenizer)

        # check if the model has made the incorrect prediction
        if correct_out.lower() != ttp[0][0].strip().lower():
            # check if the hallucinated phrase is the token predicted by the model
            if hallucinated_out.lower() != ttp[0][0].strip().lower():
                soh_count += 1
                #print(f"Input: {inp}\nCorrect output: {correct_out}\nHallucinated output: {hallucinated_out}")
                #print(f"Top-5 Probabilities:\n{ttp}\n")
            # if hallucinated_out != ttp[1][0]:
            #     soh_count += 1
        # else:
        #     if hallucinated_out != ttp[0][0]:
        #         soh_count += 1

    if i%10 == 0:
        print(f"Average SOH Count: {soh_count/(i+1):.2f}")