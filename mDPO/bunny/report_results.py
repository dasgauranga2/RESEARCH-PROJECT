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
import textwrap
from matplotlib.patches import Rectangle, FancyBboxPatch

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# IMAGE FOR REPORT

# variable to decide which checkpoints to use
# model_names = [
#     'mdpo_bunny',
#     'mdpo_bunny_dci_is',
#     'mdpo_bunny_sd'
# ]
model_name = 'mdpo_bunny'
#model_name = 'mdpo_bunny_dci_is'
#model_name = 'mdpo_bunny_sd'

# load the reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# function to load a model from the checkpoint name
def load_model(name):
    # load the reference model
    base_model = AutoModelForCausalLM.from_pretrained(
        'BAAI/Bunny-v1_0-3B',
        torch_dtype=torch.float16, # float32 for cpu
        device_map='auto',
        trust_remote_code=True)
    
    # path of saved checkpoint
    checkpoint_path = f'./checkpoint/{name}'

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path
    )

    model = model.merge_and_unload()

    # load the vision tower
    model.get_vision_tower().load_model()

    # set model to evaluation mode
    model.eval()

    return model

# load the model from the checkpoint name
model = load_model(model_name)

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    './checkpoint/mdpo_bunny',
    trust_remote_code=True)

# processes a single data point
def prepare_inputs(prompt, tokenizer):
    # dictionary to store the inputs
    batch = {}

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

    for k, toks in {
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
    #     "prompt_input_ids": [101, 2023, 2003, 2019, 2742, 3430, 2007, 2019, 999, 1],  # input token IDs for the prompt with image tokens
    #     "prompt_attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # attention mask for the prompt
    # }

    return batch

queries = [
    "How many traffic lights are there in the image?",
    "How many bicycles are there in the image?",
    "How many players are there in the image?",
    "How many chairs are there in the image?"
]

image_paths = [
    './data/test/count1.jpg',
    './data/test/count2.jpg',
    './data/test/count4.jpg',
    './data/test/count6.jpg'
]

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

cols = 4
rows = 1

# figure for the original image
fig, axes = plt.subplots(rows, cols, figsize=(cols*2.8, 4))
axes = axes.flatten()

# add a grid of 2 rows per column: image on top, text underneath
gs = fig.add_gridspec(
    nrows=2, ncols=cols,
    height_ratios=[1.0, 0.2],  # tune the bottom ratio if you need more/less text space
    hspace=0.02, wspace=0.2
)

for i, (query, image_path) in enumerate(zip(queries, image_paths)):

    # reopen the original image
    orig_image = Image.open(image_path)
    #orig_image = corrupt_image(Image.open(image_path))
    # process the image into a tensor
    image_tensor = ref_model.process_images([orig_image], ref_model.config).to(dtype=ref_model.dtype)
    # resize the image
    orig_image_resized = orig_image.convert('RGB').resize((384, 384))
    # crop the upper-left portion of the image
    crop_box = (0, 0, 378, 378)  # (left, upper, right, lower)
    orig_image_cropped = orig_image_resized.crop(crop_box)    

    # generate the model's response
    response = generate_response(query, image_tensor, tokenizer, model)
    print(f"{model_name} Response: {response}")

    axes[i].axis('off')

    # plot the image
    ax_img = fig.add_subplot(gs[0, i])
    ax_img.imshow(orig_image_cropped)
    ax_img.axis("off")

    # axis for text
    ax_txt = fig.add_subplot(gs[1, i])
    ax_txt.axis("off")

    pos_img = ax_img.get_position()
    pos_txt = ax_txt.get_position()
    #ax_txt.set_position([pos_img.x0, pos_txt.y0, pos_img.width, pos_txt.height])

    # Wrap text so it stays inside the column width
    # (adjust width to taste; smaller numbers wrap sooner)
    q_wrapped = textwrap.fill(f"Q: {query}", width=38)
    a_wrapped = textwrap.fill(f"A: {response}", width=38)

    # Draw text anchored at the top-left of the text axes
    ax_txt.text(
        0, 1,
        q_wrapped + "\n" + a_wrapped,
        va="top", ha="left",
        fontsize=9,
        wrap=True
    )

# draw border around the entire figure
# (use figure coordinates so it scales with the canvas)
border = Rectangle(
    (0, 0), 1, 1,
    transform=fig.transFigure,
    fill=False, linewidth=2, edgecolor="black",
    zorder=1000
)
fig.add_artist(border)

fig.subplots_adjust(
    left=0.01,   # reduce left margin
    right=0.99,  # reduce right margin
    top=0.99,    # reduce top margin
    bottom=0.01  # reduce bottom margin
)

plt.savefig(f'./results/report_results.png', 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=300)
plt.close()