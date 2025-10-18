import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# path of saved checkpoint
checkpoint_path = './checkpoint/mdpo_bunny_8b'

# load the base model
if '8b' in checkpoint_path:
    model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-Llama-3-8B-V',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        'BAAI/Bunny-v1_0-3B',
        torch_dtype=torch.float16, # float32 for cpu
        device_map='auto',
        trust_remote_code=True)

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

# prrompt text
prompt = 'Is the mouse connected to a computer?'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

# load the image
image = Image.open('./data/test2.png')
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

# generate the model outputs
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True,
    repetition_penalty=1.0 # increase this to avoid chattering
)[0]

# display the generated text
print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())