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

# # 1. The image contains four traffic lights. With two of them towards the left and the other two towards the right.
# # load the original image
# org_image = Image.open('./data/test/count1.jpg')
# width, height = org_image.size
# # get the left half of the image
# left_half = org_image.crop((0, 0, width // 2, height))
# # get the right half of the image
# right_half = org_image.crop((width // 2, 0, width, height))

# for image, image_type in zip([org_image, left_half, right_half], ["Original", "Left-Half", "Right-Half"]):
#     # prompt text
#     prompt = 'How many traffic lights are there in the image?'
#     text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
#     text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
#     input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

#     # generate the model outputs
#     output_ids = model.generate(
#         input_ids,
#         images=image_tensor,
#         max_new_tokens=100,
#         use_cache=True,
#         repetition_penalty=1.0 # increase this to avoid chattering
#     )[0]

#     # display the generated text
#     output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

#     print(f"{image_type}: {output_text}")


# # 2. The image shows six zebras in a jungle. 
# # load the original image
# org_image = Image.open('./data/test/count3.jpg')
# width, height = org_image.size
# # divide the image into three halves such that each portion contains two zebras
# first_half = org_image.crop((0, 0, width // 2.7, height))
# second_half = org_image.crop((width // 2.7, 0, (2*width) // 2.7, height))
# third_half = org_image.crop(((2*width) // 2.7, 0, width, height))

# for image, image_type in zip([org_image, first_half, second_half, third_half], ["Original", "First-Half", "Second-Half", "Third-Half"]):
#     # prompt text
#     prompt = 'How many zebras are there in the image?'
#     text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
#     text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
#     input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

#     # generate the model outputs
#     output_ids = model.generate(
#         input_ids,
#         images=image_tensor,
#         max_new_tokens=100,
#         use_cache=True,
#         repetition_penalty=1.0 # increase this to avoid chattering
#     )[0]

#     # display the generated text
#     output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

#     print(f"{image_type}: {output_text}")


# # 3. The image shows three players playing football. 
# # load the original image
# org_image = Image.open('./data/test/count5.jpg')
# width, height = org_image.size
# # divide the image into the left and right half
# left_half = org_image.crop((0, 0, (17*width) // 20, height))
# right_half = org_image.crop(((17*width) // 20, 0, width, (3*height) // 10))

# for image, image_type in zip([org_image, left_half, right_half], ["Original", "Left-Half", "Right-Half"]):
#     # prompt text
#     prompt = 'How many people are there in the image?'
#     text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
#     text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
#     input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

#     # generate the model outputs
#     output_ids = model.generate(
#         input_ids,
#         images=image_tensor,
#         max_new_tokens=100,
#         use_cache=True,
#         repetition_penalty=1.0 # increase this to avoid chattering
#     )[0]

#     # display the generated text
#     output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

#     print(f"{image_type}: {output_text}")


# # 4. The image shows four chairs around a dining table. 
# # load the original image
# org_image = Image.open('./data/test/count5.jpg')
# width, height = org_image.size
# # divide the image into the left and right half
# left_half = org_image.crop((0, 0, (2*width) // 5, height))
# right_half = org_image.crop(((2*width) // 5, 0, width, height))

# for image, image_type in zip([org_image, left_half, right_half], ["Original", "Left-Half", "Right-Half"]):
#     # prompt text
#     prompt = 'How many chairs are there in the image?'
#     text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
#     text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
#     input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

#     # generate the model outputs
#     output_ids = model.generate(
#         input_ids,
#         images=image_tensor,
#         max_new_tokens=100,
#         use_cache=True,
#         repetition_penalty=1.0 # increase this to avoid chattering
#     )[0]

#     # display the generated text
#     output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

#     print(f"{image_type}: {output_text}")


# # 5. The image shows four chairs around a dining table. 
# # load the original image
# org_image = Image.open('./data/test/count6.jpg')
# width, height = org_image.size
# # divide the image into the left and right half
# left_half = org_image.crop((0, 0, (11*width) // 20, height))
# right_half = org_image.crop(((11*width) // 20, 0, width, height))

# for image, image_type in zip([org_image, left_half, right_half], ["Original", "Left-Half", "Right-Half"]):
#     # prompt text
#     prompt = 'How many chairs are there in the image?'
#     text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
#     text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
#     input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

#     image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

#     # generate the model outputs
#     output_ids = model.generate(
#         input_ids,
#         images=image_tensor,
#         max_new_tokens=100,
#         use_cache=True,
#         repetition_penalty=1.0 # increase this to avoid chattering
#     )[0]

#     # display the generated text
#     output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

#     print(f"{image_type}: {output_text}")


# 6. The image shows three pillows on a sofa. 
# load the original image
org_image = Image.open('./data/test/count7.jpg')
width, height = org_image.size
# divide the image into three halves each with one pillow
first_half = org_image.crop((0, 0, (4*width) // 10, height))
second_half = org_image.crop(((4*width) // 10, 0, (6*width) // 10, height))
third_half = org_image.crop(((6*width) // 10, 0, width, height))

for image, image_type in zip([org_image, first_half, second_half, third_half], ["Original", "First-Half", "Second-Half", "Third-Half"]):
    # prompt text
    prompt = 'How many pillows are there in the image?'
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)

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
    output_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"{image_type}: {output_text}")