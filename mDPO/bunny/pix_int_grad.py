import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
from scipy import ndimage

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

# set device
device = 'cuda'
torch.set_default_device(device)

# CALCULATE PIXEL-INTENSITY GRADIENTS FOR ORIGINLA AND CORRUPTED IMAGES

# load the reference model
reference_model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-v1_0-3B',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)

# # load the mdpo model
# mdpo_model = AutoModelForCausalLM.from_pretrained(
#     'BAAI/Bunny-v1_0-3B',
#     torch_dtype=torch.float16, # float32 for cpu
#     device_map='auto',
#     trust_remote_code=True)

# path of saved checkpoint
checkpoint_path = './checkpoint/mdpo_bunny'
# determine if LoRA adapter weights should be used
use_lora = True

# if use_lora:
#     mdpo_model = PeftModel.from_pretrained(
#         mdpo_model,
#         checkpoint_path
#     )

#     mdpo_model = mdpo_model.merge_and_unload()

# load the vision towers
reference_model.get_vision_tower().load_model()
#mdpo_model.get_vision_tower().load_model()

# set model to evaluation mode
reference_model.eval()
#mdpo_model.eval()

# print(mdpo_model.get_vision_tower().is_loaded)
# print(reference_model.get_vision_tower().is_loaded)

# load the model tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True)

# function to corrupt an image like mDPO
def crop_image(image):
    resize_cropper = v2.RandomResizedCrop(size=image.size()[-2:], scale=(0.01, 0.2))
    image = resize_cropper(image.squeeze(0)).unsqueeze(0)
    return image

# function to process an image
def prepare_image(img_path, model):

    #print('./data/merged_images/' + img_path)
    image = Image.open(img_path)
    # process the image into a tensor
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype) # (1,C,W,H)

    return image_tensor

# image path
image_path = './data/merged_images/llava-reasoning-3730.jpg'

# get the image tensor
image = prepare_image(image_path, reference_model)

# function to calculate the pixel-intensity gradients of the image and display stats
def pixel_intensity_gradients(image_tensor):
    image_tensor = image_tensor.squeeze()

    # convert rgb tensor to grayscale tensor
    gray_tensor = rgb_to_grayscale(image_tensor).squeeze().cpu()

    # convert to NumPy array
    image_array = np.array(gray_tensor, dtype=np.float32)

    # compute gradients along the x and y axes
    sobel_x = ndimage.sobel(image_array, axis=0, mode='reflect')
    sobel_y = ndimage.sobel(image_array, axis=1, mode='reflect')

    # compute gradient magnitude
    gradient_magnitude = np.hypot(sobel_x, sobel_y)

    # normalize gradient magnitude to range 0-255
    gradient_magnitude *= 255.0 / np.max(gradient_magnitude)

    print(f"Min Gradient: {np.min(gradient_magnitude)}")
    print(f"Max Gradient: {np.max(gradient_magnitude)}")
    print(f"Mean Gradient: {np.mean(gradient_magnitude)}")
    print(f"Median Gradient: {np.median(gradient_magnitude)}")
    print(f"Variance Gradient: {np.var(gradient_magnitude)}")

pixel_intensity_gradients(image)
print("\n\n")

for _ in range(5):
    # get the corruped image
    corrupted_image = crop_image(image)
    pixel_intensity_gradients(corrupted_image)
    print("\n")