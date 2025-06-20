import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import random
from peft import PeftModel
from bunny_utils.util.mm_utils import tokenizer_image_token
from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale, adjust_hue
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import json
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from io import BytesIO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# function to apply elastic warping on an image
# only on the bounding box part
def elastic_transform(image, mask, alpha=600, sigma=20):
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    H, W = mask_np.shape
    random_state = np.random.RandomState(None)

    dx = ndimage.gaussian_filter((random_state.rand(H, W) * 2 - 1), sigma) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(H, W) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Warp only the masked region
    warped_np = np.copy(image_np)
    for c in range(image_np.shape[2]):
        warped_channel = ndimage.map_coordinates(image_np[..., c], indices, order=1, mode='reflect').reshape(H, W)
        warped_np[..., c] = mask_np * warped_channel + (1 - mask_np) * image_np[..., c]

    result = torch.from_numpy(warped_np).permute(2, 0, 1).unsqueeze(0).to(image.dtype).to(image.device)
    
    return result

# object to convert a PIL image into a Pytorch tensor
to_tensor = transforms.ToTensor()
# object to convert a Pytorch tensor into a PIL image
to_pil = transforms.ToPILImage()

# # open the training data json file
# with open('./data/vlfeedback_llava_10k.json', 'r') as file:
#     data = json.load(file)

# open the image
image = Image.open('./data/test3.png').convert("RGB")

# convert the image to a tensor
image_tensor = to_tensor(image)

# initialize the object detetction model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

# initialize the preprocessor
preprocess = weights.transforms()

# apply preprocessing to the image
batch = [preprocess(image_tensor)]

# get the model predictions
prediction = model(batch)[0]

# get the image dimensions
H, W = image_tensor.shape[1:]

# create a binary mask that determines the region of detected object
mask = torch.zeros((H, W), dtype=torch.float32)
for box in prediction['boxes']:
    x1, y1, x2, y2 = box.int()
    mask[y1:y2, x1:x2] = 1

# apply custom image corruption only on the bounding box
custom_corrupted_image_tensor = elastic_transform(image_tensor, mask)

# convert image tensor back back to PIL Image
custom_corrupted_image = to_pil(custom_corrupted_image_tensor.squeeze())

# figure for the original image and the corrupted image
fig, axes = plt.subplots(1, 2, figsize=(8, 6))
axes = axes.flatten()

# plot the original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# plot the custom corrupted image
axes[1].imshow(custom_corrupted_image)
axes[1].set_title(f"Custom Corruption")
axes[1].axis('off')

# save the images
plt.savefig(f'./results/image_corruption2.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()