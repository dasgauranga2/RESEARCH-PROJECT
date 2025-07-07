from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
sam = sam_model_registry["vit_b"](checkpoint="Segment Anything/sam_vit_b_01ec64.pth").to(device)

# load the image 
image = Image.open("mDPO/data/test3.png").convert("RGB")
image_np = np.array(image) 

# automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# generate mask for the image
masks = mask_generator.generate(image_np)
print(masks)

image_tensor = F.to_tensor(image_np).mul(255).to(torch.uint8)

# Select top masks
top_masks = masks[:5]
mask_list = [torch.from_numpy(m['segmentation']) for m in top_masks]

# Stack masks into (N, H, W)
mask_tensor = torch.stack(mask_list).bool()

# Draw the masks (random colors)
image_with_masks = draw_segmentation_masks(image_tensor.cpu(), mask_tensor.cpu(), alpha=0.5, colors=['blue']*5)

# Show using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_with_masks.permute(1, 2, 0))  # Convert to (H, W, C)
plt.axis('off')
plt.title(f"Top {5} SAM Masks (Torchvision)")
plt.savefig(f'Segment Anything/segment.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()