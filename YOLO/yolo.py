from ultralytics import YOLO
import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
# load the model
model = YOLO("YOLO/yolo11x-seg.pt")

# path of image
image_path = "mDPO/data/test1.png"

# run inference on image
results = model(image_path)

# load the image
image = Image.open(image_path).convert("RGB")
image_tensor = to_tensor(image).mul(255).to(torch.uint8)  # shape: (C, H, W)

# get the segmentation masks
masks = results[0].masks.data # shape: (N, H, W)

# dimensions of image
_, H, W = image_tensor.shape

# scale the masks to original scale
masks = F.interpolate(
    masks.unsqueeze(1),
    size=(H,W),
    mode='nearest'
).squeeze().bool()

# draw the segmentation masks
image_with_masks = draw_segmentation_masks(image_tensor.cpu(),
                                           masks,
                                           alpha=0.5,
                                           colors=['blue']*masks.shape[0])

# save the image
plt.figure(figsize=(10, 10))
plt.imshow(image_with_masks.permute(1, 2, 0))  # convert to (H, W, C)
plt.axis("off")
plt.title("YOLOv11 Instance Segmentation")
plt.savefig(f'YOLO/segment.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()