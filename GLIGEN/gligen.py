from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes
import os
import json
import random
import bbdraw
import matplotlib.pyplot as plt

#os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# load image processor
processor = AutoImageProcessor.from_pretrained("facebook/deformable-detr-detic")
# load object detection model
model = DeformableDetrForObjectDetection.from_pretrained("facebook/deformable-detr-detic").to("cuda")

# open the training data json file
with open('mDPO/data/vlfeedback_llava_10k.json', 'r') as file:
    data = json.load(file)

# list to store original images
orig_images = []
# list to store images with bounding boxes
bb_images = []
# path to your font file
font_path = "GLIGEN/roboto.ttf"

# RANDOMLY SAMPLE SOME IMAGES FROM THE DATASET
# iterate through the data
for sample in random.sample(data, 6):
    # load the image
    image = Image.open('mDPO/data/merged_images/' + sample['img_path']).convert("RGB")

    # preprocess the input image
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    # run inference
    outputs = model(**inputs)

    # get original image height and width
    target_sizes = torch.tensor([image.size[::-1]])
    # get results
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # list of bounding boxes
    boxes = []
    # list of bounding box labels
    labels = []

    # iterate through each detected result
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

        # get coordinates of bounding box
        x1, y1, x2, y2 = box.tolist()

        # object category
        class_name = model.config.id2label[label.item()]

        # confidence score
        conf = float(score.item())

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        labels.append(f"{class_name} {conf:.2f}")

    if boxes:
        # draw the bounding box on the image
        bb_image = draw_bounding_boxes(
            image=pil_to_tensor(image.copy()),                      # uint8 CHW
            boxes=torch.tensor(boxes, dtype=torch.int64),                    # int pixel coords
            labels=labels,                    # optional text per box
            colors="red",                          # or list of colors
            width=3,                               # line thickness
            font_size=20,                           # needs a TrueType font available
            #font=font_path
        )
        bb_image = to_pil_image(bb_image)

        orig_images.append(image)
        bb_images.append(bb_image)
    else:
        print("NO OBJECTS DETECTED")

# figure for the original image and the custom images
fig, axes = plt.subplots(len(orig_images), 2, figsize=(10, 12))
axes = axes.flatten()

for i in range(len(orig_images)):
    # plot the original image
    axes[(i*2)].imshow(orig_images[i])
    axes[(i*2)].set_title("Original Image")
    axes[(i*2)].axis('off')

    # plot the image with bounding box
    axes[(i*2)+1].imshow(bb_images[i])
    axes[(i*2)+1].set_title("Bounding Box Image")
    axes[(i*2)+1].axis('off')

# save the images
plt.savefig(f'mDPO/results/ie_custom_images.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()