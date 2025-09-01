import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple

from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

# path of model
MODEL_PATH = "OmniGen2/OmniGen2"
# INFERENCE STEPS
INF_STEPS = 50 
# INPUT IMAGE PATH
INPUT_PATH = 'example_images/test3.png'
# 
OUTPUT_PATH =  'outputs/output_edit.png'

def load_pipeline(accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    pipeline = OmniGen2Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            MODEL_PATH,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )


    # optimization options 
    # see Github repo for explanations
    # pipeline.enable_taylorseer = True
    # pipeline.transformer.enable_teacache = True
    # pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    # scheduler options: euler, dpmsolver++
    scheduler_type = 'euler'

    if scheduler_type == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    #pipeline.enable_sequential_cpu_offload()
    pipeline.enable_model_cpu_offload()
    #pipeline = pipeline.to(accelerator.device)

    return pipeline

def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images

def run(accelerator: Accelerator, 
        pipeline: OmniGen2Pipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(0)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=1024, # output image width
        height=1024, # output image height
        num_inference_steps=50, # no. of inference steps
        max_sequence_length=1024,
        text_guidance_scale=5.0, # controls how strictly the output adheres to the text prompt
        image_guidance_scale=2.0, # controls how much the final image should resemble the input reference image
        cfg_range=(0.0, 1.0), # range of CFG
        negative_prompt=negative_prompt,
        num_images_per_prompt=1, # no. of images per prompt
        generator=generator,
        output_type="pil",
    )
    return results

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)

def main() -> None:
    """Main function to run the image generation process."""
    # available data types: 'fp32', 'fp16', 'bf16'
    data_type = 'bf16'

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=data_type if data_type != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if data_type == 'fp16':
        weight_dtype = torch.float16
    elif data_type == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(accelerator, weight_dtype)
    input_images = preprocess(INPUT_PATH)

    # prompt text
    prompt = 'Replace the cat with a dog.'
    # negative prompt text
    # tells the model what you don't want to see in the image
    negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

    # Generate and save image
    results = run(accelerator, pipeline, prompt, negative_prompt, input_images)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if len(results.images) > 1:
        for i, image in enumerate(results.images):
            image_name, ext = os.path.splitext(OUTPUT_PATH)
            image.save(f"{image_name}_{i}{ext}")

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    output_image.save(OUTPUT_PATH)
    print(f"Image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    #root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    main()