from diffusers import StableDiffusionPipeline
import torch

def get_model(model_path=None, device="mps"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path if model_path else "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    return pipe
