from model import get_model
from diffusers import StableDiffusionPipeline
import torch
from config import MODEL_SAVE_PATH, GENERATED_IMAGES_PATH, DEVICE

def generate_image(prompt):
    # Load the fine-tuned model
    model = StableDiffusionPipeline.from_pretrained(MODEL_SAVE_PATH)
    model.to(DEVICE)

    # Generate an image
    with torch.autocast(DEVICE):
        image = model(prompt).images[0]

    # Save the generated image
    image.save(f"{GENERATED_IMAGES_PATH}/generated_image.png")
    print("Image saved to:", f"{GENERATED_IMAGES_PATH}/generated_image.png")

if __name__ == "__main__":
    prompt = input("Enter a text prompt: ")
    generate_image(prompt)
