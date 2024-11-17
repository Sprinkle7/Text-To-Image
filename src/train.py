import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from data_loader import TextToImageDataset
from config import DATA_PATH, CAPTIONS_FILE, MODEL_SAVE_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE

def train_model():
    os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    DEVICE = "mps" if torch.has_mps else "cpu"
    print(f"Using device: {DEVICE}")

    # Load Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32  # Use float32 for compatibility
    )

    # Move all components to the same device
    pipeline.unet.to(DEVICE)
    pipeline.vae.to(DEVICE)
    pipeline.text_encoder.to(DEVICE)

    # Enable gradient checkpointing
    pipeline.unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=LEARNING_RATE)

    dataset = TextToImageDataset(
        img_dir=DATA_PATH,
        captions_file=CAPTIONS_FILE,
        transform=transforms.Compose([
            transforms.Resize((64, 64)),  # Smaller images
            transforms.ToTensor()
        ])
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(EPOCHS):
        for step, (images, captions) in enumerate(dataloader):
            # Ensure all inputs are moved to the correct device
            images = images.to(DEVICE)

            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            tokenized_captions = tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tokenized_captions.input_ids.to(DEVICE)

            # Encode captions
            text_encoder_output = pipeline.text_encoder(input_ids)
            encoder_hidden_states = text_encoder_output.last_hidden_state.to(DEVICE)

            # Encode images
            latents = pipeline.vae.encode(images).latent_dist.sample().to(DEVICE)
            latents = latents * 0.18215  # Scale for stable diffusion
            noise = torch.randn_like(latents)
            noisy_latents = latents + noise
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.size(0),), device=DEVICE)

            # Forward pass
            unet_output = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states)
            loss = torch.nn.functional.mse_loss(unet_output.sample, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{step}/{len(dataloader)}], Loss: {loss.item()}")

    pipeline.unet.save_pretrained(MODEL_SAVE_PATH)
    print("Training complete!")

if __name__ == "__main__":
    train_model()
