import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TextToImageDataset(Dataset):
    def __init__(self, img_dir, captions_file, transform=None):
        self.img_dir = img_dir  # Directory containing images
        self.captions = pd.read_csv(captions_file)  # Load the CSV with 'filename' and 'caption' columns
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Get the filename and caption
        img_name = self.captions.iloc[idx, 0]  # Ensure this points to the filename column
        caption = self.captions.iloc[idx, 1]  # Ensure this points to the caption column

        # Construct the full path to the image file
        img_path = f"{self.img_dir}/{img_name}"

        # Open the image file
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, caption
