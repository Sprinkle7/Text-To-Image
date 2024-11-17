# config.py
import torch

# Paths
DATA_PATH = "../data/images"
CAPTIONS_FILE = "../data/captions.csv"
MODEL_SAVE_PATH = "../output/models"
GENERATED_IMAGES_PATH = "../output/images"

# Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
EPOCHS = 5

# Device
DEVICE = "cpu"  # Use GPU, MPS for macOS, or CPU
