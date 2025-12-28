from pathlib import Path
import torch

# ============================
# Project Paths
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "SDNET"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create experiments directory if it doesn't exist
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Dataset Parameters
# ============================

IMAGE_SIZE = 224
NUM_CLASSES = 2  # NEGATIVE, POSITIVE

CLASS_NAMES = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

# ============================
# Training Parameters
# ============================

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ============================
# Reproducibility
# ============================

SEED = 42

# ============================
# Device Configuration
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")