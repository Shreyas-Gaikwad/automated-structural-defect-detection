import random
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import SDNETDataset
from src.config import CLASS_NAMES

# Load dataset
dataset = SDNETDataset(split="train")

# Randomly sample indices
indices = random.sample(range(len(dataset)), 10)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for ax, idx in zip(axes, indices):
    img, label = dataset[idx]

    # Convert tensor to numpy for display
    img = img.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    ax.imshow(img)
    ax.set_title(CLASS_NAMES[label.item()])
    ax.axis("off")

plt.tight_layout()
plt.show()