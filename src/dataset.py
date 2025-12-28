import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import DATA_DIR, IMAGE_SIZE


class SDNETDataset(Dataset):
    def __init__(self, split: str = "train"):
        """
        Args:
            split (str): one of ['train', 'val', 'test']
        """
        assert split in ["train", "val", "test"], "Invalid split name"

        self.split = split
        self.root_dir = DATA_DIR / split

        self.samples = []
        self._load_samples()

        self.transform = self._build_transforms()

    def _load_samples(self):
        """
        Loads image paths and labels into memory.
        """
        class_map = {
            "NEGATIVE": 0,
            "POSITIVE": 1
        }

        for class_name, label in class_map.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing directory: {class_dir}")

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = class_dir / img_name
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

    def _build_transforms(self):
        """
        Builds image transformations.
        """
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)