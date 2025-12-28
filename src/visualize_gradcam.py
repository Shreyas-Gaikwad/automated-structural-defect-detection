import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import SDNETDataset
from src.model import CrackClassifier
from src.explain import GradCAM
from src.config import DEVICE, EXPERIMENTS_DIR


def overlay_cam(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay


def main():
    dataset = SDNETDataset(split="test")

    model = CrackClassifier().to(DEVICE)
    model.load_state_dict(
        torch.load(EXPERIMENTS_DIR / "best_model.pth", map_location=DEVICE)
    )
    model.eval()

    gradcam = GradCAM(model, model.backbone.layer4)

    # pick a cracked sample
    for img, label in dataset:
        if label.item() == 1:
            input_tensor = img.unsqueeze(0).to(DEVICE)
            break

    cam = gradcam.generate(input_tensor, class_idx=1)

    image = img.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image = np.uint8(255 * image)

    overlay = overlay_cam(image, cam)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.imshow(cam, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()