import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.dataset import SDNETDataset
from src.model import CrackClassifier
from src.config import BATCH_SIZE, DEVICE, EXPERIMENTS_DIR


@torch.no_grad()
def main():
    test_ds = SDNETDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CrackClassifier().to(DEVICE)
    model.load_state_dict(
        torch.load(EXPERIMENTS_DIR / "best_model.pth", map_location=DEVICE)
    )
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NEGATIVE", "POSITIVE"]))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()