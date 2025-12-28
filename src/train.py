import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from src.dataset import SDNETDataset
from src.model import CrackClassifier
from src.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    EXPERIMENTS_DIR
)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return running_loss / len(loader), precision, recall, f1


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return running_loss / len(loader), precision, recall, f1


def main():
    train_ds = SDNETDataset(split="train")
    val_ds = SDNETDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CrackClassifier().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_p, train_r, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_p, val_r, val_f1 = validate(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train F1: {train_f1:.3f} | Val F1: {val_f1:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                model.state_dict(),
                EXPERIMENTS_DIR / "best_model.pth"
            )

    print(f"Best Validation F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()