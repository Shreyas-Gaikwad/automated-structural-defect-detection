from collections import Counter
from src.dataset import SDNETDataset


def analyze_split(split):
    dataset = SDNETDataset(split=split)
    labels = [label.item() for _, label in dataset]
    counts = Counter(labels)

    total = len(labels)
    print(f"\nSplit: {split}")
    for cls, count in counts.items():
        pct = 100 * count / total
        print(f"  Class {cls}: {count} images ({pct:.2f}%)")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        analyze_split(split)