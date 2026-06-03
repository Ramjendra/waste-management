"""
tools/train_classifier.py — Fine-tune EfficientNet-B0 for bin state classification.

Why this works better than YOLO with 120 images:
  - Image classification needs far less data than object detection
  - No bounding box quality dependency — just the image and its class label
  - EfficientNet pre-trained on ImageNet already understands textures/shapes
  - 120 images with augmentation trains a reliable 3-class classifier

Classes:   0=empty   1=partial   2=full
Input:     data/raw/empty_*.jpg, data/raw/partial_*.jpg, data/raw/full_*.jpg
Output:    model/classifier.pt

Run:
    python3 tools/train_classifier.py
    python3 tools/train_classifier.py --epochs 60 --batch 16
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image

RAW_DIR    = Path("data/raw")
MODEL_DEST = Path("model/classifier.pt")
CLASSES    = ["empty", "partial", "full"]
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int], transform=None):
        self.paths     = image_paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_data(raw_dir: Path):
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_imgs = sorted([
            p for p in raw_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and p.stem.lower().startswith(cls_name)
        ])
        if not cls_imgs:
            print(f"[classifier] WARNING: no images found for class '{cls_name}'")
        for p in cls_imgs:
            paths.append(p)
            labels.append(cls_idx)
    return paths, labels


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 3) -> nn.Module:
    """EfficientNet-B0 with the classification head replaced for 3 classes."""
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except AttributeError:
        # Older torchvision fallback
        model = models.efficientnet_b0(pretrained=True)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == lbls).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int,   default=50)
    p.add_argument("--batch",  type=int,   default=8)
    p.add_argument("--lr",     type=float, default=1e-3)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[classifier] Device  : {device}")
    print(f"[classifier] Epochs  : {args.epochs}")
    print(f"[classifier] Batch   : {args.batch}")

    # Data
    paths, labels = load_data(RAW_DIR)
    if not paths:
        print(f"[ERROR] No images found in {RAW_DIR}/")
        sys.exit(1)

    print(f"[classifier] Images  : {len(paths)} "
          f"({labels.count(0)} empty, {labels.count(1)} partial, {labels.count(2)} full)")

    # Train/val split — 80/20
    n_val   = max(1, int(len(paths) * 0.20))
    n_train = len(paths) - n_val

    full_ds = BinDataset(paths, labels)
    train_ds_raw, val_ds_raw = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap with separate transforms
    train_ds = BinDataset(
        [paths[i] for i in train_ds_raw.indices],
        [labels[i] for i in train_ds_raw.indices],
        transform=TRAIN_TF,
    )
    val_ds = BinDataset(
        [paths[i] for i in val_ds_raw.indices],
        [labels[i] for i in val_ds_raw.indices],
        transform=VAL_TF,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    print(f"[classifier] Train   : {len(train_ds)}  |  Val: {len(val_ds)}\n")

    # Model
    model     = build_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state   = None

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("─" * 55)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        marker = " ← best" if v_acc > best_val_acc else ""
        print(f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>8.1%}  {v_loss:>9.4f}  {v_acc:>7.1%}{marker}")

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best checkpoint
    MODEL_DEST.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "classes":    CLASSES,
        "val_acc":    best_val_acc,
    }, MODEL_DEST)

    print(f"\n[classifier] Best val accuracy : {best_val_acc:.1%}")
    print(f"[classifier] Saved to          : {MODEL_DEST.resolve()}")
    print("\nNext step:")
    print("  streamlit run streamlit_app.py")
    print("  python3 -m src.main --mode webcam\n")


if __name__ == "__main__":
    main()
