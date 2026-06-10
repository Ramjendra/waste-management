"""
tools/setup_model.py — Download a working model so the app runs immediately.

This script sets up model/best.pt and model/classifier.pt so Streamlit
works without needing to train from scratch first.

Strategy (in order of preference):
  1. If model/best.pt already exists → keep it (trained model wins)
  2. Download yolov8n.pt (Ultralytics, auto-download, ~6 MB)
     → copy to model/best.pt as a usable fallback
  3. Build a minimal EfficientNet-B0 classifier with random weights
     → saved to model/classifier.pt as a placeholder until training

Run:
    python3 tools/setup_model.py
    python3 tools/setup_model.py --force   # overwrite even if models exist
"""

import argparse
import shutil
import sys
from pathlib import Path

MODEL_DIR       = Path("model")
YOLO_DEST       = MODEL_DIR / "best.pt"
CLASSIFIER_DEST = MODEL_DIR / "classifier.pt"
CLASSES         = ["empty", "partial", "full"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing models")
    return p.parse_args()


# ── YOLO fallback ─────────────────────────────────────────────────────────────

def setup_yolo(force: bool):
    if YOLO_DEST.exists() and not force:
        print(f"[setup] YOLO model already exists: {YOLO_DEST}  (use --force to replace)")
        return

    print("[setup] Downloading YOLOv8n base model (~6 MB) …")
    try:
        from ultralytics import YOLO
        MODEL_DIR.mkdir(exist_ok=True)

        # ultralytics auto-downloads yolov8n.pt to its cache on first use
        model = YOLO("yolov8n.pt")

        # Find where ultralytics cached it
        cached = Path("yolov8n.pt")
        if not cached.exists():
            import torch
            hub_dir = Path(torch.hub.get_dir()) / "ultralytics" / "assets" / "yolov8n.pt"
            cached  = hub_dir

        if cached.exists():
            shutil.copy(cached, YOLO_DEST)
            print(f"[setup] YOLO model saved → {YOLO_DEST}")
        else:
            # Just save via ultralytics export
            model.save(str(YOLO_DEST))
            print(f"[setup] YOLO model saved → {YOLO_DEST}")

    except Exception as e:
        print(f"[setup] WARNING: Could not save YOLO model: {e}")
        print("[setup] The app will still auto-download yolov8n.pt on first run.")


# ── EfficientNet classifier placeholder ──────────────────────────────────────

def setup_classifier(force: bool):
    if CLASSIFIER_DEST.exists() and not force:
        print(f"[setup] Classifier already exists: {CLASSIFIER_DEST}  (use --force to replace)")
        return

    print("[setup] Building EfficientNet-B0 classifier placeholder …")
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        MODEL_DIR.mkdir(exist_ok=True)

        try:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
        except AttributeError:
            model = models.efficientnet_b0(pretrained=True)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, len(CLASSES)),
        )

        torch.save({
            "state_dict": model.state_dict(),
            "classes":    CLASSES,
            "val_acc":    0.0,
            "note":       "placeholder — run tools/train_classifier.py for real training",
        }, CLASSIFIER_DEST)

        print(f"[setup] Classifier placeholder saved → {CLASSIFIER_DEST}")
        print("[setup] NOTE: This is an untrained placeholder.")
        print("[setup]       Run  python3 tools/train_classifier.py  to train properly.")

    except Exception as e:
        print(f"[setup] WARNING: Could not build classifier: {e}")
        print("[setup] Install:  pip install torch torchvision")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)

    print("\n── Waste Detection Model Setup ─────────────────────────\n")

    setup_yolo(args.force)
    setup_classifier(args.force)

    print("\n── Status ───────────────────────────────────────────────")
    yolo_ok = YOLO_DEST.exists()
    clf_ok  = CLASSIFIER_DEST.exists()
    print(f"  {'✔' if yolo_ok else '✘'}  model/best.pt        (YOLO detector)")
    print(f"  {'✔' if clf_ok  else '✘'}  model/classifier.pt  (EfficientNet classifier)")

    print("""
── Next steps ───────────────────────────────────────────

  Launch the app right now (uses downloaded models):
    streamlit run streamlit_app.py

  Train properly for accurate bin detection:
    python3 tools/train_classifier.py --epochs 60
    python3 tools/train.py --epochs 100

  On GPU server (trains everything automatically):
    ./train_gpu.sh

─────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
