"""
tools/train.py — Train YOLOv8n on your labeled bin dataset.

Run after prepare_dataset.py:
    python tools/train.py

Optional flags:
    --epochs  50        (default 50, increase for more accuracy)
    --imgsz   640       (image size, 640 is standard)
    --batch   8         (lower if you run out of GPU memory)
    --model   yolov8n   (n=nano, s=small, m=medium — bigger = more accurate but slower)

GPU:
    Automatically uses CUDA if available. Falls back to CPU (slower but works).

Output:
    runs/bin_detect/train/weights/best.pt  ← your trained model
    Automatically copied to model/best.pt  when training is done.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


YAML_PATH  = Path("data/dataset/data.yaml")
MODEL_DEST = Path("model/best.pt")
RUN_DIR    = Path("runs/bin_detect")


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on bin detection dataset")
    p.add_argument("--epochs", type=int,   default=50)
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--batch",  type=int,   default=8)
    p.add_argument("--model",  type=str,   default="yolov8n",
                   help="yolov8n | yolov8s | yolov8m")
    return p.parse_args()


def check_dataset():
    if not YAML_PATH.exists():
        print(f"[ERROR] Dataset not found at {YAML_PATH}")
        print("        Run  python tools/prepare_dataset.py  first.")
        sys.exit(1)

    train_imgs = list((Path("data/dataset/train/images")).glob("*"))
    print(f"[train] Training images: {len(train_imgs)}")

    if len(train_imgs) == 0:
        print("[ERROR] No training images found. Run prepare_dataset.py first.")
        sys.exit(1)


def main():
    args = parse_args()

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n[train] Device : {'GPU (CUDA)' if device == '0' else 'CPU'}")
    print(f"[train] Model  : {args.model}")
    print(f"[train] Epochs : {args.epochs}")
    print(f"[train] ImgSz  : {args.imgsz}")
    print(f"[train] Batch  : {args.batch}\n")

    check_dataset()

    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data    = str(YAML_PATH.resolve()),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = device,
        project = str(RUN_DIR),
        name    = "train",
        exist_ok= True,
        patience= 15,       # stop early if no improvement for 15 epochs
        verbose = True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"

    if best_pt.exists():
        shutil.copy(best_pt, MODEL_DEST)
        print(f"\n[train] Training complete!")
        print(f"[train] Best weights → {MODEL_DEST.resolve()}")
        _print_metrics(results)
        _print_next_steps()
    else:
        print(f"[ERROR] Training finished but best.pt not found at {best_pt}")
        sys.exit(1)


def _print_metrics(results):
    try:
        box = results.results_dict
        print(f"\n[train] Final metrics:")
        print(f"        mAP50     : {box.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"        mAP50-95  : {box.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
        print(f"        Precision : {box.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"        Recall    : {box.get('metrics/recall(B)', 'N/A'):.3f}")
    except Exception:
        pass


def _print_next_steps():
    print("\n" + "─" * 50)
    print("NEXT STEPS — run your waste detection system:")
    print()
    print("  Webcam:")
    print("    python -m src.main --mode webcam")
    print()
    print("  Video file:")
    print("    python -m src.main --mode video --path video.mp4")
    print()
    print("  Image:")
    print("    python -m src.main --mode image --path photo.jpg")
    print()
    print("  Streamlit UI:")
    print("    streamlit run streamlit_app.py")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    main()
