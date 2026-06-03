"""
setup_model.py — Download a bin-detection model without a Roboflow key.

Three options (run this script and pick one):

  Option 1 — Kaggle dataset  (best bin classes, free Kaggle account)
  Option 2 — TACO dataset    (public waste dataset, no account needed)
  Option 3 — YOLOv8n demo    (COCO pretrained, runs immediately, no bin classes)

Usage:
    python setup_model.py --option 1   # Kaggle
    python setup_model.py --option 2   # TACO / public
    python setup_model.py --option 3   # Quick demo (default)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_DEST = Path("model/best.pt")


def already_exists() -> bool:
    if MODEL_DEST.exists():
        print(f"[setup] model/best.pt already exists. Delete it first to re-download.")
        return True
    return False


# ── Option 3: YOLOv8n pretrained (fastest, no account) ────────────────────────
def download_demo_model():
    """
    Downloads the official YOLOv8n COCO weights (~6 MB).
    Classes: person, car, bottle, cup, etc. — NOT bin states.
    Use this only to verify the pipeline runs end-to-end.
    """
    print("\n[Option 3] Downloading YOLOv8n demo weights (COCO, ~6 MB)...")
    try:
        from ultralytics import YOLO
        m = YOLO("yolov8n.pt")          # auto-downloads to ~/.ultralytics cache
        shutil.copy(m.ckpt_path, MODEL_DEST)
        print(f"[setup] Saved to {MODEL_DEST}")
        print("[setup] NOTE: This model does NOT know empty/partial/full bins.")
        print("              Use Option 1 or 2 to get a real bin-detection model.\n")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# ── Option 1: Kaggle — Waste Bin Detection dataset ────────────────────────────
def download_kaggle_model():
    """
    Downloads a YOLOv8-compatible bin detection dataset from Kaggle,
    then trains a small model (~5 min on GPU, ~20 min on CPU).

    Steps done automatically:
      1. Install kaggle CLI
      2. Download dataset
      3. Train YOLOv8n for 30 epochs
      4. Copy best.pt → model/best.pt
    """
    print("\n[Option 1] Kaggle bin-detection setup")
    print("─" * 50)
    print("PREREQUISITES (one-time, free):")
    print("  1. Create account at https://www.kaggle.com")
    print("  2. Go to  Account → API → Create New Token")
    print("  3. Place the downloaded kaggle.json at:")
    print("       Linux/Mac:  ~/.kaggle/kaggle.json")
    print("       Windows:    C:\\Users\\<you>\\.kaggle\\kaggle.json")
    print("─" * 50)

    proceed = input("\nHave you placed kaggle.json? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborted. Complete the steps above then re-run.")
        return

    # Install kaggle package if missing
    try:
        import kaggle
    except ImportError:
        print("[setup] Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        import kaggle

    dataset = "mostafaabla/garbage-classification"
    out_dir = Path("kaggle_data")

    print(f"\n[setup] Downloading dataset: {dataset}")
    try:
        import kaggle as kg
        kg.api.authenticate()
        kg.api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
        print(f"[setup] Downloaded to {out_dir}/")
    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        print("Check your kaggle.json and internet connection.")
        sys.exit(1)

    # Build a minimal data.yaml for training
    yaml_path = out_dir / "data.yaml"
    if not yaml_path.exists():
        _write_garbage_yaml(out_dir, yaml_path)

    print("\n[setup] Training YOLOv8n for 30 epochs (this may take a few minutes)...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model.train(data=str(yaml_path), epochs=30, imgsz=640, project="runs/bin")

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, MODEL_DEST)
        print(f"\n[setup] Trained model saved to {MODEL_DEST}")
        _print_classes(MODEL_DEST)
    else:
        print(f"[ERROR] Training finished but best.pt not found at {best}")
        sys.exit(1)


def _write_garbage_yaml(data_dir: Path, yaml_path: Path):
    """Write a basic data.yaml if the dataset doesn't include one."""
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val" if (data_dir / "val").exists() else train_dir

    content = f"""path: {data_dir.resolve()}
train: {train_dir.name}
val:   {val_dir.name}

nc: 3
names: ['empty', 'partial', 'full']
"""
    yaml_path.write_text(content)
    print(f"[setup] Created {yaml_path}")


# ── Option 2: TACO public dataset ─────────────────────────────────────────────
def download_taco_model():
    """
    TACO (Trash Annotations in Context) is a public waste dataset.
    It contains litter/trash classes (not bin states), but it is freely
    downloadable with no account.

    Website : http://tacodataset.org
    GitHub  : https://github.com/pedropro/TACO

    This function clones the repo and downloads the images automatically.
    After download it trains YOLOv8n as a waste detector.
    """
    print("\n[Option 2] TACO public waste dataset")
    print("  Classes: bag, bottle, can, cup, carton, straw, etc.")
    print("  Note: bin state (empty/full) is NOT in TACO.")
    print("  Use this if you want general waste/litter detection.\n")

    taco_dir = Path("taco_dataset")
    if not taco_dir.exists():
        print("[setup] Cloning TACO repo...")
        subprocess.check_call(["git", "clone",
                               "https://github.com/pedropro/TACO.git",
                               str(taco_dir)])

    print("[setup] Downloading TACO images (this may take several minutes)...")
    dl_script = taco_dir / "downloader.py"
    if dl_script.exists():
        subprocess.check_call([sys.executable, str(dl_script),
                               "--dataset_path", str(taco_dir / "data"),
                               "--num_threads", "4"])
    else:
        print("[ERROR] downloader.py not found in TACO repo.")
        print(f"        Try running manually: cd {taco_dir} && python downloader.py")
        sys.exit(1)

    print("\n[setup] TACO downloaded. To train:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8n.pt')")
    print("  model.train(data='taco_dataset/data.yaml', epochs=50, imgsz=640)")
    print("  Then copy runs/.../best.pt → model/best.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _print_classes(model_path: Path):
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    print(f"\n[setup] Model classes: {m.names}")
    print("        Update business_logic.py label sets if names differ from")
    print("        expected: 'empty', 'partial', 'full'")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download/prepare bin detection model")
    parser.add_argument(
        "--option", type=int, choices=[1, 2, 3], default=3,
        help="1=Kaggle  2=TACO(public)  3=YOLOv8n demo (default)"
    )
    args = parser.parse_args()

    if already_exists():
        return

    if args.option == 1:
        download_kaggle_model()
    elif args.option == 2:
        download_taco_model()
    else:
        download_demo_model()


if __name__ == "__main__":
    main()
