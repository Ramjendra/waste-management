"""
tools/setup_labeling.py — Install labelme and open it for bin labeling.

labelme is used instead of LabelImg because LabelImg has known
compatibility issues with Python 3.12+.

Run:
    python3 tools/setup_labeling.py

labelme controls:
  Create Rectangle  → Edit menu → Create Rectangle  (or press Ctrl+R)
  Save              → Ctrl+S
  Next image        → D  (or Next Image button)
  Delete box        → Delete key

NOTE: If you want to skip manual labeling entirely (recommended for
downloaded images), run the auto-labeler instead:
    python3 tools/auto_label.py
"""

import subprocess
import sys
from pathlib import Path


RAW_DIR      = Path("data/raw")
LABELS_DIR   = Path("data/labels")
CLASSES_FILE = Path("data/classes.txt")
CLASSES      = ["empty", "partial", "full"]


def pip_install(package: str):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", package,
        "-q", "--break-system-packages",
    ])


def setup_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Images dir  → {RAW_DIR.resolve()}")
    print(f"[setup] Labels dir  → {LABELS_DIR.resolve()}")


def write_classes():
    CLASSES_FILE.write_text("\n".join(CLASSES) + "\n")
    print(f"[setup] Classes     → {CLASSES}")


def install_labelme():
    try:
        import labelme  # noqa: F401
        print("[setup] labelme already installed.")
    except ImportError:
        print("[setup] Installing labelme (Python 3.12 compatible)...")
        pip_install("labelme")
        print("[setup] labelme installed.")


def open_labelme():
    imgs = list(RAW_DIR.glob("*.[jJpP][pPnN][gG]")) + list(RAW_DIR.glob("*.jpeg"))

    if not imgs:
        print(f"\n[WARNING] No images in {RAW_DIR}/")
        print(f"          Run  python3 tools/download_images.py  first.")
        return

    print(f"\n[setup] Found {len(imgs)} image(s). Opening labelme...\n")
    print("─" * 55)
    print("HOW TO LABEL IN LABELME:")
    print("  1.  Ctrl+R (or Edit → Create Rectangle)")
    print("  2.  Draw a box around the bin")
    print("  3.  Type the label:  empty  OR  partial  OR  full")
    print("  4.  Ctrl+S to save")
    print("  5.  Ctrl+D for next image")
    print()
    print("TIP: labelme saves .json files — the converter in")
    print("     prepare_dataset.py handles them automatically.")
    print("─" * 55)
    print("\nWhen done → run:  python3 tools/prepare_dataset.py\n")

    subprocess.Popen([
        sys.executable, "-m", "labelme",
        str(RAW_DIR.resolve()),
        "--labels", ",".join(CLASSES),
        "--autosave",
        "--output", str(LABELS_DIR.resolve()),
    ])


if __name__ == "__main__":
    setup_dirs()
    write_classes()
    install_labelme()
    open_labelme()
