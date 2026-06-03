"""
tools/auto_label.py — Automatically create YOLO labels for downloaded bin images.

Since download_images.py already names files as:
    empty_0001.jpg, partial_0002.jpg, full_0003.jpg

...the class is known from the filename. This script:
  1. Reads every image in data/raw/
  2. Extracts the class from the filename prefix (empty / partial / full)
  3. Uses YOLOv8n to find the most prominent object bounding box
  4. Falls back to the center 80% of the image if YOLO finds nothing
  5. Saves a YOLO-format .txt label in data/labels/

No GUI, no manual work. Run immediately after download_images.py.

Run:
    python3 tools/auto_label.py

After this → skip setup_labeling.py and go straight to:
    python3 tools/prepare_dataset.py
    python3 tools/train.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

RAW_DIR    = Path("data/raw")
LABELS_DIR = Path("data/labels")

# Must match business_logic.py and data.yaml
CLASSES    = ["empty", "partial", "full"]
CLASS_ID   = {name: idx for idx, name in enumerate(CLASSES)}

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

# YOLOv8n confidence for finding the dominant object in the image
DETECT_CONF = 0.20


def pip_install(package: str):
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", package,
        "-q", "--break-system-packages",
    ])


def load_yolo():
    """Load YOLOv8n for object detection (auto-downloaded ~6 MB)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[auto_label] Installing ultralytics...")
        pip_install("ultralytics")
        from ultralytics import YOLO

    print("[auto_label] Loading YOLOv8n for bounding box detection...")
    return YOLO("yolov8n.pt")


def class_from_filename(stem: str) -> str | None:
    """
    Return the bin class encoded in the filename prefix, or None.
    Examples:
        empty_0001  → "empty"
        full_0023   → "full"
        partial_0007 → "partial"
        random_name  → None
    """
    for cls in CLASSES:
        if stem.lower().startswith(cls):
            return cls
    return None


def largest_box(results, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """
    Return the YOLO-format (cx, cy, w, h) — all normalised 0-1 — for
    the largest bounding box found by the detector.
    Returns None if no boxes were detected.
    """
    best_area = 0
    best_box  = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def center_box(margin: float = 0.10) -> tuple[float, float, float, float]:
    """Fallback: bounding box covering the central (1-2*margin) area."""
    size = 1.0 - 2 * margin
    return 0.5, 0.5, size, size


def label_image(img_path: Path, class_id: int, model) -> bool:
    """
    Detect the main object and write a YOLO .txt label.
    Returns True on success.
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [SKIP] Cannot read: {img_path.name}")
        return False

    h, w = frame.shape[:2]

    results = model.predict(source=frame, conf=DETECT_CONF, verbose=False)
    box = largest_box(results, w, h)

    if box is None:
        box = center_box()
        source = "fallback"
    else:
        source = "yolo"

    cx, cy, bw, bh = box
    label_line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

    label_path = LABELS_DIR / (img_path.stem + ".txt")
    label_path.write_text(label_line)

    return True


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])

    if not images:
        print(f"[auto_label] No images found in {RAW_DIR}/")
        print(f"             Run  python3 tools/download_images.py  first.")
        sys.exit(1)

    print(f"[auto_label] Found {len(images)} image(s) in {RAW_DIR}/\n")

    # Separate known-class images from unknowns
    known   = [(p, class_from_filename(p.stem)) for p in images]
    unnamed = [p for p, cls in known if cls is None]
    labeled = [(p, cls) for p, cls in known if cls is not None]

    if unnamed:
        print(f"[auto_label] {len(unnamed)} image(s) have no class prefix — will be skipped:")
        for p in unnamed:
            print(f"             {p.name}")
        print()

    if not labeled:
        print("[auto_label] No images with class prefix found.")
        print("             Images must start with:  empty_  partial_  full_")
        sys.exit(1)

    model = load_yolo()

    counts  = {"empty": 0, "partial": 0, "full": 0}
    skipped = 0

    print(f"\n[auto_label] Labeling {len(labeled)} images...\n")

    for img_path, cls in labeled:
        already = LABELS_DIR / (img_path.stem + ".txt")
        if already.exists():
            counts[cls] += 1
            continue

        ok = label_image(img_path, CLASS_ID[cls], model)
        if ok:
            counts[cls] += 1
        else:
            skipped += 1

    # Summary
    total = sum(counts.values())
    print(f"\n[auto_label] Done — {total} labels written, {skipped} skipped.\n")
    print("─" * 40)
    for cls, n in counts.items():
        print(f"  {cls:8s}  {n:3d} images")
    print("─" * 40)

    if total < 10:
        print("\n[WARNING] Very few labeled images.")
        print("          Run  python3 tools/download_images.py --per-class 60  for more.\n")
    else:
        print("\nNext steps:")
        print("  python3 tools/prepare_dataset.py")
        print("  python3 tools/train.py\n")


if __name__ == "__main__":
    main()
