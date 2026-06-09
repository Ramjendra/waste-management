"""
tools/video_to_dataset.py — Extract training frames from a video file.

Usage:
    python3 tools/video_to_dataset.py --video data/10061891-hd_1920_1080_24fps.mp4 --state full
    python3 tools/video_to_dataset.py --video data/my_empty_bin.mp4 --state empty --every 5

Arguments:
    --video   : path to source video
    --state   : bin state shown in video  (empty | partial | full)
    --every   : extract every Nth frame   (default: 8)
    --max     : max frames to extract     (default: 200)
    --out     : output directory          (default: data/raw/)

Extracted frames are named:   {state}_video_{NNNN}.jpg
They are automatically picked up by train_classifier.py.
"""

import argparse
import sys
from pathlib import Path

import cv2


RAW_DIR = Path("data/raw")
VALID_STATES = {"empty", "partial", "full"}


def parse_args():
    p = argparse.ArgumentParser(description="Extract training frames from a video")
    p.add_argument("--video",  required=True,      help="Path to the source video")
    p.add_argument("--state",  required=True,      help="Bin state: empty | partial | full")
    p.add_argument("--every",  type=int, default=8, help="Extract every Nth frame (default 8)")
    p.add_argument("--max",    type=int, default=200, help="Max frames to extract (default 200)")
    p.add_argument("--out",    type=str, default=str(RAW_DIR), help="Output directory")
    return p.parse_args()


def extract_frames(video_path: str, state: str, every: int, max_frames: int, out_dir: Path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur     = total / max(fps, 1)

    print(f"\n[video_to_dataset] Video     : {video_path}")
    print(f"[video_to_dataset] State     : {state}")
    print(f"[video_to_dataset] Frames    : {total}  ({dur:.1f}s @ {fps:.1f}fps)")
    print(f"[video_to_dataset] Size      : {width}x{height}")
    print(f"[video_to_dataset] Sampling  : every {every} frames  →  max {max_frames} saved")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Find the next available index so we don't overwrite existing files
    existing = list(out_dir.glob(f"{state}_video_*.jpg"))
    start_idx = len(existing)

    saved    = 0
    frame_no = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % every == 0:
            fname = out_dir / f"{state}_video_{start_idx + saved:04d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

            if saved % 20 == 0 or saved == 1:
                print(f"  Saved {saved} frames …", end="\r", flush=True)

        frame_no += 1

    cap.release()
    print(f"\n[video_to_dataset] Done — {saved} frames saved to {out_dir.resolve()}")
    return saved


def count_dataset(out_dir: Path):
    print("\n[video_to_dataset] Dataset summary:")
    for state in ("empty", "partial", "full"):
        imgs = list(out_dir.glob(f"{state}*.jpg")) + list(out_dir.glob(f"{state}*.png"))
        print(f"  {state:<10}: {len(imgs):>4} images")
    total = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    print(f"  {'TOTAL':<10}: {total:>4} images")


def main():
    args = parse_args()

    state = args.state.lower().strip()
    if state not in VALID_STATES:
        print(f"[ERROR] --state must be one of: {', '.join(VALID_STATES)}")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    saved   = extract_frames(str(video_path), state, args.every, args.max, out_dir)
    count_dataset(out_dir)

    print("\n─── Next steps ──────────────────────────────────────────")
    print("  1. Train the EfficientNet classifier:")
    print("       python3 tools/train_classifier.py --epochs 60 --batch 16")
    print("  2. Train YOLO (if needed):")
    print("       python3 tools/train.py --epochs 100")
    print("  3. Launch the app:")
    print("       streamlit run streamlit_app.py")
    print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
