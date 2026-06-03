"""
main.py — CLI entry point for the waste detection system.

Usage:
    python -m src.main --mode webcam
    python -m src.main --mode video  --path video.mp4
    python -m src.main --mode image  --path photo.jpg
    python -m src.main --mode rtsp   --path rtsp://user:pass@host:554/stream

Optional:
    --model model/best.pt   # override model path
    --conf  0.25            # minimum detection confidence
"""

import argparse
import sys

from src.detector import WasteDetector
from src.processor import VideoProcessor


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time Waste Detection System (YOLOv8)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["webcam", "video", "image", "rtsp"],
        help="Input source mode.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="File or stream URL (required for video / image / rtsp modes).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to YOLO .pt weights (default: model/best.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25).",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()

    try:
        detector  = WasteDetector(model_path=args.model, conf_threshold=args.conf)
        processor = VideoProcessor(detector=detector, show_window=True)

        if args.mode == "image":
            processor.run_image(path=args.path)
        else:
            processor.run_stream(mode=args.mode, path=args.path)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
