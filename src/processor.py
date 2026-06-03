"""
processor.py — Video/image processing pipeline.

Supports four input modes:
    webcam  — live feed from cv2.VideoCapture(0)
    video   — offline video file
    image   — single image file
    rtsp    — RTSP network stream
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from src.detector import WasteDetector
from src import business_logic as bl
from utils.display import annotate


def _open_capture(mode: str, path: str | None) -> cv2.VideoCapture:
    """Return an opened VideoCapture or raise ValueError."""
    sources = {
        "webcam": 0,
        "video":  path,
        "rtsp":   path,
    }
    if mode not in sources:
        raise ValueError(f"Unknown mode '{mode}'. Choose: webcam, video, image, rtsp")
    if mode in ("video", "rtsp") and not path:
        raise ValueError(f"--path is required for mode '{mode}'")

    cap = cv2.VideoCapture(sources[mode])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source for mode='{mode}' path='{path}'")
    return cap


class VideoProcessor:
    def __init__(self, detector: WasteDetector, show_window: bool = True):
        self.detector    = detector
        self.show_window = show_window

    def run_stream(self, mode: str, path: str | None = None) -> None:
        """Process a live or file-based video stream until 'q' is pressed."""
        cap = _open_capture(mode, path)
        print(f"[Processor] Starting stream — mode='{mode}'. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Processor] Stream ended or frame unavailable.")
                    break

                annotated = self._process_frame(frame)

                if self.show_window:
                    cv2.imshow("Waste Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Processor] Quit requested.")
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_image(self, path: str) -> np.ndarray:
        """Process a single image file and display the result."""
        img_path = Path(path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: '{path}'")

        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"cv2 could not read image: '{path}'")

        annotated = self._process_frame(frame)

        if self.show_window:
            cv2.imshow("Waste Detection — Image", annotated)
            print("[Processor] Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Core pipeline: detect → decide → annotate."""
        detections = self.detector.detect(frame)
        decision   = bl.evaluate(detections)
        return annotate(frame, detections, decision)
