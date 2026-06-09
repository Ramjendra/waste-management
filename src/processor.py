"""
processor.py — Video/image processing pipeline.

Modes:
    webcam  — live feed (cv2.VideoCapture(0))
    video   — offline file  → annotated output saved to disk
    image   — single image
    rtsp    — RTSP network stream
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

from src.detector import WasteDetector
from src import business_logic as bl
from utils.display import annotate


def _open_capture(mode: str, path: str | None) -> cv2.VideoCapture:
    sources = {"webcam": 0, "video": path, "rtsp": path}
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

    # ── Live stream (webcam / RTSP) ───────────────────────────────────────────
    def run_stream(self, mode: str, path: str | None = None) -> None:
        cap = _open_capture(mode, path)
        print(f"[Processor] Starting stream — mode='{mode}'. Press 'q' to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Processor] Stream ended.")
                    break
                annotated = self._process_frame(frame)
                if self.show_window:
                    cv2.imshow("Waste Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ── Offline video file → saves annotated output ───────────────────────────
    def process_video_file(self,
                           input_path: str,
                           output_path: str | None = None,
                           frame_skip: int = 1,
                           progress_cb=None) -> tuple[str, list[dict]]:
        """
        Process every frame (or every Nth if frame_skip > 1) of an offline
        video, write annotated output to a file, and return a detection log.

        Args:
            input_path   : path to source video
            output_path  : where to save annotated video (auto-generated if None)
            frame_skip   : process 1 in every N frames (1 = all frames)
            progress_cb  : optional callable(frame_idx, total_frames)

        Returns:
            (output_path, log)   where log is a list of per-frame dicts
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: '{input_path}'")

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = max(cap.get(cv2.CAP_PROP_FPS), 1)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Banner is added on top — account for its height
        banner_h = 56
        out_h    = h + banner_h

        if output_path is None:
            src = Path(input_path)
            output_path = str(src.parent / f"{src.stem}_detected{src.suffix}")

        # Use mp4v codec — most compatible with browsers / st.video()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, out_h))

        log      = []
        frame_no = 0
        last_annotated = None

        print(f"[Processor] Video: {total} frames @ {fps:.1f} fps → {output_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % frame_skip == 0:
                detections = self.detector.detect(frame)
                decision   = bl.evaluate(detections)
                last_annotated = annotate(frame.copy(), detections, decision)
                log.append({
                    "frame":      frame_no,
                    "status":     decision.status.value,
                    "detections": decision.detection_count,
                    "max_conf":   round(decision.max_confidence, 3),
                    "labels":     decision.labels,
                })
            else:
                # Reuse previous annotated frame for skipped frames
                if last_annotated is None:
                    last_annotated = annotate(frame.copy(), [], bl.evaluate([]))

            writer.write(last_annotated)

            if progress_cb:
                progress_cb(frame_no, total)

            frame_no += 1

        cap.release()
        writer.release()

        processed = len(log)
        print(f"[Processor] Done — {processed} frames processed, saved to {output_path}")
        return output_path, log

    # ── Single image ──────────────────────────────────────────────────────────
    def run_image(self, path: str) -> np.ndarray:
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
        detections = self.detector.detect(frame)
        decision   = bl.evaluate(detections)
        return annotate(frame, detections, decision)
