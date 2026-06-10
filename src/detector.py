"""
detector.py — Two-stage bin detection pipeline.

Stage 1 (YOLO):       Finds WHERE the bins are in the frame (bounding boxes).
Stage 2 (Classifier): Determines WHAT STATE each bin is in (empty/partial/full).

Why two stages?
  YOLO trained on 120 auto-labeled images struggles to classify bin states.
  EfficientNet classifier is far more accurate with small datasets because
  it only needs to answer "which class is this image?" — not locate objects.

Falls back to classifier-only (whole-frame) when YOLO finds nothing.
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class WasteDetector:
    YOLO_MODEL_PATH       = Path("model/best.pt")
    CLASSIFIER_MODEL_PATH = Path("model/classifier.pt")

    # YOLO confidence kept very low — we only need rough box locations,
    # the classifier re-scores each crop independently.
    YOLO_CONF = 0.05

    def __init__(self,
                 model_path: str | Path = None,
                 conf_threshold: float  = 0.10):
        self.conf_threshold = conf_threshold
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Load YOLO (localization) ──────────────────────────────────────
        yolo_path = Path(model_path) if model_path else self.YOLO_MODEL_PATH
        self._yolo = self._load_yolo(yolo_path)

        # ── Load EfficientNet classifier (state classification) ───────────
        self._clf = self._load_classifier()

        if self._clf:
            print(f"[Detector] Mode: two-stage  (YOLO location + classifier state)")
        else:
            print(f"[Detector] Mode: YOLO-only  (classifier not found — run train_classifier.py)")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_yolo(self, path: Path):
        if not path.exists():
            # Fall back to base YOLOv8n — downloads automatically, works without training
            fallback = Path("yolov8n.pt")
            print(f"[Detector] WARNING: '{path}' not found — using base yolov8n.pt (untrained)")
            print(f"[Detector] Train your own model:  python3 tools/train.py")
            m = YOLO(str(fallback))
            m.to(self.device)
            self.model_path = str(fallback)
            return m
        print(f"[Detector] YOLO     : {path}  (device={self.device.upper()})")
        m = YOLO(str(path))
        m.to(self.device)
        return m

    def _load_classifier(self):
        try:
            from src.classifier import BinClassifier
            return BinClassifier(self.CLASSIFIER_MODEL_PATH)
        except FileNotFoundError:
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run the full pipeline on a BGR frame.
        Returns list of dicts:
            [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...]
        """
        if self._clf:
            return self._two_stage(frame)
        return self._yolo_only(frame)

    # ── Two-stage pipeline ───────────────────────────────────────────────────

    def _two_stage(self, frame: np.ndarray) -> list[dict]:
        # Stage 1: YOLO at very low threshold to get bin regions
        boxes = self._yolo_boxes(frame, conf=self.YOLO_CONF)

        # Stage 2: Classify each crop; filter by user confidence threshold
        raw = self._clf.classify_frame(frame, boxes if boxes else None)
        return [d for d in raw if d["confidence"] >= self.conf_threshold]

    def _yolo_boxes(self, frame: np.ndarray, conf: float) -> list[list[int]]:
        results = self._yolo.predict(
            source=frame, conf=conf, device=self.device, verbose=False
        )
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
        return boxes

    # ── YOLO-only fallback ───────────────────────────────────────────────────

    def _yolo_only(self, frame: np.ndarray) -> list[dict]:
        results = self._yolo.predict(
            source=frame, conf=self.conf_threshold,
            device=self.device, verbose=False
        )
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label  = self._yolo.names.get(cls_id, f"class_{cls_id}")
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "box":        [x1, y1, x2, y2],
                })
        return detections
