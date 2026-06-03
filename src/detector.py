"""
detector.py — YOLO model loader and inference engine.

Wraps Ultralytics YOLO with GPU support and a clean detect() interface
that the rest of the pipeline calls.
"""

import torch
from ultralytics import YOLO
from pathlib import Path


class WasteDetector:
    DEFAULT_MODEL = Path("model/best.pt")

    def __init__(self, model_path: str | Path = None, conf_threshold: float = 0.25):
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'.\n"
                "Place your YOLOv8 weights at model/best.pt, or pass --model <path>.\n"
                "To download a pretrained baseline: yolo export model=yolov8n.pt"
            )
        print(f"[Detector] Loading model from '{self.model_path}' on {self.device.upper()}")
        model = YOLO(str(self.model_path))
        model.to(self.device)
        return model

    def detect(self, frame):
        """
        Run inference on a single BGR frame (numpy array).

        Returns a list of dicts:
            [{"label": str, "confidence": float, "box": [x1, y1, x2, y2]}, ...]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                })

        return detections
