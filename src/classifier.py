"""
classifier.py — EfficientNet-B0 bin state classifier.

Two-stage pipeline:
  Stage 1: YOLO detects bin bounding boxes  (location)
  Stage 2: Classifier determines bin state  (empty / partial / full)

This approach outperforms YOLO-only detection with small datasets because:
  - Classification needs much less data than detection
  - State classification is the hard part; localisation is easier

Usage:
    clf = BinClassifier()
    state, conf = clf.classify(crop)   # crop = BGR numpy array
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from PIL import Image

CLASSIFIER_PATH = Path("model/classifier.pt")
CLASSES         = ["empty", "partial", "full"]

INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class BinClassifier:
    def __init__(self, model_path: str | Path = None):
        self.model_path = Path(model_path) if model_path else CLASSIFIER_PATH
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes    = CLASSES
        self.model      = self._load()

    def _load(self) -> nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at '{self.model_path}'.\n"
                "Run:  python3 tools/train_classifier.py"
            )
        ckpt = torch.load(self.model_path, map_location=self.device)

        # Rebuild same architecture
        try:
            model = models.efficientnet_b0(weights=None)
        except TypeError:
            model = models.efficientnet_b0(pretrained=False)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, len(CLASSES)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()

        saved_acc = ckpt.get("val_acc", 0)
        print(f"[Classifier] Loaded '{self.model_path}'  (val acc {saved_acc:.1%})")
        return model

    def classify(self, bgr_crop: np.ndarray) -> tuple[str, float]:
        """
        Classify a BGR numpy crop (from YOLO box or full frame).
        Returns (class_name, confidence).
        """
        import cv2
        rgb   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        tensor = INFER_TF(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        conf, idx = probs.max(0)
        return self.classes[idx.item()], conf.item()

    def classify_frame(self, frame: np.ndarray,
                       boxes: list[list[int]] | None = None
                       ) -> list[dict]:
        """
        Classify all boxes in a frame.
        If boxes is None, classifies the whole frame as one detection.

        Returns list of dicts compatible with business_logic.evaluate():
            [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...]
        """
        h, w = frame.shape[:2]

        if not boxes:
            # No YOLO boxes — classify the whole frame
            label, conf = self.classify(frame)
            return [{"label": label, "confidence": conf, "box": [0, 0, w, h]}]

        results = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            label, conf = self.classify(crop)
            results.append({"label": label, "confidence": conf,
                            "box": [x1, y1, x2, y2]})
        return results
