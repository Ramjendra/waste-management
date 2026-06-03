"""
display.py — Frame annotation utilities.

draw_detections()  — per-detection bounding box + label, color-coded by bin state
draw_decision()    — top-of-frame SLA banner with bin count summary
"""

import cv2
import numpy as np
from src.business_logic import Decision, _bin_state


FONT           = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS  = 2
FONT_SCALE     = 0.55
FONT_THICKNESS = 1
BANNER_HEIGHT  = 48


# BGR colors per bin state — consistent with the decision banner
_BOX_COLOR = {
    "empty":   (0,   200, 0),      # green
    "partial": (200, 120, 0),      # blue
    "full":    (0,   140, 255),    # orange
    None:      (180, 180, 180),    # grey — unknown class
}


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes color-coded by bin state with confidence label."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label  = det["label"]
        conf   = det["confidence"]
        state  = _bin_state(label)
        color  = _BOX_COLOR[state]
        text   = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

    return frame


def draw_decision(frame: np.ndarray, decision: Decision) -> np.ndarray:
    """Render a colored SLA banner across the top with bin state counts."""
    h, w = frame.shape[:2]
    banner = np.zeros((BANNER_HEIGHT, w, 3), dtype=np.uint8)
    banner[:] = decision.color

    line1 = f"  {decision.status.value}"
    line2 = f"  Bins: {decision.summary}   |   Total: {decision.detection_count}   |   Max Conf: {decision.max_confidence:.2f}"

    cv2.putText(banner, line1, (8, 16), FONT, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, line2, (8, 38), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, frame])


def annotate(frame: np.ndarray, detections: list[dict], decision: Decision) -> np.ndarray:
    """Full pipeline: color-coded boxes + decision banner."""
    frame = draw_detections(frame, detections)
    frame = draw_decision(frame, decision)
    return frame
