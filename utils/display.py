"""
display.py — Frame annotation utilities.

draw_detections()  — prominent per-bin bounding box with state badge
draw_decision()    — top-of-frame SLA banner with bin count summary
"""

import cv2
import numpy as np
from src.business_logic import Decision, _bin_state


FONT          = cv2.FONT_HERSHEY_SIMPLEX
BANNER_HEIGHT = 56

# Per-state colours (BGR)
_COLOR = {
    "empty":   (34,  197, 94),    # green
    "partial": (234, 179, 8),     # yellow
    "full":    (239, 68,  68),    # red
    None:      (156, 163, 175),   # grey — unknown
}

# Human-readable badge text per state
_BADGE = {
    "empty":   "EMPTY BIN",
    "partial": "PARTIAL BIN",
    "full":    "FULL BIN",
    None:      "UNKNOWN",
}


def _draw_filled_box(frame: np.ndarray, x1: int, y1: int,
                     x2: int, y2: int, color: tuple, alpha: float = 0.15):
    """Draw a semi-transparent filled rectangle over the bin region."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_label_badge(frame: np.ndarray, x1: int, y1: int,
                      text: str, color: tuple):
    """
    Draw a solid pill-shaped badge above the bounding box.
    Two lines: bin state on top, confidence below.
    """
    lines = text.split("\n")
    font_scale   = 0.65
    font_thick   = 2
    pad          = 8

    # Measure the widest line
    sizes = [cv2.getTextSize(l, FONT, font_scale, font_thick)[0] for l in lines]
    badge_w = max(s[0] for s in sizes) + pad * 2
    line_h  = max(s[1] for s in sizes)
    badge_h = line_h * len(lines) + pad * (len(lines) + 1)

    bx1 = x1
    by2 = y1 - 4
    by1 = by2 - badge_h
    bx2 = bx1 + badge_w

    # Clamp so badge doesn't go off-screen top
    if by1 < 0:
        shift = -by1
        by1  += shift
        by2  += shift

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

    for i, line in enumerate(lines):
        ty = by1 + pad + line_h + i * (line_h + pad)
        cv2.putText(frame, line, (bx1 + pad, ty),
                    FONT, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw per-detection bounding boxes color-coded by bin state.
    Each box has:
      - Semi-transparent fill
      - Thick colored border
      - Badge showing: BIN STATE + confidence score
    """
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf  = det["confidence"]
        state = _bin_state(label)
        color = _COLOR[state]
        badge = _BADGE[state]

        # Semi-transparent fill
        _draw_filled_box(frame, x1, y1, x2, y2, color, alpha=0.20)

        # Thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

        # Corner accent squares for visibility
        corner = 12
        cv2.rectangle(frame, (x1, y1), (x1 + corner, y1 + corner), color, -1)
        cv2.rectangle(frame, (x2 - corner, y1), (x2, y1 + corner), color, -1)
        cv2.rectangle(frame, (x1, y2 - corner), (x1 + corner, y2), color, -1)
        cv2.rectangle(frame, (x2 - corner, y2 - corner), (x2, y2), color, -1)

        # Badge: state + confidence on two lines
        badge_text = f"{badge}\nConf: {conf:.2f}"
        _draw_label_badge(frame, x1, y1, badge_text, color)

    return frame


def draw_decision(frame: np.ndarray, decision: Decision) -> np.ndarray:
    """Solid SLA banner across the top of the frame."""
    h, w = frame.shape[:2]
    banner = np.zeros((BANNER_HEIGHT, w, 3), dtype=np.uint8)
    banner[:] = decision.color

    line1 = f"  {decision.status.value}"
    line2 = (f"  Bins detected: {decision.detection_count}"
             f"   |   {decision.summary}"
             f"   |   Max conf: {decision.max_confidence:.2f}")

    cv2.putText(banner, line1, (10, 22), FONT, 0.70,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, line2, (10, 46), FONT, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, frame])


def annotate(frame: np.ndarray, detections: list[dict], decision: Decision) -> np.ndarray:
    """Full pipeline: draw boxes then overlay the decision banner."""
    frame = draw_detections(frame, detections)
    frame = draw_decision(frame, decision)
    return frame
