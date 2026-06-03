"""
business_logic.py — Class-aware SLA decision engine for bin state detection.

Primary rule: class label (what the model detected)
Fallback rule: confidence threshold (when label is unknown)

Supported Roboflow class name variants — all map to the same 3 bin states:
  EMPTY   → empty_bin, empty, Empty, bin-empty, 0
  PARTIAL → partial_bin, partial, half, half-full, Partial, bin-half, 1
  FULL    → full_bin, full, Full, overflowing, bin-full, 2

SLA mapping:
  EMPTY   → "No Action Required"      (green)
  PARTIAL → "Pickup OK"               (blue)
  FULL    → "Warning: Bin Full"       (orange)
  Unknown + high conf  → "Pickup OK"
  Unknown + med  conf  → "Manual Review Required"
  Unknown + low  conf  → "Manual Review Required"
  No detections        → "No Bins Detected"
"""

from dataclasses import dataclass
from enum import Enum


# ── Confidence fallback thresholds (used when class is unrecognised) ───────────
HIGH_CONF = 0.70
MED_CONF  = 0.40


# ── Label → canonical bin state mapping ───────────────────────────────────────
# Add any new Roboflow class names here without touching the logic below.
_EMPTY_LABELS   = {"empty_bin", "empty", "bin-empty", "0", "Empty"}
_PARTIAL_LABELS = {"partial_bin", "partial", "half", "half-full", "bin-half", "Partial", "1"}
_FULL_LABELS    = {"full_bin", "full", "overflowing", "bin-full", "Full", "2"}


def _bin_state(label: str) -> str | None:
    """Return canonical state string or None if label is not a known bin class."""
    if label in _EMPTY_LABELS:
        return "empty"
    if label in _PARTIAL_LABELS:
        return "partial"
    if label in _FULL_LABELS:
        return "full"
    return None


# ── Status enum ───────────────────────────────────────────────────────────────
class Status(str, Enum):
    NO_BINS  = "No Bins Detected"
    EMPTY    = "No Action Required"       # bin is empty
    PARTIAL  = "Pickup OK"                # bin is partially filled
    FULL     = "Warning: Bin Full"        # bin needs pickup
    REVIEW   = "Manual Review Required"   # low confidence / unknown class


# Severity order (higher index = more urgent action needed)
SEVERITY = [Status.NO_BINS, Status.EMPTY, Status.PARTIAL, Status.REVIEW, Status.FULL]


# ── Decision dataclass ────────────────────────────────────────────────────────
@dataclass
class Decision:
    status: Status
    max_confidence: float
    detection_count: int
    labels: list[str]
    bin_counts: dict[str, int]   # {"empty": 2, "partial": 1, "full": 1}

    @property
    def color(self) -> tuple[int, int, int]:
        """BGR color for the overlay banner."""
        return {
            Status.NO_BINS:  (80,  80,  80),    # dark grey
            Status.EMPTY:    (0,   200, 0),      # green
            Status.PARTIAL:  (200, 120, 0),      # blue
            Status.FULL:     (0,   140, 255),    # orange
            Status.REVIEW:   (0,   0,   200),    # red
        }[self.status]

    @property
    def summary(self) -> str:
        parts = [f"{v} {k}" for k, v in self.bin_counts.items() if v > 0]
        return ", ".join(parts) if parts else "no bins"


# ── Main evaluation function ──────────────────────────────────────────────────
def evaluate(detections: list[dict]) -> Decision:
    """
    Convert WasteDetector.detect() output into a Decision.

    Each detection is first checked against the class-name table.
    If the label is unrecognised, confidence thresholds act as the fallback.
    The most severe status across all detections is reported.
    """
    if not detections:
        return Decision(
            status=Status.NO_BINS,
            max_confidence=0.0,
            detection_count=0,
            labels=[],
            bin_counts={"empty": 0, "partial": 0, "full": 0},
        )

    worst_status = Status.NO_BINS
    max_conf     = 0.0
    labels       = []
    bin_counts   = {"empty": 0, "partial": 0, "full": 0}

    for det in detections:
        label = det["label"]
        conf  = det["confidence"]
        max_conf = max(max_conf, conf)
        labels.append(label)

        state = _bin_state(label)

        if state == "empty":
            status = Status.EMPTY
            bin_counts["empty"] += 1
        elif state == "partial":
            status = Status.PARTIAL
            bin_counts["partial"] += 1
        elif state == "full":
            status = Status.FULL
            bin_counts["full"] += 1
        else:
            # Unknown class — fall back to confidence thresholds
            status = Status.PARTIAL if conf >= HIGH_CONF else Status.REVIEW

        if SEVERITY.index(status) > SEVERITY.index(worst_status):
            worst_status = status

    return Decision(
        status=worst_status,
        max_confidence=max_conf,
        detection_count=len(detections),
        labels=labels,
        bin_counts=bin_counts,
    )
