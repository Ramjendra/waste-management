"""
streamlit_app.py — Streamlit demo UI for the waste detection system.

Run with:
    streamlit run streamlit_app.py
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.detector import WasteDetector
from src import business_logic as bl
from utils.display import annotate


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Detection System",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Real-Time Waste Detection System")
st.markdown("Powered by **YOLOv8** · Upload a file or start your webcam to begin.")


# ── Sidebar — model settings ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_path  = st.text_input("Model path", value="model/best.pt")

    # Default 0.10 — freshly trained models often score below 0.25
    conf_thresh = st.slider("Confidence threshold", 0.01, 0.95, 0.10, 0.01,
                            help="Lower this if bins are not detected. Try 0.05–0.15 for a new model.")

    st.markdown("---")
    st.markdown("**SLA decision logic**")
    st.markdown("🟢 **Empty bin** → No Action Required")
    st.markdown("🔵 **Partial bin** → Pickup OK")
    st.markdown("🟠 **Full bin** → Warning: Bin Full")
    st.markdown("🔴 **Unknown / low conf** → Manual Review")
    st.markdown("---")
    st.caption("Tip: if nothing is detected, lower the confidence slider.")


# ── Model loader (cached per session) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_detector(path: str, conf: float) -> WasteDetector:
    return WasteDetector(model_path=path, conf_threshold=conf)


def get_detector() -> WasteDetector | None:
    try:
        return load_detector(model_path, conf_thresh)
    except FileNotFoundError as e:
        st.error(str(e))
        return None


# ── Raw detections helper (debug mode) ────────────────────────────────────────
def _raw_detections(detector: WasteDetector, frame: np.ndarray) -> list[dict]:
    """Run inference at near-zero threshold to see ALL raw scores."""
    from src.detector import WasteDetector as WD
    tmp = WD.__new__(WD)
    tmp.model_path   = detector.model_path
    tmp.conf_threshold = 0.01
    tmp.device       = detector.device
    tmp.model        = detector.model
    return tmp.detect(frame)


# ── Helper: process one frame ─────────────────────────────────────────────────
def process_frame(detector: WasteDetector, frame: np.ndarray):
    detections = detector.detect(frame)
    decision   = bl.evaluate(detections)
    annotated  = annotate(frame.copy(), detections, decision)
    return annotated, decision, detections


# ── Decision card ─────────────────────────────────────────────────────────────
def _render_decision_card(decision: bl.Decision, detections: list[dict],
                          detector: WasteDetector, frame: np.ndarray):
    color_map = {
        bl.Status.NO_BINS: "info",
        bl.Status.EMPTY:   "success",
        bl.Status.PARTIAL: "success",
        bl.Status.FULL:    "warning",
        bl.Status.REVIEW:  "error",
    }
    render = getattr(st, color_map[decision.status])
    render(
        f"**{decision.status.value}**  "
        f"| Detections: {decision.detection_count}  "
        f"| Max confidence: {decision.max_confidence:.2f}  "
        f"| Labels: {', '.join(decision.labels) or 'none'}"
    )

    # ── "No bins detected" help section ───────────────────────────────────────
    if decision.status == bl.Status.NO_BINS:
        with st.expander("⚠️ Nothing detected — click here to diagnose", expanded=True):
            st.markdown(
                "The model found **no objects** above the current confidence threshold. "
                "Common causes and fixes:"
            )

            # Show what the model actually sees at threshold=0.01
            raw = _raw_detections(detector, frame)
            if raw:
                st.markdown("**Raw model scores** (all detections at threshold = 0.01):")
                rows = [(d["label"], f"{d['confidence']:.3f}") for d in raw]
                st.table({"Label": [r[0] for r in rows],
                          "Confidence": [r[1] for r in rows]})
                best_conf = max(d["confidence"] for d in raw)
                suggested = max(0.01, round(best_conf * 0.8, 2))
                st.info(
                    f"Best raw score: **{best_conf:.3f}**  \n"
                    f"Try setting the confidence slider to **{suggested}** or lower."
                )
            else:
                st.warning(
                    "Even at threshold=0.01, the model found nothing.  \n"
                    "This usually means the model needs more training data or the "
                    "image doesn't match the training distribution."
                )

            st.markdown("**Checklist:**")
            st.markdown("- Lower the **Confidence threshold** slider (try 0.05)")
            st.markdown("- Make sure **model/best.pt** is the trained model, not the demo")
            st.markdown("- If training just finished, check mAP in `runs/bin_detect/train/`")
            st.markdown("- Re-train with more images: `python3 tools/train.py --epochs 150`")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam = st.tabs(["🖼️ Image", "🎞️ Video File", "📷 Webcam"])


# ── Tab 1: Image ───────────────────────────────────────────────────────────────
with tab_image:
    st.subheader("Upload an image for single-frame analysis")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        detector = get_detector()
        if detector:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated, decision, detections = process_frame(detector, frame)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col_orig, col_result = st.columns(2)
            with col_orig:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         caption="Original", use_container_width=True)
            with col_result:
                st.image(annotated_rgb, caption="Detections", use_container_width=True)

            _render_decision_card(decision, detections, detector, frame)


# ── Tab 2: Video File ──────────────────────────────────────────────────────────
with tab_video:
    st.subheader("Offline Video — Bin Detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        detector = get_detector()
        if detector:
            # Save upload to a temp file
            suffix = Path(video_file.name).suffix
            tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix,        delete=False)
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4",        delete=False)
            tmp_in.write(video_file.read())
            tmp_in.flush()
            input_path  = tmp_in.name
            output_path = tmp_out.name
            tmp_in.close()
            tmp_out.close()

            # Read video metadata
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = max(cap.get(cv2.CAP_PROP_FPS), 1)
            cap.release()

            # Video info row
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames",   total_frames)
            c2.metric("FPS",      f"{fps:.1f}")
            c3.metric("Duration", f"{total_frames / fps:.1f}s")

            # Frame-skip control — speeds up processing on long videos
            frame_skip = st.select_slider(
                "Process every N-th frame",
                options=[1, 2, 3, 5, 10],
                value=1,
                help="1 = every frame (best quality). Higher = faster processing.",
            )

            if st.button("▶ Process Video"):
                progress_bar  = st.progress(0, text="Processing…")
                status_text   = st.empty()

                from src.processor import VideoProcessor

                def _progress(frame_idx, total):
                    pct = min((frame_idx + 1) / max(total, 1), 1.0)
                    progress_bar.progress(pct,
                        text=f"Processing frame {frame_idx + 1} / {total}")

                proc = VideoProcessor(detector, show_window=False)
                out_path, log = proc.process_video_file(
                    input_path  = input_path,
                    output_path = output_path,
                    frame_skip  = frame_skip,
                    progress_cb = _progress,
                )

                progress_bar.progress(1.0, text="Done!")
                st.success(f"Processed {len(log)} frames.")

                # ── Play annotated video ──────────────────────────────────
                st.markdown("### Annotated Output")
                with open(out_path, "rb") as f:
                    st.video(f.read())

                # ── Detection summary table ───────────────────────────────
                st.markdown("### Frame-by-Frame Detection Log")

                # Summarise: show only frames that have detections
                detected_log = [r for r in log if r["detections"] > 0]

                if detected_log:
                    col_f, col_s, col_d, col_c, col_l = st.columns([1, 2, 1, 1, 3])
                    col_f.markdown("**Frame**")
                    col_s.markdown("**Status**")
                    col_d.markdown("**Bins**")
                    col_c.markdown("**Conf**")
                    col_l.markdown("**Labels**")

                    for row in detected_log:
                        col_f.write(row["frame"])
                        col_s.write(row["status"])
                        col_d.write(row["detections"])
                        col_c.write(f"{row['max_conf']:.2f}")
                        col_l.write(", ".join(row["labels"]) or "—")
                else:
                    st.warning(
                        "No bins detected in any frame.  \n"
                        "Lower the **Confidence threshold** slider in the sidebar and try again."
                    )

                # ── Summary stats ─────────────────────────────────────────
                if log:
                    st.markdown("### Summary")
                    from collections import Counter
                    status_counts = Counter(r["status"] for r in log)
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Total Frames Analysed", len(log))
                    sc2.metric("Frames with Bins",      len(detected_log))
                    sc3.metric("Most Common Status",
                               max(status_counts, key=status_counts.get))
                    avg_conf = (sum(r["max_conf"] for r in detected_log) /
                                max(len(detected_log), 1))
                    sc4.metric("Avg Confidence", f"{avg_conf:.2f}")


# ── Tab 3: Webcam ──────────────────────────────────────────────────────────────
with tab_webcam:
    st.subheader("Live webcam inference")
    st.warning(
        "Browser-based webcam capture in Streamlit requires an HTTPS connection or localhost. "
        "For raw RTSP/USB streams, use the CLI instead: `python3 -m src.main --mode webcam`"
    )

    cam_image = st.camera_input("📷 Take a snapshot")

    if cam_image:
        detector = get_detector()
        if detector:
            file_bytes = np.frombuffer(cam_image.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated, decision, detections = process_frame(detector, frame)

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Detection result",
                use_container_width=True,
            )
            _render_decision_card(decision, detections, detector, frame)
