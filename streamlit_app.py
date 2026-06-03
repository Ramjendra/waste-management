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
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    st.markdown("---")
    st.markdown("**SLA decision logic**")
    st.markdown("🟢 **Empty bin** → No Action Required")
    st.markdown("🔵 **Partial bin** → Pickup OK")
    st.markdown("🟠 **Full bin** → Warning: Bin Full")
    st.markdown("🔴 **Unknown / low conf** → Manual Review")


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


# ── Helper: process one frame and return annotated image + decision ────────────
def process_frame(detector: WasteDetector, frame: np.ndarray):
    detections = detector.detect(frame)
    decision   = bl.evaluate(detections)
    annotated  = annotate(frame.copy(), detections, decision)
    return annotated, decision


# ── Decision card helper ───────────────────────────────────────────────────────
def _render_decision_card(decision: bl.Decision):
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

            annotated, decision = process_frame(detector, frame)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col_orig, col_result = st.columns(2)
            with col_orig:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with col_result:
                st.image(annotated_rgb, caption="Detections", use_container_width=True)

            _render_decision_card(decision)


# ── Tab 2: Video File ──────────────────────────────────────────────────────────
with tab_video:
    st.subheader("Upload a video for offline processing")
    video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        detector = get_detector()
        if detector:
            with tempfile.NamedTemporaryFile(suffix=Path(video_file.name).suffix, delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25

            st.info(f"Video: {total_frames} frames @ {fps:.1f} fps")

            frame_placeholder = st.empty()
            decision_placeholder = st.empty()
            progress = st.progress(0)
            stop_btn = st.button("⏹ Stop")

            frame_idx = 0
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, decision = process_frame(detector, frame)
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                decision_placeholder.info(
                    f"Frame {frame_idx + 1}/{total_frames}  —  **{decision.status.value}**  "
                    f"| Detections: {decision.detection_count}  "
                    f"| Max conf: {decision.max_confidence:.2f}"
                )
                progress.progress(min((frame_idx + 1) / max(total_frames, 1), 1.0))
                frame_idx += 1
                time.sleep(1.0 / fps)  # approximate playback speed

            cap.release()
            st.success("Video processing complete." if not stop_btn else "Stopped.")


# ── Tab 3: Webcam ──────────────────────────────────────────────────────────────
with tab_webcam:
    st.subheader("Live webcam inference")
    st.warning(
        "Browser-based webcam capture in Streamlit requires an HTTPS connection or localhost. "
        "For raw RTSP/USB streams, use the CLI instead: `python -m src.main --mode webcam`"
    )

    cam_image = st.camera_input("📷 Take a snapshot")

    if cam_image:
        detector = get_detector()
        if detector:
            file_bytes = np.frombuffer(cam_image.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated, decision = process_frame(detector, frame)

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Detection result",
                use_container_width=True,
            )
            _render_decision_card(decision)
