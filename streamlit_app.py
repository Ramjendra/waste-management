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
        d = load_detector(model_path, conf_thresh)
        if not Path(model_path).exists():
            st.warning(
                "⚠️ **model/best.pt not found** — using base YOLOv8n (untrained).  \n"
                "Detection results will be poor until you train:  \n"
                "`python3 tools/train.py` then `python3 tools/train_classifier.py`  \n"
                "Or on GPU server: `./train_gpu.sh`"
            )
        return d
    except Exception as e:
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


# ── helpers ────────────────────────────────────────────────────────────────────
import subprocess, shutil as _shutil
from collections import Counter

def _to_h264(src: str, dst: str) -> bool:
    """Re-encode src → dst with H.264 using ffmpeg. Returns True on success."""
    if not _shutil.which("ffmpeg"):
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-vcodec", "libx264", "-crf", "23",
             "-pix_fmt", "yuv420p", dst],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _key_frames(input_path: str, log: list[dict],
                detector, max_show: int = 6) -> list[tuple]:
    """Extract up to max_show annotated key frames that had detections."""
    detected = [r for r in log if r["detections"] > 0]
    if not detected:
        return []
    # Pick evenly spaced frames
    step    = max(1, len(detected) // max_show)
    chosen  = detected[::step][:max_show]
    frames  = []
    cap     = cv2.VideoCapture(input_path)
    for row in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        ret, frame = cap.read()
        if not ret:
            continue
        detections = detector.detect(frame)
        decision   = bl.evaluate(detections)
        annotated  = annotate(frame.copy(), detections, decision)
        frames.append((row["frame"], cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       row["status"]))
    cap.release()
    return frames


# ── Tab 2: Video File ──────────────────────────────────────────────────────────
with tab_video:
    st.subheader("Offline Video — Bin Detection")
    video_file = st.file_uploader(
        "Upload your video",
        type=["mp4", "avi", "mov", "mkv", "MP4", "AVI", "MOV"],
        help="Upload any bin footage — the system will detect empty / partial / full bins in every frame.",
    )

    if video_file:
        detector = get_detector()
        if detector:
            # ── Save uploaded file ────────────────────────────────────────
            suffix  = Path(video_file.name).suffix or ".mp4"
            tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix,  delete=False)
            tmp_raw = tempfile.NamedTemporaryFile(suffix=".mp4",  delete=False)
            tmp_h264= tempfile.NamedTemporaryFile(suffix=".mp4",  delete=False)
            tmp_in.write(video_file.read())
            tmp_in.flush(); tmp_in.close()
            tmp_raw.close(); tmp_h264.close()

            input_path   = tmp_in.name
            raw_out_path = tmp_raw.name
            h264_path    = tmp_h264.name

            # ── Video metadata ────────────────────────────────────────────
            cap          = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = max(cap.get(cv2.CAP_PROP_FPS), 1)
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frames",     total_frames)
            c2.metric("FPS",        f"{fps:.1f}")
            c3.metric("Duration",   f"{total_frames / fps:.1f}s")
            c4.metric("Resolution", f"{width}×{height}")

            frame_skip = st.select_slider(
                "Process every N-th frame",
                options=[1, 2, 3, 5, 10], value=2,
                help="1 = every frame (slow but most accurate). 2-3 = good balance.",
            )

            st.caption(
                f"Estimated frames to analyse: **{total_frames // frame_skip}**  "
                f"— GPU inference ~0.05s/frame"
            )

            if st.button("▶ Process Video", type="primary"):

                # ── Processing ────────────────────────────────────────────
                prog  = st.progress(0, text="Starting…")
                info  = st.empty()

                from src.processor import VideoProcessor

                def _cb(idx, total):
                    pct = min((idx + 1) / max(total, 1), 1.0)
                    prog.progress(pct,
                        text=f"Analysing frame {idx + 1} / {total} …")

                proc = VideoProcessor(detector, show_window=False)
                raw_path, log = proc.process_video_file(
                    input_path  = input_path,
                    output_path = raw_out_path,
                    frame_skip  = frame_skip,
                    progress_cb = _cb,
                )
                prog.progress(1.0, text="Detection done — encoding video…")

                # ── Re-encode to H.264 for browser playback ───────────────
                use_h264 = _to_h264(raw_path, h264_path)
                playback_path = h264_path if use_h264 else raw_path
                prog.empty()

                detected_log = [r for r in log if r["detections"] > 0]
                st.success(
                    f"Done — {len(log)} frames analysed, "
                    f"**{len(detected_log)}** frames had bin detections."
                )

                # ── Annotated video player ────────────────────────────────
                st.markdown("---")
                st.markdown("### Annotated Video")
                with open(playback_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)

                # Download button (always available even if player fails)
                st.download_button(
                    label="⬇ Download Annotated Video",
                    data=video_bytes,
                    file_name=f"{Path(video_file.name).stem}_detected.mp4",
                    mime="video/mp4",
                )

                # ── Key frame preview grid ────────────────────────────────
                st.markdown("---")
                st.markdown("### Key Frames with Detections")
                kf = _key_frames(input_path, log, detector, max_show=6)
                if kf:
                    cols = st.columns(min(len(kf), 3))
                    for i, (fno, img, status) in enumerate(kf):
                        with cols[i % 3]:
                            st.image(img, use_container_width=True,
                                     caption=f"Frame {fno} — {status}")
                else:
                    st.info("No frames with bin detections to preview.")

                # ── Detection log table ───────────────────────────────────
                st.markdown("---")
                st.markdown("### Frame-by-Frame Detection Log")
                if detected_log:
                    import pandas as pd
                    df = pd.DataFrame(detected_log)[
                        ["frame", "status", "detections", "max_conf", "labels"]
                    ]
                    df["labels"] = df["labels"].apply(lambda x: ", ".join(x) if x else "—")
                    df.columns   = ["Frame", "Status", "Bins", "Max Conf", "Labels"]
                    st.dataframe(df, use_container_width=True, height=300)
                else:
                    st.warning(
                        "No bins detected in any frame.  \n"
                        "Try lowering the **Confidence threshold** slider (sidebar) to 0.05."
                    )

                # ── Summary metrics ───────────────────────────────────────
                st.markdown("---")
                st.markdown("### Summary")
                status_counts = Counter(r["status"] for r in log)
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Frames Analysed",  len(log))
                sm2.metric("Frames with Bins", len(detected_log))
                sm3.metric("Most Common Status",
                           max(status_counts, key=status_counts.get)
                           if status_counts else "—")
                avg_conf = (sum(r["max_conf"] for r in detected_log) /
                            max(len(detected_log), 1))
                sm4.metric("Avg Confidence",   f"{avg_conf:.2f}")


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
