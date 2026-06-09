#!/usr/bin/env bash
# =============================================================================
# train_gpu.sh — GPU Training Script for Waste Bin Detection
#
# Run this after cloning the repo on your GPU server:
#   chmod +x train_gpu.sh
#   ./train_gpu.sh
#
# What it does:
#   1. Detects Python / CUDA
#   2. Creates venv + installs GPU PyTorch + all deps
#   3. Downloads internet images (empty/partial/full bins) via icrawler
#   4. Extracts training frames from any .mp4 in data/ (auto-detects state)
#   5. Trains EfficientNet-B0 classifier  → model/classifier.pt
#   6. Trains YOLOv8 detector             → model/best.pt
#   7. Prints launch command for Streamlit
#
# Options:
#   --skip-download      skip icrawler image download (use existing data/raw/)
#   --skip-yolo          skip YOLO training (classifier only)
#   --skip-classifier    skip EfficientNet training (YOLO only)
#   --epochs-yolo N      YOLO training epochs  (default 100)
#   --epochs-clf N       Classifier epochs     (default 60)
#   --batch N            batch size            (default 16)
#   --video PATH         explicit video path to extract frames from
#   --video-state STATE  bin state in video: empty|partial|full (default: full)
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[✔]${RESET} $*"; }
info() { echo -e "${CYAN}[→]${RESET} $*"; }
warn() { echo -e "${YELLOW}[!]${RESET} $*"; }
fail() { echo -e "${RED}[✘]${RESET} $*"; exit 1; }

banner() {
  echo -e "${BOLD}${CYAN}"
  echo "  ╔══════════════════════════════════════════════════════╗"
  echo "  ║       Waste Bin Detection — GPU Training Script      ║"
  echo "  ║   EfficientNet-B0 classifier  +  YOLOv8 detector     ║"
  echo "  ╚══════════════════════════════════════════════════════╝"
  echo -e "${RESET}"
}

# ── Argument defaults ─────────────────────────────────────────────────────────
SKIP_DOWNLOAD=false
SKIP_YOLO=false
SKIP_CLASSIFIER=false
EPOCHS_YOLO=100
EPOCHS_CLF=60
BATCH=16
VIDEO_PATH=""
VIDEO_STATE="full"   # default: assume the provided video shows a FULL bin

for arg in "$@"; do
  case $arg in
    --skip-download)      SKIP_DOWNLOAD=true ;;
    --skip-yolo)          SKIP_YOLO=true ;;
    --skip-classifier)    SKIP_CLASSIFIER=true ;;
    --epochs-yolo=*)      EPOCHS_YOLO="${arg#*=}" ;;
    --epochs-clf=*)       EPOCHS_CLF="${arg#*=}" ;;
    --batch=*)            BATCH="${arg#*=}" ;;
    --video=*)            VIDEO_PATH="${arg#*=}" ;;
    --video-state=*)      VIDEO_STATE="${arg#*=}" ;;
    --help|-h)
      echo "Usage: ./train_gpu.sh [options]"
      echo "  --skip-download        Skip downloading images from internet"
      echo "  --skip-yolo            Skip YOLO training"
      echo "  --skip-classifier      Skip EfficientNet training"
      echo "  --epochs-yolo=N        YOLO epochs         (default 100)"
      echo "  --epochs-clf=N         Classifier epochs   (default 60)"
      echo "  --batch=N              Batch size          (default 16)"
      echo "  --video=PATH           Video to extract training frames from"
      echo "  --video-state=STATE    Bin state in video: empty|partial|full (default: full)"
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
banner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Working directory: $SCRIPT_DIR"

# ── Step 1: Python ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 1: Python ────────────────────────────────────────${RESET}"

PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 9 ]]; then
      PYTHON="$cmd"
      break
    fi
  fi
done
[[ -z "$PYTHON" ]] && fail "Python 3.9+ not found."
log "Python: $($PYTHON --version)  →  $(command -v $PYTHON)"

# ── Step 2: GPU detection ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 2: GPU Detection ─────────────────────────────────${RESET}"

USE_GPU=false
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null || true)
  if [[ -n "$GPU_NAME" ]]; then
    USE_GPU=true
    log "GPU  : $GPU_NAME"
    log "CUDA : $CUDA_VERSION"
  fi
fi
[[ "$USE_GPU" == false ]] && warn "No GPU detected — training will be slow on CPU."

# ── Step 3: Virtual environment ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 3: Virtual Environment ───────────────────────────${RESET}"

VENV_DIR="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating .venv ..."
  $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
$PIP install --upgrade pip --quiet
log "venv ready: $VENV_DIR"

# ── Step 4: PyTorch (GPU or CPU) ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 4: PyTorch ───────────────────────────────────────${RESET}"

TORCH_OK=false
if $PY -c "import torch" &>/dev/null 2>&1; then
  TORCH_GPU=$($PY -c "import torch; print(torch.cuda.is_available())")
  if [[ "$USE_GPU" == true && "$TORCH_GPU" == "True" ]]; then
    log "PyTorch (GPU) already installed."; TORCH_OK=true
  elif [[ "$USE_GPU" == false && "$TORCH_GPU" == "False" ]]; then
    log "PyTorch (CPU) already installed."; TORCH_OK=true
  fi
fi

if [[ "$TORCH_OK" == false ]]; then
  if [[ "$USE_GPU" == true ]]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
      IDX="https://download.pytorch.org/whl/cu121"
    elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
      IDX="https://download.pytorch.org/whl/cu118"
    else
      IDX="https://download.pytorch.org/whl/cu117"
    fi
    info "Installing PyTorch for CUDA ${CUDA_VERSION} ..."
    $PIP install torch torchvision --index-url "$IDX" --quiet
  else
    info "Installing PyTorch (CPU) ..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
  fi
fi

# ── Step 5: Project dependencies ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 5: Project Dependencies ──────────────────────────${RESET}"

$PIP install ultralytics opencv-python streamlit Pillow numpy icrawler --quiet
log "All dependencies installed."

# ── Step 6: Verify imports ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 6: Verify Environment ────────────────────────────${RESET}"

$PY - <<'PYEOF'
import torch, cv2, ultralytics, streamlit, torchvision
print(f"  torch       {torch.__version__}  cuda={'yes — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (CPU)'}")
print(f"  torchvision {torchvision.__version__}")
print(f"  opencv      {cv2.__version__}")
print(f"  ultralytics {ultralytics.__version__}")
print(f"  streamlit   {streamlit.__version__}")
PYEOF
log "All imports OK."

# ── Step 7: Download images (icrawler) ────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 7: Download Training Images ─────────────────────${RESET}"

if [[ "$SKIP_DOWNLOAD" == true ]]; then
  warn "--skip-download passed — using existing data/raw/ images."
else
  info "Downloading bin images (empty / partial / full) via icrawler …"
  $PY tools/download_images.py || warn "Image download had errors — continuing with existing images."
  log "Image download done."
fi

TOTAL_IMGS=$(ls data/raw/*.jpg data/raw/*.png 2>/dev/null | wc -l || echo 0)
log "Images in data/raw/ : $TOTAL_IMGS"

# ── Step 8: Extract frames from video ─────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 8: Extract Video Frames ──────────────────────────${RESET}"

# Auto-detect video if not specified
if [[ -z "$VIDEO_PATH" ]]; then
  for ext in mp4 MP4 avi AVI mov MOV mkv MKV; do
    FOUND=$(find data/ -maxdepth 1 -name "*.${ext}" 2>/dev/null | head -1 || true)
    if [[ -n "$FOUND" ]]; then
      VIDEO_PATH="$FOUND"
      break
    fi
  done
fi

if [[ -n "$VIDEO_PATH" && -f "$VIDEO_PATH" ]]; then
  info "Extracting frames from: $VIDEO_PATH  (state=$VIDEO_STATE)"
  $PY tools/video_to_dataset.py \
    --video  "$VIDEO_PATH" \
    --state  "$VIDEO_STATE" \
    --every  8 \
    --max    200
  log "Frame extraction complete."
else
  warn "No video found in data/ — skipping frame extraction."
  warn "To add video frames later:  python3 tools/video_to_dataset.py --video PATH --state full"
fi

# Count final dataset
EMPTY_CNT=$(ls data/raw/empty*.jpg data/raw/empty*.png 2>/dev/null | wc -l || echo 0)
PARTIAL_CNT=$(ls data/raw/partial*.jpg data/raw/partial*.png 2>/dev/null | wc -l || echo 0)
FULL_CNT=$(ls data/raw/full*.jpg data/raw/full*.png 2>/dev/null | wc -l || echo 0)
TOTAL_CNT=$(( EMPTY_CNT + PARTIAL_CNT + FULL_CNT ))

echo ""
echo -e "  Dataset summary:"
echo -e "    empty   : ${EMPTY_CNT}"
echo -e "    partial : ${PARTIAL_CNT}"
echo -e "    full    : ${FULL_CNT}"
echo -e "    TOTAL   : ${TOTAL_CNT}"

if [[ "$TOTAL_CNT" -lt 10 ]]; then
  fail "Too few images ($TOTAL_CNT). Add more images to data/raw/ first."
fi

# ── Step 9: Train EfficientNet classifier ────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 9: Train EfficientNet-B0 Classifier ─────────────${RESET}"

if [[ "$SKIP_CLASSIFIER" == true ]]; then
  warn "--skip-classifier passed."
elif [[ -f "model/classifier.pt" ]]; then
  warn "model/classifier.pt already exists."
  warn "Delete it and re-run to retrain: rm model/classifier.pt"
else
  info "Training EfficientNet-B0 classifier on $TOTAL_CNT images …"
  info "Epochs: $EPOCHS_CLF   Batch: $BATCH"
  $PY tools/train_classifier.py \
    --epochs "$EPOCHS_CLF" \
    --batch  "$BATCH"
  [[ -f "model/classifier.pt" ]] \
    && log "Classifier saved → model/classifier.pt" \
    || fail "Classifier training failed."
fi

# ── Step 10: Train YOLO ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 10: Train YOLOv8 Detector ───────────────────────${RESET}"

if [[ "$SKIP_YOLO" == true ]]; then
  warn "--skip-yolo passed."
elif [[ -f "model/best.pt" ]]; then
  warn "model/best.pt already exists."
  warn "Delete it and re-run to retrain: rm model/best.pt"
else
  if [[ ! -f "data/dataset/data.yaml" ]]; then
    warn "data.yaml not found — running prepare_dataset.py first …"
    $PY tools/prepare_dataset.py || warn "prepare_dataset.py had errors — check above."
  fi

  info "Training YOLOv8 detector …"
  info "Epochs: $EPOCHS_YOLO   Batch: $BATCH"
  $PY tools/train.py \
    --epochs "$EPOCHS_YOLO" \
    --batch  "$BATCH"
  [[ -f "model/best.pt" ]] \
    && log "YOLO model saved → model/best.pt" \
    || fail "YOLO training failed."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  Training complete!${RESET}"
echo ""
echo -e "  Models saved:"
[[ -f "model/classifier.pt" ]] && echo -e "    ${GREEN}✔${RESET}  model/classifier.pt  (EfficientNet state classifier)"
[[ -f "model/best.pt"       ]] && echo -e "    ${GREEN}✔${RESET}  model/best.pt         (YOLOv8 bin detector)"
echo ""
echo -e "  Launch the Streamlit app:"
echo -e "    ${CYAN}source .venv/bin/activate${RESET}"
echo -e "    ${CYAN}streamlit run streamlit_app.py --server.port 8501${RESET}"
echo ""
echo -e "  Or use the full start script:"
echo -e "    ${CYAN}./start.sh --skip-train${RESET}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════${RESET}"
