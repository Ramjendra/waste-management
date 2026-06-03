#!/usr/bin/env bash
# =============================================================================
# start.sh — One-shot startup script for the Waste Detection System
#
# Usage (after cloning the repo):
#   chmod +x start.sh
#   ./start.sh              # full setup + train + launch Streamlit
#   ./start.sh --skip-train # skip training, launch Streamlit only
#   ./start.sh --train-only # train only, no Streamlit
#
# What it does:
#   1. Detects Python & GPU (CUDA)
#   2. Creates a virtual environment
#   3. Installs the right PyTorch (GPU or CPU)
#   4. Installs all requirements
#   5. Trains YOLOv8 on the bin dataset  (skipped if model/best.pt exists)
#   6. Launches the Streamlit app
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
  echo "  ╔══════════════════════════════════════════════╗"
  echo "  ║      Waste Bin Detection System (YOLOv8)     ║"
  echo "  ║         GPU-Ready Startup Script             ║"
  echo "  ╚══════════════════════════════════════════════╝"
  echo -e "${RESET}"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_TRAIN=false
TRAIN_ONLY=false
STREAMLIT_PORT=8501
EPOCHS=100
BATCH=16

for arg in "$@"; do
  case $arg in
    --skip-train)  SKIP_TRAIN=true ;;
    --train-only)  TRAIN_ONLY=true ;;
    --port=*)      STREAMLIT_PORT="${arg#*=}" ;;
    --epochs=*)    EPOCHS="${arg#*=}" ;;
    --batch=*)     BATCH="${arg#*=}" ;;
    --help|-h)
      echo "Usage: ./start.sh [options]"
      echo "  --skip-train       Skip training, launch Streamlit only"
      echo "  --train-only       Train only, do not launch Streamlit"
      echo "  --epochs=N         Training epochs (default: 100)"
      echo "  --batch=N          Batch size (default: 16)"
      echo "  --port=N           Streamlit port (default: 8501)"
      exit 0 ;;
  esac
done

# ── Step 0: Banner ────────────────────────────────────────────────────────────
banner

# Make sure we run from the repo root (where start.sh lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Working directory: $SCRIPT_DIR"

# ── Step 1: Python check ──────────────────────────────────────────────────────
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

[[ -z "$PYTHON" ]] && fail "Python 3.9+ not found. Install it and re-run."
log "Using Python $($PYTHON --version) → $(command -v $PYTHON)"

# ── Step 2: GPU / CUDA detection ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 2: GPU Detection ─────────────────────────────────${RESET}"

CUDA_VERSION=""
USE_GPU=false

if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" 2>/dev/null || true)
  if [[ -n "$GPU_NAME" ]]; then
    USE_GPU=true
    log "GPU detected: $GPU_NAME"
    log "CUDA version: $CUDA_VERSION"
  fi
else
  warn "nvidia-smi not found — will use CPU."
fi

[[ "$USE_GPU" == false ]] && warn "No GPU detected. Training will be slower on CPU."

# ── Step 3: Virtual environment ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 3: Virtual Environment ───────────────────────────${RESET}"

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at .venv ..."
  $PYTHON -m venv "$VENV_DIR"
  log "Virtual environment created."
else
  log "Virtual environment already exists."
fi

# Activate venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"

$PIP install --upgrade pip --quiet

# ── Step 4: Install PyTorch (GPU or CPU) ──────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 4: PyTorch Installation ──────────────────────────${RESET}"

# Check if torch is already installed and matches GPU mode
TORCH_INSTALLED=false
if $PYTHON_VENV -c "import torch" &>/dev/null 2>&1; then
  TORCH_GPU=$($PYTHON_VENV -c "import torch; print(torch.cuda.is_available())")
  if [[ "$USE_GPU" == true && "$TORCH_GPU" == "True" ]]; then
    log "PyTorch (GPU) already installed."
    TORCH_INSTALLED=true
  elif [[ "$USE_GPU" == false && "$TORCH_GPU" == "False" ]]; then
    log "PyTorch (CPU) already installed."
    TORCH_INSTALLED=true
  else
    warn "PyTorch installed but GPU mode mismatch — reinstalling."
  fi
fi

if [[ "$TORCH_INSTALLED" == false ]]; then
  if [[ "$USE_GPU" == true ]]; then
    # Pick torch index based on CUDA version
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
      info "Installing PyTorch for CUDA 12.x ..."
    elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu118"
      info "Installing PyTorch for CUDA 11.8 ..."
    else
      TORCH_INDEX="https://download.pytorch.org/whl/cu117"
      info "Installing PyTorch for CUDA 11.7 ..."
    fi

    $PIP install torch torchvision --index-url "$TORCH_INDEX" --quiet
  else
    info "Installing PyTorch (CPU only) ..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
  fi
  log "PyTorch installed."
fi

# ── Step 5: Install project requirements ─────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 5: Project Dependencies ──────────────────────────${RESET}"

# Install everything except torch (already handled above)
$PIP install ultralytics opencv-python streamlit Pillow numpy icrawler --quiet
log "All dependencies installed."

# ── Step 6: Verify GPU inside Python ──────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 6: Environment Check ─────────────────────────────${RESET}"

$PYTHON_VENV - <<'EOF'
import torch, cv2, ultralytics, streamlit
print(f"  torch       {torch.__version__}")
print(f"  cuda        {'available — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU mode)'}")
print(f"  opencv      {cv2.__version__}")
print(f"  ultralytics {ultralytics.__version__}")
print(f"  streamlit   {streamlit.__version__}")
EOF
log "All imports OK."

# ── Step 7: Training ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Step 7: Model Training ────────────────────────────────${RESET}"

if [[ "$SKIP_TRAIN" == true ]]; then
  warn "--skip-train passed. Skipping training."
elif [[ -f "model/best.pt" ]]; then
  log "model/best.pt already exists — skipping training."
  warn "To retrain: delete model/best.pt and re-run ./start.sh"
else
  if [[ ! -f "data/dataset/data.yaml" ]]; then
    fail "data/dataset/data.yaml not found. Dataset is missing from the repo."
  fi

  info "Starting GPU training: epochs=$EPOCHS  batch=$BATCH"
  echo ""

  $PYTHON_VENV tools/train.py --epochs "$EPOCHS" --batch "$BATCH"

  if [[ -f "model/best.pt" ]]; then
    log "Training complete — model/best.pt saved."
  else
    fail "Training finished but model/best.pt not found. Check logs above."
  fi
fi

# ── Step 8: Train classifier (EfficientNet) ───────────────────────────────────
echo ""
echo -e "${BOLD}── Step 8: Classifier Training (EfficientNet-B0) ─────────${RESET}"

if [[ "$SKIP_TRAIN" == true ]]; then
  warn "--skip-train passed. Skipping classifier training."
elif [[ -f "model/classifier.pt" ]]; then
  log "model/classifier.pt already exists — skipping classifier training."
else
  info "Training EfficientNet-B0 classifier on bin images..."
  $PYTHON_VENV tools/train_classifier.py --epochs 50 --batch 8
  [[ -f "model/classifier.pt" ]] && log "Classifier saved → model/classifier.pt" \
    || warn "Classifier training failed — YOLO-only mode will be used."
fi

# ── Step 9: Launch Streamlit ──────────────────────────────────────────────────
if [[ "$TRAIN_ONLY" == true ]]; then
  echo ""
  log "Training done. Streamlit not started (--train-only)."
  echo -e "\nTo launch the app later, run:"
  echo -e "  source .venv/bin/activate"
  echo -e "  streamlit run streamlit_app.py --server.port $STREAMLIT_PORT"
  exit 0
fi

echo ""
echo -e "${BOLD}── Step 9: Launching Streamlit ───────────────────────────${RESET}"
echo ""
echo -e "${GREEN}${BOLD}  App will be available at:"
echo -e "  http://localhost:$STREAMLIT_PORT${RESET}"
echo -e "${CYAN}  Press Ctrl+C to stop.${RESET}"
echo ""

$PYTHON_VENV -m streamlit run streamlit_app.py \
  --server.port "$STREAMLIT_PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
