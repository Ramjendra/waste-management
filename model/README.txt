Place your YOLOv8 weights file here as:   best.pt

Options:
  1. Use your own trained model exported with:
       yolo export model=runs/detect/train/weights/best.pt format=torchscript

  2. Use the pretrained YOLOv8n baseline for quick testing:
       from ultralytics import YOLO
       YOLO("yolov8n.pt")      # auto-downloads to ~/.ultralytics/
       # then copy / symlink to model/best.pt

GPU note:
  The system auto-detects CUDA via torch.cuda.is_available().
  No extra configuration is required.
