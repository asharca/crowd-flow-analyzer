#!/bin/sh
set -e

WEIGHTS_DIR=/app/model_weights
mkdir -p "$WEIGHTS_DIR"

# Download YOLO model weights on first run
if [ ! -f "$WEIGHTS_DIR/yolo11m.pt" ] || [ ! -f "$WEIGHTS_DIR/yolo11n.pt" ]; then
    echo "First run: downloading YOLO model weights..."
    uv run --no-sync python -c "
import os
os.chdir('$WEIGHTS_DIR')
from ultralytics import YOLO
YOLO('yolo11m.pt')
YOLO('yolo11n.pt')
os.chdir('/app')
"
fi

# Symlink into /app so app code can find them by relative name
ln -sf "$WEIGHTS_DIR/yolo11m.pt" /app/yolo11m.pt
ln -sf "$WEIGHTS_DIR/yolo11n.pt" /app/yolo11n.pt

# Download MiVOLO v2 on first run (HF_HOME already points to the volume)
if [ ! -d "$WEIGHTS_DIR/huggingface/hub/models--iitolstykh--mivolo_v2" ]; then
    echo "First run: downloading MiVOLO v2 model..."
    uv run --no-sync python -c "
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
m = 'iitolstykh/mivolo_v2'
AutoConfig.from_pretrained(m, trust_remote_code=True)
AutoImageProcessor.from_pretrained(m, trust_remote_code=True)
AutoModelForImageClassification.from_pretrained(m, trust_remote_code=True)
"
fi

exec "\$@"
