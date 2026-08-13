#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CYST_PRETRAIN_DIR="${CYST_PRETRAIN_DIR:-$ROOT_DIR/pretrain}"
WEIGHT_PATH="$CYST_PRETRAIN_DIR/model_swinvit.pt"
if [[ ! -f "$WEIGHT_PATH" ]]; then
  echo "Swin-UNETR pretrained checkpoint not found: $WEIGHT_PATH" >&2
  echo "Training Swin-UNETR from scratch." >&2
else
  echo "Found Swin-UNETR pretrained checkpoint: $WEIGHT_PATH"
  echo "Pretrained weights will be loaded before training."
fi
"${PYTHON:-python}" main.py --config config/3D_model/swin_unetr.yaml
