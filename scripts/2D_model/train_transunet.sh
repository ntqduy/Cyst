#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WEIGHT_PATH="$ROOT_DIR/baseline/TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
if [[ ! -f "$WEIGHT_PATH" ]]; then
  echo "Missing TransUNet pretrained checkpoint: $WEIGHT_PATH" >&2
  exit 1
fi

export CYST_PRETRAIN_DIR="${CYST_PRETRAIN_DIR:-$ROOT_DIR/pretrain}"
"${PYTHON:-python}" main.py --config config/2D_model/transunet.yaml
