#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CYST_PRETRAIN_DIR="${CYST_PRETRAIN_DIR:-$ROOT_DIR/pretrain}"
"${PYTHON:-python}" main.py --config config/loss_ablation/mobi_style_3d_L5_attention.yaml
