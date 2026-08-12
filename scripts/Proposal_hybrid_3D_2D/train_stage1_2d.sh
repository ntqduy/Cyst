#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CYST_PRETRAIN_DIR="${CYST_PRETRAIN_DIR:-$ROOT_DIR/pretrain}"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" || ! "$(command -v "$PYTHON_BIN" 2>/dev/null)" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python interpreter not found. Set PYTHON=/path/to/python." >&2
    exit 127
  fi
fi
CONFIG="${CONFIG:-config/Proposal_hybrid_3D_2D/Hybrid_3D_2D_stage1_2d.yaml}"
"$PYTHON_BIN" main.py --config "$CONFIG" "$@"
