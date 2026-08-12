#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! "${PYTHON:-python}" -c "import mamba_ssm, causal_conv1d" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Missing SegMamba CUDA extensions. Install them in the active CUDA environment:
  pip install ./baseline/SegMamba/causal-conv1d
  pip install ./baseline/SegMamba/mamba
EOF
  exit 1
fi

export CYST_PRETRAIN_DIR="${CYST_PRETRAIN_DIR:-$ROOT_DIR/pretrain}"
"${PYTHON:-python}" main.py --config config/3D_model/segmamba.yaml
