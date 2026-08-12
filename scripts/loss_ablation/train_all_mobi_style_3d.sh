#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

bash scripts/loss_ablation/train_L1_mobi_style_3d.sh
bash scripts/loss_ablation/train_L2_mobi_style_3d.sh
bash scripts/loss_ablation/train_L3_mobi_style_3d.sh
bash scripts/loss_ablation/train_L4_mobi_style_3d.sh
bash scripts/loss_ablation/train_L5_mobi_style_3d.sh
bash scripts/loss_ablation/train_L6_mobi_style_3d.sh
