#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/train_2d.sh"
"$SCRIPT_DIR/train_3d.sh"
"$SCRIPT_DIR/train_hybrid.sh"
