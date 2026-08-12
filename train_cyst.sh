#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${CONFIG:-config/cyst.yaml}"

python main.py --config "$CONFIG" "$@"
