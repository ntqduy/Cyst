#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
DEFAULT_CONFIG="${CONFIG:-config/cyst.yaml}"
SLICE_POSITION_ARG="${SLICE_POSITION:-label_foreground}"
VISUAL_SELECTION_ARG="${VISUAL_SELECTION:-per_source}"
VISUAL_SEED_ARG="${VISUAL_SEED:-42}"
SAMPLES_PER_SOURCE_ARG="${SAMPLES_PER_SOURCE:-1}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pth}"
SEARCH_ROOT="${SEARCH_ROOT:-outputs}"
GPU_IDS_ARG="${GPU_IDS:-}"
USE_CUDA_ARG="${USE_CUDA:-}"
MULTI_GPU_ARG="${MULTI_GPU:-}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_one() {
  local checkpoint_path="$1"
  local output_dir="$2"
  local fold_arg="${3:-}"
  local config_path="${4:-${DEFAULT_CONFIG}}"
  local use_checkpoint_config="${5:-1}"

  local cmd=(
    "${PYTHON_BIN}"
    main.py
    --config "${config_path}"
    --visualize-only
    --checkpoint "${checkpoint_path}"
    --slice-position "${SLICE_POSITION_ARG}"
    --visual-selection "${VISUAL_SELECTION_ARG}"
    --visual-seed "${VISUAL_SEED_ARG}"
    --samples-per-source "${SAMPLES_PER_SOURCE_ARG}"
    --evaluate-2d-as-volume
  )

  if [[ -n "${GPU_IDS_ARG}" ]]; then
    cmd+=(--gpu-ids "${GPU_IDS_ARG}")
  fi

  if [[ -n "${USE_CUDA_ARG}" ]]; then
    if is_true "${USE_CUDA_ARG}"; then
      cmd+=(--use-cuda)
    else
      cmd+=(--cpu)
    fi
  elif [[ -n "${GPU_IDS_ARG}" ]]; then
    cmd+=(--use-cuda)
  fi

  if [[ -n "${MULTI_GPU_ARG}" ]]; then
    if is_true "${MULTI_GPU_ARG}"; then
      cmd+=(--multi-gpu)
    else
      cmd+=(--single-gpu)
    fi
  elif [[ "${GPU_IDS_ARG}" == *,* ]]; then
    cmd+=(--multi-gpu)
  fi

  if [[ "${use_checkpoint_config}" == "1" ]]; then
    cmd+=(--use-checkpoint-config)
  fi

  if [[ -n "${output_dir}" ]]; then
    cmd+=(--output-dir "${output_dir}")
  fi

  if [[ -n "${fold_arg}" ]]; then
    cmd+=(--fold "${fold_arg}")
  fi

  if [[ -n "${NUM_VISUALS:-}" ]]; then
    cmd+=(--num-visuals "${NUM_VISUALS}")
  fi

  echo "[visualize] checkpoint=${checkpoint_path}"
  echo "[visualize] output=${output_dir:-auto}"
  if [[ -n "${GPU_IDS_ARG}" ]]; then
    echo "[visualize] gpu_ids=${GPU_IDS_ARG}"
  fi
  "${cmd[@]}"
}

detect_fold() {
  local run_dir="$1"
  local name
  name="$(basename "${run_dir}")"
  if [[ "${name}" =~ ^fold_0*([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  fi
}

relative_run_dir() {
  local run_dir="$1"
  local root="$2"
  local rel="${run_dir#${root}/}"
  if [[ "${rel}" == "${run_dir}" ]]; then
    rel="$(basename "${run_dir}")"
  fi
  echo "${rel}"
}

run_all() {
  local batch_output_root="${1:-${OUTPUT_ROOT:-outputs/visualize_synced}}"
  local fail_fast="${FAIL_FAST:-0}"
  local failures=()
  local checkpoints=()

  if [[ ! -d "${SEARCH_ROOT}" ]]; then
    echo "Search root does not exist: ${SEARCH_ROOT}" >&2
    exit 2
  fi

  while IFS= read -r -d '' checkpoint; do
    checkpoints+=("${checkpoint}")
  done < <(find "${SEARCH_ROOT}" -type f -path "*/checkpoint/${CHECKPOINT_NAME}" -print0 | sort -z)

  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No checkpoints found under ${SEARCH_ROOT} with name ${CHECKPOINT_NAME}" >&2
    exit 2
  fi

  echo "[visualize-all] checkpoints=${#checkpoints[@]}"
  echo "[visualize-all] output_root=${batch_output_root}"
  echo "[visualize-all] selection=${VISUAL_SELECTION_ARG}, samples_per_source=${SAMPLES_PER_SOURCE_ARG}, seed=${VISUAL_SEED_ARG}, slice_position=${SLICE_POSITION_ARG}"
  if [[ -n "${GPU_IDS_ARG}" ]]; then
    echo "[visualize-all] gpu_ids=${GPU_IDS_ARG}, multi_gpu=${MULTI_GPU_ARG:-auto}"
  fi

  for checkpoint in "${checkpoints[@]}"; do
    local checkpoint_dir
    local run_dir
    local rel_dir
    local output_dir
    local fold_arg

    checkpoint_dir="$(dirname "${checkpoint}")"
    run_dir="$(dirname "${checkpoint_dir}")"
    rel_dir="$(relative_run_dir "${run_dir}" "${SEARCH_ROOT}")"
    output_dir="${batch_output_root}/${rel_dir}"
    fold_arg="$(detect_fold "${run_dir}")"

    if ! run_one "${checkpoint}" "${output_dir}" "${fold_arg}" "${DEFAULT_CONFIG}" "1"; then
      failures+=("${checkpoint}")
      if [[ "${fail_fast}" == "1" ]]; then
        break
      fi
    fi
  done

  if [[ "${#failures[@]}" -gt 0 ]]; then
    echo "[visualize-all] failed ${#failures[@]} checkpoint(s):" >&2
    printf '  %s\n' "${failures[@]}" >&2
    exit 1
  fi

  echo "[visualize-all] done"
}

show_usage() {
  cat <<'EOF'
Usage:
  # Re-visualize one checkpoint. Uses the supplied config unless USE_CHECKPOINT_CONFIG=1.
  bash scripts/visualize_from_weight.sh CONFIG CHECKPOINT [OUTPUT_DIR] [SLICE_POSITION] [FOLD]

  # Re-visualize all trained checkpoints under outputs/.
  bash scripts/visualize_from_weight.sh ALL [OUTPUT_ROOT]

Examples:
  bash scripts/visualize_from_weight.sh ALL outputs/visualize_synced

  bash scripts/visualize_from_weight.sh \
    config/2D_model/unet.yaml \
    outputs/unet/pos_center_sam_center_spv1_sl1_ax2/unet_encoder_epoch60/checkpoint/best.pth \
    outputs/visualize_synced/unet \
    label_foreground

Useful env:
  SEARCH_ROOT=outputs
  CHECKPOINT_NAME=best.pth
  SLICE_POSITION=label_foreground
  VISUAL_SELECTION=per_source
  VISUAL_SEED=42
  SAMPLES_PER_SOURCE=1
  NUM_VISUALS=7
  GPU_IDS=0
  GPU_IDS="0,1" MULTI_GPU=true
  USE_CUDA=false
  USE_CHECKPOINT_CONFIG=1
  FAIL_FAST=1
EOF
}

MODE_OR_CONFIG="${1:-${CONFIG_PATH:-}}"

if [[ -z "${MODE_OR_CONFIG}" ]]; then
  show_usage
  exit 2
fi

if [[ "${MODE_OR_CONFIG}" == "ALL" || "${MODE_OR_CONFIG}" == "all" ]]; then
  run_all "${2:-${OUTPUT_ROOT:-outputs/visualize_synced}}"
  exit 0
fi

CONFIG_PATH="${MODE_OR_CONFIG}"
CHECKPOINT_PATH="${2:-${CHECKPOINT:-}}"
OUTPUT_DIR_ARG="${3:-${OUTPUT_DIR:-}}"
if [[ $# -ge 4 ]]; then
  SLICE_POSITION_ARG="$4"
fi
FOLD_ARG="${5:-${FOLD:-}}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  show_usage
  exit 2
fi

run_one "${CHECKPOINT_PATH}" "${OUTPUT_DIR_ARG}" "${FOLD_ARG}" "${CONFIG_PATH}" "${USE_CHECKPOINT_CONFIG:-0}"
