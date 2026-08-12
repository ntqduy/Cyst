#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Main grid:
#   pairs: Unet3D+Unet2D, Unet3D+nnUnet2D, nnUnet3D+Unet2D, nnUnet3D+nnUnet2D
#   decoder style: same_scale(U-Net), nested_dense(UNet++), full_scale(UNet3+)
#   seeds: 42 123 205
#
# For one setup, edit config/Proposal_Experiment and run train_hybrid.sh.
# This grid script intentionally generates many override configs.
#
# Optional grid overrides:
#   SLICE=proposal|middle|uniform|random POS=true|false FUSION=add|concat
#   DECODER_MODEL=unet3d|unetpp3d|unet3plus3d|nnunet3d
#   PARALLEL_FOLDS=true|false MAX_PARALLEL_FOLDS=2 PARALLEL_GPU_IDS=0,1
#   NUM_WORKERS=4 PERSISTENT_WORKERS=true PREFETCH_FACTOR=2
#   OUT_ROOT=outputs/your_root FOLDS=3 NUM_SLICES=15 NUM_GROUPS=15

SEEDS=(${SEEDS:-$GRID_SEEDS})
DECODERS=(${DECODERS:-$GRID_DECODERS})
PAIRS=(${PAIRS:-$GRID_PAIRS})
RUN_PRETRAIN="${RUN_PRETRAIN:-0}"

ARGS=("$@")

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    declare -A trained_2d=()
    declare -A trained_3d=()
    for pair in "${PAIRS[@]}"; do
      enc2d="${pair##*:}"
      key_2d="$seed|$enc2d"
      [[ -n "${trained_2d[$key_2d]:-}" ]] && continue
      trained_2d[$key_2d]=1
      env_args=(SEED="$seed" ENC2D="$enc2d")
      add_env_if_set env_args "${COMMON_PASS_ENV[@]}"
      run_script_with_env scripts/Proposal_Experiment/train_stage1_2d.sh "${env_args[@]}"
    done
    for pair in "${PAIRS[@]}"; do
      enc3d="${pair%%:*}"
      for decoder_style in "${DECODERS[@]}"; do
        decoder_model="$(decoder_model_for_style "$decoder_style" "$enc3d")"
        key_3d="$seed|$enc3d|$decoder_model|$decoder_style"
        [[ -n "${trained_3d[$key_3d]:-}" ]] && continue
        trained_3d[$key_3d]=1
        env_args=(SEED="$seed" ENC3D="$enc3d" DECODER_MODEL="$decoder_model" DECODER_STYLE="$decoder_style")
        add_env_if_set env_args OUT_ROOT FOLDS PARALLEL_FOLDS MAX_PARALLEL_FOLDS PARALLEL_GPU_IDS NUM_WORKERS PERSISTENT_WORKERS PREFETCH_FACTOR
        run_script_with_env scripts/Proposal_Experiment/train_stage2_3d.sh "${env_args[@]}"
      done
    done
  done
fi

for seed in "${SEEDS[@]}"; do
  for pair in "${PAIRS[@]}"; do
    enc3d="${pair%%:*}"
    enc2d="${pair##*:}"
    for decoder_style in "${DECODERS[@]}"; do
      decoder_model="$(decoder_model_for_style "$decoder_style" "$enc3d")"
      env_args=(SEED="$seed" ENC3D="$enc3d" ENC2D="$enc2d" DECODER_MODEL="$decoder_model" DECODER_STYLE="$decoder_style")
      add_env_if_set env_args "${COMMON_PASS_ENV[@]}"
      run_script_with_env scripts/Proposal_Experiment/train_hybrid.sh "${env_args[@]}"
    done
  done
done
