#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Ablation for the best encoder/decoder combo. For one setup, edit config.
# This script intentionally generates slice/position override configs.
# Set ENC3D/ENC2D/DECODER_MODEL/DECODER_STYLE after you choose the best model.
# DECODER is still accepted as an alias of DECODER_STYLE.
# Set OUT_ROOT=outputs/Proposal_Model_Experiment_Ablation if you want a separate root.

SEEDS=(${SEEDS:-$GRID_SEEDS})
SLICES=(${SLICES:-$ABLATION_SLICES})
POSITIONS=(${POSITIONS:-$ABLATION_POSITIONS})
RUN_PRETRAIN="${RUN_PRETRAIN:-0}"
ARGS=("$@")

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  best_enc2d="${ENC2D:-unet}"
  best_enc3d="${ENC3D:-unet3plus3d}"
  best_decoder_style="${DECODER_STYLE:-${DECODER:-full_scale}}"
  best_decoder_model="$(decoder_model_for_style "$best_decoder_style" "$best_enc3d")"
  for seed in "${SEEDS[@]}"; do
    env_args=(SEED="$seed" ENC3D="$best_enc3d" DECODER_MODEL="$best_decoder_model" DECODER_STYLE="$best_decoder_style")
    add_env_if_set env_args OUT_ROOT FOLDS PARALLEL_FOLDS MAX_PARALLEL_FOLDS PARALLEL_GPU_IDS NUM_WORKERS PERSISTENT_WORKERS PREFETCH_FACTOR
    run_script_with_env scripts/Proposal_Experiment/train_stage2_3d.sh "${env_args[@]}"
    for slice in "${SLICES[@]}"; do
      for pos in "${POSITIONS[@]}"; do
        env_args=(SEED="$seed" ENC2D="$best_enc2d" SLICE="$slice" POS="$pos")
        add_env_if_set env_args "${COMMON_PASS_ENV[@]}"
        run_script_with_env scripts/Proposal_Experiment/train_stage1_2d.sh "${env_args[@]}"
      done
    done
  done
fi

for seed in "${SEEDS[@]}"; do
  for slice in "${SLICES[@]}"; do
    for pos in "${POSITIONS[@]}"; do
      env_args=(SEED="$seed" SLICE="$slice" POS="$pos")
      add_env_if_set env_args "${ABLATION_PASS_ENV[@]}"
      run_script_with_env scripts/Proposal_Experiment/train_hybrid.sh "${env_args[@]}"
    done
  done
done
