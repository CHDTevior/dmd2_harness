#!/usr/bin/env bash
set -euo pipefail

ROOT="/vepfs-cnbja62d5d769987/suntengjiao/distill"
DMD2_REPO="${ROOT}/dmd2"
BASE_MODEL="${ROOT}/dmd2_assets/sdxl-base-1.0-fp16"
LORA="${ROOT}/dmd2_assets/DMD2/dmd2_sdxl_4step_lora_fp16.safetensors"
OUT_DIR="${ROOT}/dmd2_outputs/smoke_sdxl_lora_4step"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate twin_flow_qwen

export HF_HOME="${ROOT}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${DMD2_REPO}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/smoke_sdxl_lora_diffusers.py \
  --base_model "${BASE_MODEL}" \
  --lora "${LORA}" \
  --out_dir "${OUT_DIR}" \
  --prompt "a photo of a small clay cat on a wooden table, studio lighting" \
  --seed 20260622 \
  --height 1024 \
  --width 1024

