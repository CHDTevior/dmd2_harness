#!/usr/bin/env bash
set -euo pipefail

ROOT="/vepfs-cnbja62d5d769987/suntengjiao/distill"
BASE_DIR="${ROOT}/dmd2_assets/sdxl-base-1.0-fp16"
DMD2_DIR="${ROOT}/dmd2_assets/DMD2"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate red_train

mkdir -p "${BASE_DIR}" "${DMD2_DIR}"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_HOME="${ROOT}/.cache/huggingface" \
  hf download stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir "${BASE_DIR}" \
    --include \
      'model_index.json' \
      'scheduler/*' \
      'tokenizer/*' \
      'tokenizer_2/*' \
      'text_encoder/config.json' \
      'text_encoder/model.fp16.safetensors' \
      'text_encoder_2/config.json' \
      'text_encoder_2/model.fp16.safetensors' \
      'unet/config.json' \
      'unet/diffusion_pytorch_model.fp16.safetensors' \
      'vae/config.json' \
      'vae/diffusion_pytorch_model.fp16.safetensors' \
    --max-workers 4

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_HOME="${ROOT}/.cache/huggingface" \
  hf download tianweiy/DMD2 dmd2_sdxl_4step_lora_fp16.safetensors \
    --local-dir "${DMD2_DIR}" \
    --max-workers 2

