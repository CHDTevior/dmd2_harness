#!/usr/bin/env bash
#SBATCH -J dmd2_firered_lora_fast
#SBATCH -p gpu-a800-traing-queue-02-single
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH -t 2-00:00:00
#SBATCH -o /vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/logs/dmd2_firered_lora_fast_%j.out
#SBATCH -e /vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/logs/dmd2_firered_lora_fast_%j.err

set -euo pipefail

PROJECT_DIR=/vepfs-cnbja62d5d769987/suntengjiao/distill/dmd2_firered_porting_harness
LOG_DIR=/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/logs
CONDA_SH=/vepfs-cnbja62d5d769987/suntengjiao/anaconda3/etc/profile.d/conda.sh
CONDA_ENV=/vepfs-cnbja62d5d769987/suntengjiao/anaconda3/envs/twin_flow_qwen

CONFIG_PATH=${CONFIG_PATH:-configs/firered_gray_dmd2_lora_smoke.yaml}
DMD2_STEPS=${DMD2_STEPS:-100}
QA_MAX_SAMPLES=${QA_MAX_SAMPLES:-4}
QA_SEED=${QA_SEED:-42}
RUN_ID=${RUN_ID:-slurm_fastrun_${SLURM_JOB_ID:-manual}}

mkdir -p "${LOG_DIR}"

echo "=============================="
echo "Job started at: $(date)"
echo "Node(s): ${SLURM_NODELIST:-manual}"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "DMD2_STEPS: ${DMD2_STEPS}"
echo "RUN_ID: ${RUN_ID}"
echo "QA_MAX_SAMPLES: ${QA_MAX_SAMPLES}"
echo "=============================="

if command -v module &>/dev/null; then
  module purge
  module load cuda/12.1 || true
  module load gcc/11.3.0 || true
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export FIRERED_DISABLE_FLASH_ATTN=${FIRERED_DISABLE_FLASH_ATTN:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_${USER}_${SLURM_JOB_ID:-manual}}

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
CUDA_SO="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so(\s|$)/{print $NF; exit}')"
CUDA_SO1="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')"
echo "[CUDA lib check] libcuda.so=${CUDA_SO:-not found} libcuda.so.1=${CUDA_SO1:-not found}"
if [ -z "${CUDA_SO:-}" ] && [ -n "${CUDA_SO1:-}" ]; then
  CUDA_LINK_DIR="${PROJECT_DIR}/.cuda_link"
  mkdir -p "${CUDA_LINK_DIR}"
  ln -sfn "${CUDA_SO1}" "${CUDA_LINK_DIR}/libcuda.so"
  export LIBRARY_PATH="${CUDA_LINK_DIR}:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CUDA_LINK_DIR}:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "${TRITON_CACHE_DIR}"

echo "[Git]"
git rev-parse HEAD
git status --porcelain | sed -n '1,20p'

python - <<'PY'
import ctypes
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
ctypes.CDLL("libcuda.so")
print("libcuda.so load: OK")
PY

python scripts/preflight_firered_dmd2.py --config "${CONFIG_PATH}" --sample-check-count 20

python scripts/train_firered_dmd2_local.py \
  --config "${CONFIG_PATH}" \
  --steps "${DMD2_STEPS}" \
  --run-id "${RUN_ID}" \
  --device cuda \
  --eval-samples "${QA_MAX_SAMPLES}"

RUN_DIR="$(python - "${CONFIG_PATH}" "${RUN_ID}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(Path(cfg["project"]["output_dir"]).expanduser() / sys.argv[2]))
PY
)"
CKPT_DIR="${RUN_DIR}/checkpoints/global_step_$(printf "%06d" "${DMD2_STEPS}")"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[FATAL] Missing checkpoint: ${CKPT_DIR}" >&2
  exit 2
fi

python scripts/eval_firered_dmd2_qa.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT_DIR}" \
  --output-dir "${RUN_DIR}/offline_eval_qa" \
  --max-samples "${QA_MAX_SAMPLES}" \
  --seed "${QA_SEED}" \
  --device cuda \
  --dtype bf16

echo "RUN_DIR=${RUN_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "QA_CONTACT_SHEET=${RUN_DIR}/offline_eval_qa/$(basename "${CKPT_DIR}")/contact_sheet.png"
echo "=============================="
echo "Job finished at: $(date)"
echo "=============================="
