#!/usr/bin/env bash
#SBATCH -J dmd2_firered_full
#SBATCH -p gpu-a800-traing-queue-02-single
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH -t 2-00:00:00
#SBATCH -o /vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/logs/dmd2_firered_full_%j.out
#SBATCH -e /vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth/logs/dmd2_firered_full_%j.err
#SBATCH --open-mode=append

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TWINFLOW_SRC="${TWINFLOW_SRC:-/vepfs-cnbja62d5d769987/suntengjiao/TwinFlow/src}"
FIRERED_ROOT="${FIRERED_ROOT:-/vepfs-cnbja62d5d769987/suntengjiao/distill/firered_gray_depth}"
LOG_DIR="${FIRERED_ROOT}/logs"
CONFIG_ARG="${1:-${PROJECT_DIR}/configs/firered_gray_dmd2_full_official_cfg4_4nfe_1024_3k_lr5e6_dmd2renoise_gan.yaml}"
CONDA_ENV="${CONDA_ENV:-red_train}"

mkdir -p "${LOG_DIR}"

echo "=============================="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Requeue restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node(s): ${SLURM_NODELIST:-manual}"
echo "Num nodes: ${SLURM_NNODES:-1}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "CONFIG_ARG: ${CONFIG_ARG}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "=============================="

if command -v module &>/dev/null; then
  module purge
  module load cuda/12.1 || true
  module load gcc/11.3.0 || true
fi

source /vepfs-cnbja62d5d769987/suntengjiao/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${TWINFLOW_SRC}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export FIRERED_DISABLE_FLASH_ATTN="${FIRERED_DISABLE_FLASH_ATTN:-1}"
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FIRERED_USE_COS=1
export FIRERED_DATA_PREFIX="${FIRERED_ROOT}/cos_data_clay"
export COS_YAML_OVERRIDE="${COS_YAML_OVERRIDE:-/vepfs-cnbja62d5d769987/suntengjiao/.cos.yaml}"
export REDEDIT_LOG_LEVEL="${REDEDIT_LOG_LEVEL:-INFO}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF}}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if ! COS_EXPORTS="$(
python - <<'PY'
import os
import shlex
from pathlib import Path

import yaml

paths = [
    Path(os.environ.get("COS_YAML_OVERRIDE", "")),
    Path.home() / ".cos.yaml",
    Path("/vepfs-cnbja62d5d769987/suntengjiao/.cos.yaml"),
]
cfg_path = next((p for p in paths if str(p) and p.is_file()), None)
if cfg_path is None:
    raise SystemExit("COS config not found")
data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
base = (((data or {}).get("cos") or {}).get("base") or {})
buckets = (((data or {}).get("cos") or {}).get("buckets") or [])
bucket = buckets[0] if buckets else {}
endpoint = bucket.get("endpoint") or ""
region = os.environ.get("COS_REGION") or bucket.get("region") or ""
if not region and endpoint.startswith("cos.") and endpoint.endswith(".myqcloud.com"):
    region = endpoint[len("cos."):-len(".myqcloud.com")]
region = region or "ap-guangzhou"
values = {
    "COS_SECRET_ID": os.environ.get("COS_SECRET_ID") or base.get("secretid") or "",
    "COS_SECRET_KEY": os.environ.get("COS_SECRET_KEY") or base.get("secretkey") or "",
    "COS_SESSION_TOKEN": os.environ.get("COS_SESSION_TOKEN") or base.get("sessiontoken") or "",
    "COS_REGION": region,
    "COS_ENDPOINT": os.environ.get("COS_ENDPOINT") or endpoint or f"cos.{region}.myqcloud.com",
}
if not values["COS_SECRET_ID"] or not values["COS_SECRET_KEY"]:
    raise SystemExit("COS config lacks secretid/secretkey")
for key, value in values.items():
    if value:
        print(f"export {key}={shlex.quote(str(value))}")
PY
)"; then
  echo "[ERR] Failed to export COS credentials" >&2
  exit 1
fi
eval "${COS_EXPORTS}"
unset COS_EXPORTS

if [[ -z "${COS_SECRET_ID:-}" || -z "${COS_SECRET_KEY:-}" ]]; then
  echo "[ERR] COS credentials were not exported from ${COS_YAML_OVERRIDE}" >&2
  exit 1
fi
echo "[cos] credentials exported from ${COS_YAML_OVERRIDE}: endpoint=${COS_ENDPOINT:-<auto>} sts=$( [[ -n "${COS_SESSION_TOKEN:-}" ]] && echo yes || echo no )"

CUDA_LINK_DIR="${PROJECT_DIR}/.cuda_link"
mkdir -p "${CUDA_LINK_DIR}"
if [[ -f "/lib/x86_64-linux-gnu/libcuda.so.1" ]]; then
  ln -sfn "/lib/x86_64-linux-gnu/libcuda.so.1" "${CUDA_LINK_DIR}/libcuda.so"
fi
export LIBRARY_PATH="${CUDA_LINK_DIR}:/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_LINK_DIR}:/usr/lib/x86_64-linux-gnu:/usr/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
  MASTER_ADDR="$(getent ahostsv4 "${MASTER_HOST}" | awk '{print $1; exit}')"
  if [[ -z "${MASTER_ADDR}" ]]; then
    echo "[ERR] failed to resolve IPv4 for MASTER_HOST=${MASTER_HOST}" >&2
    exit 1
  fi
else
  MASTER_HOST=127.0.0.1
  MASTER_ADDR=127.0.0.1
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-19541}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-10000}"
export REDEDIT_NCCL_TIMEOUT_MIN="${REDEDIT_NCCL_TIMEOUT_MIN:-30}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

if [[ "${CONFIG_ARG}" = /* ]]; then
  CONFIG_PATH="${CONFIG_ARG}"
else
  CONFIG_PATH="${PROJECT_DIR}/${CONFIG_ARG}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[Git]"
git rev-parse HEAD
git status --porcelain | sed -n '1,40p'

echo "[Preflight] standalone"
python scripts/preflight_firered_dmd2.py \
  --config "${CONFIG_PATH}" \
  --sample-check-count "${PREFLIGHT_SAMPLE_CHECK_COUNT:-1}"

python - <<PY
import shutil
from pathlib import Path
from omegaconf import OmegaConf

cfg_path = Path("${CONFIG_PATH}")
cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
output_base = Path(str(cfg["train"]["output_dir"]))
exp_name = Path(cfg_path.parent.name) / cfg_path.stem
resolved_output_dir = output_base / exp_name
expected_gb = float(cfg["train"].get("checkpoint_expected_size_gb", 0) or 0)
preclean_before_save = bool(cfg["train"].get("checkpoint_preclean_before_save", False))
checkpoint_limit = int(cfg["train"].get("checkpoints_total_limit", 0) or 0)

print("[Preflight] output_dir:", resolved_output_dir)
resolved_output_dir.mkdir(parents=True, exist_ok=True)
(resolved_output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(resolved_output_dir / "offline_eval").mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(str(resolved_output_dir)).free / (1024**3)
print(f"[Preflight] free_space_gib: {free_gb:.1f}")
if expected_gb > 0 and free_gb < expected_gb:
    if not (preclean_before_save and checkpoint_limit > 0):
        raise SystemExit(f"[ERR] insufficient free space: free={free_gb:.1f}GiB expected_checkpoint={expected_gb:.1f}GiB")
    print(
        "[Preflight] free space is below one full checkpoint, but the training "
        "script will preclean retained checkpoints before its authoritative guard"
    )
print("[Preflight] output directory OK")
PY

export NNODES="${SLURM_NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PROJECT_DIR CONFIG_PATH NPROC_PER_NODE MASTER_ADDR MASTER_PORT

echo "[Launch] torchrun nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT} master_host=${MASTER_HOST} ifname=${NCCL_SOCKET_IFNAME}"
if (( NNODES > 1 )); then
  srun \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    bash -lc 'cd "${PROJECT_DIR}" && \
      LOCAL_ADDR="$(ip route get "${MASTER_ADDR}" | awk '"'"'{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}'"'"')" && \
      if [[ -z "${LOCAL_ADDR}" ]]; then LOCAL_ADDR="$(hostname -I | awk '"'"'{print $1; exit}'"'"')"; fi && \
      echo "[LaunchNode] node=$(hostname) node_rank=${SLURM_PROCID} local_addr=${LOCAL_ADDR}" && \
      torchrun \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${SLURM_PROCID}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      --local_addr="${LOCAL_ADDR}" \
      --max_restarts=0 \
      scripts/train_firered_dmd2_full_fsdp.py "${CONFIG_PATH}"'
else
  LOCAL_ADDR=127.0.0.1
  echo "[LaunchNode] node=$(hostname) node_rank=0 local_addr=${LOCAL_ADDR}"
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --local_addr="${LOCAL_ADDR}" \
    --max_restarts=0 \
    scripts/train_firered_dmd2_full_fsdp.py "${CONFIG_PATH}"
fi

echo "=============================="
echo "Job finished at: $(date)"
echo "=============================="
