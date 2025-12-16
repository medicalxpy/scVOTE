#!/usr/bin/env bash
set -euo pipefail

# Launch a training grid for:
#   datasets: Spleen, lung, PBMC4k, kidney
#   methods:  structure, contrastive
#   topics:   20, 50, 100
#
# Each job is launched via nohup in the background and writes logs/ and results/
# under the repository root.
#
# Usage (run from repo root):
#   bash test/train_grid.sh
#
# Optional env vars:
#   DATASETS="Spleen lung PBMC4k kidney"
#   TOPICS="20 50 100"
#   METHODS="structure contrastive"
#   DRY_RUN=1                 # only print commands
#   CUDA_VISIBLE_DEVICES=0    # pin all jobs to a GPU (or set GPU_IDS below)
#   GPU_IDS="0 1 2 3"         # round-robin GPUs when CUDA_VISIBLE_DEVICES is not set

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

DATASETS_DEFAULT=("Spleen" "lung" "PBMC4k" "kidney")
TOPICS_DEFAULT=(20 50 100)
METHODS_DEFAULT=("structure" "contrastive")

IFS=' ' read -r -a DATASETS <<< "${DATASETS:-${DATASETS_DEFAULT[*]}}"
IFS=' ' read -r -a TOPICS <<< "${TOPICS:-${TOPICS_DEFAULT[*]}}"
IFS=' ' read -r -a METHODS <<< "${METHODS:-${METHODS_DEFAULT[*]}}"

GPU_IDS_DEFAULT=("0" "1" "2" "3")
IFS=' ' read -r -a GPU_IDS <<< "${GPU_IDS:-${GPU_IDS_DEFAULT[*]}}"

DRY_RUN="${DRY_RUN:-0}"

_train_one() {
  local dataset="$1"
  local method="$2"
  local k="$3"
  local gpu="$4"

  local structure_align=0
  local contrastive_align=0
  if [[ "${method}" == "structure" ]]; then
    structure_align=1
  elif [[ "${method}" == "contrastive" ]]; then
    contrastive_align=1
  else
    echo "Unknown method: ${method}" >&2
    return 2
  fi

  local log_path="logs/train_${dataset}_${method}_K${k}.log"
  local cmd="N_TOPICS=${k} STRUCTURE_ALIGN=${structure_align} CONTRASTIVE_ALIGN=${contrastive_align} bash train.sh ${dataset}"

  # Only set CUDA_VISIBLE_DEVICES if the user has not explicitly pinned it.
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${gpu}" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "nohup bash -c '$cmd' > ${log_path} 2>&1 &"
    return 0
  fi

  nohup bash -c "${cmd}" > "${log_path}" 2>&1 &
  echo "[train_grid] launched: dataset=${dataset} method=${method} K=${k} gpu=${gpu:-\"(inherit)\"} log=${log_path}"
}

gpu_i=0
for dataset in "${DATASETS[@]}"; do
  if [[ ! -f "data/${dataset}.h5ad" ]]; then
    echo "[train_grid] skip ${dataset}: data/${dataset}.h5ad not found" >&2
    continue
  fi

  for method in "${METHODS[@]}"; do
    for k in "${TOPICS[@]}"; do
      gpu="${GPU_IDS[$((gpu_i % ${#GPU_IDS[@]}))]}"
      gpu_i=$((gpu_i + 1))
      _train_one "${dataset}" "${method}" "${k}" "${gpu}"
      sleep 1
    done
  done
done

echo "[train_grid] done launching jobs."
