#!/usr/bin/env bash
set -euo pipefail

# Launch a training grid for:
#   datasets: Spleen, lung, PBMC4k, kidney
#   methods:  structure, contrastive
#   topics:   20, 50, 100
#
# This script does two stages:
#   1) Precompute embeddings once per dataset (serial) to avoid races.
#   2) Run training jobs with concurrency capped to one job per GPU.
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
#   EMBED_ONLY=1              # only compute embeddings (no training)
#   CUDA_VISIBLE_DEVICES=0    # pin all jobs to a GPU (or set GPU_IDS below)
#   GPU_IDS="0 1 2 3"         # round-robin GPUs when CUDA_VISIBLE_DEVICES is not set
#   MAX_PARALLEL=4            # max concurrent training jobs (default: number of GPUs)

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
EMBED_ONLY="${EMBED_ONLY:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EMB_DIR="${ROOT_DIR}/results/cell_embedding"
GENE_LIST_PATH="${GENE_LIST_PATH:-${ROOT_DIR}/data/gene_list/C2_C5_GO.csv}"
GENEPT_FILTER="${GENEPT_FILTER:-1}"
N_TOP_GENES_EMB="${N_TOP_GENES_EMB:-0}"
EMB_MAX_CELLS="${EMB_MAX_CELLS:-}"

mkdir -p "${EMB_DIR}"

_embed_one() {
  local dataset="$1"
  local gpu="$2"

  local adata_path="data/${dataset}.h5ad"
  local emb_path="${EMB_DIR}/${dataset}_vae.pkl"
  local log_path="logs/embed_${dataset}.log"

  if [[ -f "${emb_path}" ]]; then
    echo "[train_grid] embedding exists, skip: ${emb_path}"
    return 0
  fi

  local genept_flag=""
  if [[ "${GENEPT_FILTER}" != "1" ]]; then
    genept_flag="--no_genept_filter"
  fi

  local max_cells_arg=()
  if [[ -n "${EMB_MAX_CELLS}" ]]; then
    max_cells_arg=(--max_cells "${EMB_MAX_CELLS}")
  fi

  local cmd="${PYTHON_BIN} get_cell_emb.py --input_data ${adata_path} --dataset_name ${dataset} --n_latent 128 --output_dir ${EMB_DIR} --n_top_genes ${N_TOP_GENES_EMB} --gene_list_path ${GENE_LIST_PATH} ${genept_flag} ${max_cells_arg[*]} --early_stopping --verbose"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${gpu}" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "bash -c '$cmd' > ${log_path} 2>&1"
    return 0
  fi

  echo "[train_grid] embedding start: dataset=${dataset} gpu=${gpu:-\"(inherit)\"} log=${log_path}"
  # Run embedding in the foreground (this script itself is typically run under nohup).
  bash -c "${cmd}" > "${log_path}" 2>&1
  if [[ ! -f "${emb_path}" ]]; then
    echo "[train_grid] embedding failed: ${dataset} (missing ${emb_path}), see ${log_path}" >&2
    return 1
  fi
  echo "[train_grid] embedding done: ${emb_path}"
}

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

echo "[train_grid] stage 1/2: embeddings"
embed_gpu="${GPU_IDS[0]}"
for dataset in "${DATASETS[@]}"; do
  if [[ ! -f "data/${dataset}.h5ad" ]]; then
    echo "[train_grid] skip ${dataset}: data/${dataset}.h5ad not found" >&2
    continue
  fi
  _embed_one "${dataset}" "${embed_gpu}" || true
done

if [[ "${EMBED_ONLY}" == "1" ]]; then
  echo "[train_grid] EMBED_ONLY=1, stopping after embeddings."
  exit 0
fi

echo "[train_grid] stage 2/2: training (one job per GPU)"

MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS[@]}}"
declare -A PID2GPU=()
declare -a RUNNING_PIDS=()
declare -a FREE_GPUS=()
FREE_GPUS=("${GPU_IDS[@]}")

_reap_finished() {
  local new_pids=()
  for pid in "${RUNNING_PIDS[@]}"; do
    if [[ -z "${pid}" ]]; then
      continue
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
    else
      local g="${PID2GPU["${pid}"]-}"
      if [[ -n "${g}" ]]; then
        FREE_GPUS+=("${g}")
      fi
      unset PID2GPU["${pid}"] || true
    fi
  done
  RUNNING_PIDS=("${new_pids[@]}")
}

_wait_for_slot() {
  while true; do
    _reap_finished
    if [[ "${#RUNNING_PIDS[@]}" -lt "${MAX_PARALLEL}" && "${#FREE_GPUS[@]}" -gt 0 ]]; then
      return 0
    fi
    sleep 10
  done
}

_launch_controlled() {
  local dataset="$1"
  local method="$2"
  local k="$3"

  _wait_for_slot
  local gpu="${FREE_GPUS[0]}"
  FREE_GPUS=("${FREE_GPUS[@]:1}")

  local log_path="logs/train_${dataset}_${method}_K${k}.log"
  local structure_align=0
  local contrastive_align=0
  if [[ "${method}" == "structure" ]]; then
    structure_align=1
  else
    contrastive_align=1
  fi

  local cmd="N_TOPICS=${k} STRUCTURE_ALIGN=${structure_align} CONTRASTIVE_ALIGN=${contrastive_align} bash train.sh ${dataset}"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "nohup bash -c '$cmd' > ${log_path} 2>&1 &"
    return 0
  fi

  nohup bash -c "${cmd}" > "${log_path}" 2>&1 &
  local pid=$!
  PID2GPU["${pid}"]="${gpu}"
  RUNNING_PIDS+=("${pid}")
  echo "[train_grid] launched: dataset=${dataset} method=${method} K=${k} gpu=${gpu} pid=${pid} log=${log_path}"
}

for dataset in "${DATASETS[@]}"; do
  local_emb="${EMB_DIR}/${dataset}_vae.pkl"
  if [[ ! -f "${local_emb}" ]]; then
    echo "[train_grid] skip ${dataset}: embedding missing (${local_emb})" >&2
    continue
  fi

  for method in "${METHODS[@]}"; do
    for k in "${TOPICS[@]}"; do
      _launch_controlled "${dataset}" "${method}" "${k}"
      sleep 1
    done
  done
done

echo "[train_grid] waiting for remaining jobs: ${#RUNNING_PIDS[@]}"
for pid in "${RUNNING_PIDS[@]:-}"; do
  wait "${pid}" || true
done
echo "[train_grid] all jobs finished."
