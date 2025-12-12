#!/usr/bin/env bash
set -euo pipefail

# Quick driver script to train scFASTopic with/without the new
# structural alignment losses. Adjust dataset_name and paths as needed.

dataset_name=${1:-"Spleen"}
# Resolve repository root from this script's location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADATA_PATH="${ROOT_DIR}/data/${dataset_name}.h5ad"
EMB_DIR="${ROOT_DIR}/results/cell_embedding"
EMB_PATH="${EMB_DIR}/${dataset_name}_vae.pkl"
RESULTS_DIR="${ROOT_DIR}/results"

# Training params
N_TOPICS=${N_TOPICS:-50}
EPOCHS=${EPOCHS:-1000}
LR=${LR:-0.01}
DT_ALPHA=${DT_ALPHA:-1}
TW_ALPHA=${TW_ALPHA:-8}
THETA_TEMP=${THETA_TEMP:-5}
N_TOP_GENES_TRAIN=${N_TOP_GENES_TRAIN:-0}
N_TOP_GENES_EMB=${N_TOP_GENES_EMB:-0}
EMB_MAX_CELLS=${EMB_MAX_CELLS:-}

# Gene filtering (keep in sync with train_fastopic / get_cell_emb)
GENE_LIST_PATH=${GENE_LIST_PATH:-"${ROOT_DIR}/data/gene_list/C2_C5_GO.csv"}
GENEPT_FILTER=${GENEPT_FILTER:-1}   # 1: enable GenePT filtering, 0: disable

# Alignment params
ALIGN_ALPHA=${ALIGN_ALPHA:-1e-3}
ALIGN_BETA=${ALIGN_BETA:-1e-3}
ALIGN_K=${ALIGN_K:-1024}
CKA_SAMPLE_N=${CKA_SAMPLE_N:-2048}
# Alignment switches
STRUCTURE_ALIGN=${STRUCTURE_ALIGN:-1}      # 1: enable structural (Laplacian+CKA) alignment, 0: disable
CONTRASTIVE_ALIGN=${CONTRASTIVE_ALIGN:-0}  # 1: enable GenePT contrastive alignment, 0: disable
GENEPT_LOSS_WEIGHT=${GENEPT_LOSS_WEIGHT:-1e-3}

# Workaround for numba caching issues in some environments
export NUMBA_CACHE_DIR="${ROOT_DIR}/.numba_cache"
export NUMBA_DISABLE_CACHING=1
mkdir -p "$NUMBA_CACHE_DIR"

RUN_TAG=${RUN_TAG:-}
if [[ -n "${RUN_TAG}" ]]; then
  OUTPUT_DIR_RUN="${RESULTS_DIR}/tuning/${RUN_TAG}"
  EVAL_OUT_DIR="${OUTPUT_DIR_RUN}/evaluation"
else
  OUTPUT_DIR_RUN="${RESULTS_DIR}"
  EVAL_OUT_DIR="${RESULTS_DIR}/evaluation"
fi

echo "[train.sh] Dataset=${dataset_name}"
echo "[train.sh] ADATA=${ADATA_PATH}"
echo "[train.sh] EMBEDDING=${EMB_PATH}"
echo "[train.sh] Output dir=${OUTPUT_DIR_RUN} (tag='${RUN_TAG}')"


echo "[train.sh] Checking/Extracting embedding..."
mkdir -p "${EMB_DIR}"
if [[ -f "${EMB_PATH}" && "${FORCE_REEMBED:-0}" != "1" ]]; then
  echo "[train.sh] Found existing embedding: ${EMB_PATH} (skip, set FORCE_REEMBED=1 to recompute)"
else

  EMB_GENEPT_FLAG=""
  if [[ "${GENEPT_FILTER}" != "1" ]]; then
    EMB_GENEPT_FLAG="--no_genept_filter"
  fi

  EMB_MAX_CELLS_ARG=()
  if [[ -n "${EMB_MAX_CELLS}" ]]; then
    EMB_MAX_CELLS_ARG=(--max_cells "${EMB_MAX_CELLS}")
  fi

  python "${ROOT_DIR}/get_cell_emb.py" \
    --input_data "${ADATA_PATH}" \
    --dataset_name "${dataset_name}" \
    --n_latent 128 \
    --output_dir "${EMB_DIR}" \
    --n_top_genes "${N_TOP_GENES_EMB}" \
    --gene_list_path "${GENE_LIST_PATH}" \
    ${EMB_GENEPT_FLAG} \
    "${EMB_MAX_CELLS_ARG[@]}" \
    --early_stopping \
    --verbose
fi
  
# echo "[train.sh] Training WITHOUT structural alignment (baseline)"
# python train_fastopic.py \
#   --embedding_file "${EMB_PATH}" \
#   --adata_path "${ADATA_PATH}" \
#   --dataset "${dataset_name}_scVI" \
#   --n_topics ${N_TOPICS} \
#   --epochs ${EPOCHS} \
#   --lr ${LR} \
#   --DT_alpha ${DT_ALPHA} \
#   --TW_alpha ${TW_ALPHA} \
#   --theta_temp ${THETA_TEMP} \
#   --n_top_genes ${N_TOP_GENES_TRAIN} \
#   --no_align \
#   --genept_loss_weight 1e-3

# Build alignment flags
STRUCTURE_FLAG=""
if [[ "${STRUCTURE_ALIGN}" == "1" ]]; then
  STRUCTURE_FLAG="--structure"
else
  STRUCTURE_FLAG="--no_align"
fi

CONTRASTIVE_FLAG=""
GENEPT_WEIGHT_ARG="--genept_loss_weight 0"
if [[ "${CONTRASTIVE_ALIGN}" == "1" ]]; then
  CONTRASTIVE_FLAG="--contrastive"
  GENEPT_WEIGHT_ARG="--genept_loss_weight ${GENEPT_LOSS_WEIGHT}"
fi

TRAIN_GENEPT_FLAG=""
if [[ "${GENEPT_FILTER}" != "1" ]]; then
  TRAIN_GENEPT_FLAG="--no_genept_filter"
fi

RUN_DATASET_SUFFIX="scVI_align"
if [[ "${STRUCTURE_ALIGN}" == "0" && "${CONTRASTIVE_ALIGN}" == "0" ]]; then
  echo "[train.sh] Training WITHOUT alignment (baseline)"
  RUN_DATASET_SUFFIX="scVI"
else
  echo "[train.sh] Training WITH alignment (structure=${STRUCTURE_ALIGN}, contrastive=${CONTRASTIVE_ALIGN})"
fi
RUN_DATASET="${dataset_name}_${RUN_DATASET_SUFFIX}"

python "${ROOT_DIR}/train_fastopic.py" \
  --embedding_file "${EMB_PATH}" \
  --adata_path "${ADATA_PATH}" \
  --dataset "${RUN_DATASET}" \
  --n_topics ${N_TOPICS} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --DT_alpha ${DT_ALPHA} \
  --TW_alpha ${TW_ALPHA} \
  --theta_temp ${THETA_TEMP} \
  --n_top_genes ${N_TOP_GENES_TRAIN} \
  --align_alpha ${ALIGN_ALPHA} \
  --align_beta ${ALIGN_BETA} \
  --align_knn_k ${ALIGN_K} \
  --align_cka_sample_n ${CKA_SAMPLE_N} \
  ${STRUCTURE_FLAG} \
  ${CONTRASTIVE_FLAG} \
  ${GENEPT_WEIGHT_ARG} \
  --gene_list_path "${GENE_LIST_PATH}" \
  ${TRAIN_GENEPT_FLAG} \
  --output_dir "${OUTPUT_DIR_RUN}"
  # --no_align

echo "[train.sh] Evaluating clustering quality (ARI/NMI)"
python "${ROOT_DIR}/evaluation.py" \
  --adata_path "${ADATA_PATH}" \
  --results_dir "${OUTPUT_DIR_RUN}" \
  --dataset "${RUN_DATASET}" \
  --n_topics ${N_TOPICS} \
  --label_key "${LABEL_KEY:-cell_type}" \
  --res_min ${RES_MIN:-0.0} \
  --res_max ${RES_MAX:-2.0} \
  --res_step ${RES_STEP:-0.1} \
  --out_dir "${EVAL_OUT_DIR}" \
  --tag "${RUN_TAG}"

echo "[train.sh] Done. Results saved to ${RESULTS_DIR}"
