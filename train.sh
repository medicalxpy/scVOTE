#!/usr/bin/env bash
set -euo pipefail

# Quick driver script to train scFASTopic with/without the new
# structural alignment losses. Adjust dataset_name and paths as needed.

dataset_name=${1:-"PBMC4k"}
# Resolve repository root from this script's location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADATA_PATH="${ROOT_DIR}/data/${dataset_name}.h5ad"
EMB_DIR="${ROOT_DIR}/results/cell_embedding"
EMB_PATH="${EMB_DIR}/${dataset_name}_scvi.pkl"
RESULTS_DIR="${ROOT_DIR}/results"

# Training params
N_TOPICS=${N_TOPICS:-50}
EPOCHS=${EPOCHS:-1000}
LR=${LR:-0.01}
DT_ALPHA=${DT_ALPHA:-1}
TW_ALPHA=${TW_ALPHA:-8}
THETA_TEMP=${THETA_TEMP:-5}

# Alignment params
ALIGN_ALPHA=${ALIGN_ALPHA:-1e-3}
ALIGN_BETA=${ALIGN_BETA:-1e-3}
ALIGN_K=${ALIGN_K:-1024}
CKA_SAMPLE_N=${CKA_SAMPLE_N:-2048}

# Workaround for numba caching issues in some environments
export NUMBA_CACHE_DIR="${ROOT_DIR}/.numba_cache"
export NUMBA_DISABLE_CACHING=1
mkdir -p "$NUMBA_CACHE_DIR"

echo "[train.sh] Dataset=${dataset_name}"
echo "[train.sh] ADATA=${ADATA_PATH}"
echo "[train.sh] EMBEDDING=${EMB_PATH}"


echo "[train.sh] Extracting embedding with scVI..."
mkdir -p "${EMB_DIR}"
python "${ROOT_DIR}/get_cell_emb.py" \
  --input_data "${ADATA_PATH}" \
  --dataset_name "${dataset_name}" \
  --n_latent 128 \
  --output_dir "${EMB_DIR}" \
  --n_top_genes 0 \
  --early_stopping \
  --verbose \
  
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
#   --no_align \
#   --genept_loss_weight 1e-3

echo "[train.sh] Training WITH structural alignment (Laplacian + CKA)"
python "${ROOT_DIR}/train_fastopic.py" \
  --embedding_file "${EMB_PATH}" \
  --adata_path "${ADATA_PATH}" \
  --dataset "${dataset_name}_scVI_align" \
  --n_topics ${N_TOPICS} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --DT_alpha ${DT_ALPHA} \
  --TW_alpha ${TW_ALPHA} \
  --theta_temp ${THETA_TEMP} \
  --align_alpha ${ALIGN_ALPHA} \
  --align_beta ${ALIGN_BETA} \
  --align_knn_k ${ALIGN_K} \
  --align_cka_sample_n ${CKA_SAMPLE_N} \
  --genept_loss_weight 0.0

echo "[train.sh] Done. Results saved to ${RESULTS_DIR}"
