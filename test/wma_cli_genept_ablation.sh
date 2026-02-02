#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -u pipefail

# --- GenePT ablation study across specified datasets ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADATA_DIR="${ROOT_DIR}/data"
EMB_DIR="${ROOT_DIR}/results/genept_embedding"
RESULTS_DIR="${ROOT_DIR}/results/genept_tuning"

# Training params (override via env)
N_TOPICS=${N_TOPICS:-50}
EPOCHS=${EPOCHS:-1000}
LR=${LR:-0.01}
DT_ALPHA=${DT_ALPHA:-1}
TW_ALPHA=${TW_ALPHA:-8}
THETA_TEMP=${THETA_TEMP:-5}
N_TOP_GENES_TRAIN=${N_TOP_GENES_TRAIN:-0}

# Gene filtering (keep in sync with train_fastopic)
GENE_LIST_PATH=${GENE_LIST_PATH:-"${ROOT_DIR}/data/gene_list/C2_C5_GO.csv"}
GENEPT_FILTER=${GENEPT_FILTER:-1}   # 1: enable GenePT filtering, 0: disable

# Alignment params
ALIGN_ALPHA=${ALIGN_ALPHA:-1e-3}
ALIGN_BETA=${ALIGN_BETA:-1e-3}
ALIGN_K=${ALIGN_K:-1024}
CKA_SAMPLE_N=${CKA_SAMPLE_N:-2048}
GENEPT_LOSS_WEIGHT=${GENEPT_LOSS_WEIGHT:-1e-3}
TOPIC_DIVERSITY_WEIGHT=${TOPIC_DIVERSITY_WEIGHT:-0}

# Enable nullglob so non-matching globs are skipped
shopt -s nullglob

DATASET_FILES=(
#   "${ADATA_DIR}/human_PBMC_batch1_ind"*.h5ad
#   "${ADATA_DIR}/human_PBMC_sca_method"*.h5ad
#   "${ADATA_DIR}/kidney.h5ad"
#   "${ADATA_DIR}/lung.h5ad"
  "${ADATA_DIR}/Spleen.h5ad"
  "${ADATA_DIR}/wang.h5ad"
  "${ADATA_DIR}/PBMC4k.h5ad"
  "${ADATA_DIR}/PBMC8k.h5ad"
  "${ADATA_DIR}/PBMC12k.h5ad"
)

run_variant() {
  local dataset_name="$1"
  local adata_path="$2"
  local emb_path="$3"
  local structure_align="$4"
  local contrastive_align="$5"
  local run_tag="$6"

  local structure_flag=""
  if [[ "${structure_align}" == "1" ]]; then
    structure_flag="--structure"
  else
    structure_flag="--no_align"
  fi

  local contrastive_flag=""
  local genept_weight_arg="--genept_loss_weight 0"
  if [[ "${contrastive_align}" == "1" ]]; then
    contrastive_flag="--contrastive"
    genept_weight_arg="--genept_loss_weight ${GENEPT_LOSS_WEIGHT}"
  fi

  local train_genept_flag=""
  if [[ "${GENEPT_FILTER}" != "1" ]]; then
    train_genept_flag="--no_genept_filter"
  fi

  local run_name=""
  if [[ "${structure_align}" == "1" && "${contrastive_align}" == "0" ]]; then
    run_name="${dataset_name}_structure_genept_K${N_TOPICS}"
  elif [[ "${structure_align}" == "0" && "${contrastive_align}" == "1" ]]; then
    run_name="${dataset_name}_contrastive_genept_K${N_TOPICS}"
  elif [[ "${structure_align}" == "0" && "${contrastive_align}" == "0" ]]; then
    run_name="${dataset_name}_baseline_genept_K${N_TOPICS}"
  else
    run_name="${dataset_name}_structure_contrastive_genept_K${N_TOPICS}"
  fi

  local output_dir_run=""
  if [[ -n "${run_tag}" ]]; then
    output_dir_run="${RESULTS_DIR}/${run_tag}"
  else
    output_dir_run="${RESULTS_DIR}/${run_name}"
  fi
  local eval_out_dir="${output_dir_run}/evaluation"

  local run_dataset_suffix="genept_align"
  if [[ "${structure_align}" == "0" && "${contrastive_align}" == "0" ]]; then
    run_dataset_suffix="genept"
  fi
  local run_dataset="${dataset_name}_${run_dataset_suffix}"

  python "${ROOT_DIR}/train_fastopic.py" \
    --embedding_file "${emb_path}" \
    --adata_path "${adata_path}" \
    --dataset "${run_dataset}" \
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
    ${structure_flag} \
    ${contrastive_flag} \
    ${genept_weight_arg} \
    --topic_diversity_weight ${TOPIC_DIVERSITY_WEIGHT} \
    --gene_list_path "${GENE_LIST_PATH}" \
    ${train_genept_flag} \
    --output_dir "${output_dir_run}" || return 1
}

for file in "${DATASET_FILES[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Skipping missing file: ${file}"
    continue
  fi

  dataset_name=$(basename "$file" .h5ad)
  emb_path="${EMB_DIR}/${dataset_name}_genept.pkl"

  if [[ ! -f "${emb_path}" ]]; then
    echo "GenePT embedding not found for ${dataset_name}: ${emb_path}"
    continue
  fi

  echo "Processing ${file} with dataset_name ${dataset_name}"

  # Variant A: contrastive align ON, topic diversity OFF
  run_variant "${dataset_name}" "${file}" "${emb_path}" 0 1 "${dataset_name}_contrastive_alignw1e-3_tdw0" \
    > "${ROOT_DIR}/logs/train_${dataset_name}_genept_contrastive_alignw1e-3_tdw0_K${N_TOPICS}.log" 2>&1 || \
    echo "[WARN] Variant A failed for ${dataset_name}"

  # Variant B: structure align ON, topic diversity OFF
  run_variant "${dataset_name}" "${file}" "${emb_path}" 1 0 "${dataset_name}_structure_alignw1e-3_tdw0" \
    > "${ROOT_DIR}/logs/train_${dataset_name}_genept_structure_alignw1e-3_tdw0_K${N_TOPICS}.log" 2>&1 || \
    echo "[WARN] Variant B failed for ${dataset_name}"

  # Variant C: no alignment, topic diversity OFF
  run_variant "${dataset_name}" "${file}" "${emb_path}" 0 0 "${dataset_name}_alignw0_tdw0" \
    > "${ROOT_DIR}/logs/train_${dataset_name}_genept_alignw0_tdw0_K${N_TOPICS}.log" 2>&1 || \
    echo "[WARN] Variant C failed for ${dataset_name}"

  # Variant D: contrastive align ON, topic diversity ON (reference)
  TOPIC_DIVERSITY_WEIGHT=1e-3 run_variant "${dataset_name}" "${file}" "${emb_path}" 0 1 "${dataset_name}_contrastive_alignw1e-3_tdw1e-3" \
    > "${ROOT_DIR}/logs/train_${dataset_name}_genept_contrastive_alignw1e-3_tdw1e-3_K${N_TOPICS}.log" 2>&1 || \
    echo "[WARN] Variant D failed for ${dataset_name}"

  # Variant E: structure align ON, topic diversity ON
  TOPIC_DIVERSITY_WEIGHT=1e-3 run_variant "${dataset_name}" "${file}" "${emb_path}" 1 0 "${dataset_name}_structure_alignw1e-3_tdw1e-3" \
    > "${ROOT_DIR}/logs/train_${dataset_name}_genept_structure_alignw1e-3_tdw1e-3_K${N_TOPICS}.log" 2>&1 || \
    echo "[WARN] Variant E failed for ${dataset_name}"

done
