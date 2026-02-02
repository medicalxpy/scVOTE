#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# --- ablation study across specified datasets ---
ADATA_DIR="/data1021/xiepengyu/scVOTE/data"

# Enable nullglob so non-matching globs are skipped
shopt -s nullglob

DATASET_FILES=(
  "${ADATA_DIR}/human_PBMC_batch1_ind"*.h5ad
  "${ADATA_DIR}/human_PBMC_sca_method"*.h5ad
  "${ADATA_DIR}/kidney.h5ad"
  "${ADATA_DIR}/lung.h5ad"
  "${ADATA_DIR}/Spleen.h5ad"
  "${ADATA_DIR}/wang.h5ad"
  "${ADATA_DIR}/PBMC4k.h5ad"
  "${ADATA_DIR}/PBMC8k.h5ad"
  "${ADATA_DIR}/PBMC12k.h5ad"
)


for file in "${DATASET_FILES[@]}"; do
  # Extract dataset name from filename (remove .h5ad extension)
  dataset_name=$(basename "$file" .h5ad)

  echo "Processing $file with dataset_name $dataset_name"

  python get_cell_emb.py \
    --input_data "$file" \
    --dataset_name "$dataset_name" \
    --output_dir results/cell_embedding \
    --n_latent 128 \
    --gene_list_path data/gene_list/C2_C5_GO.csv \
    --early_stopping \
    --verbose

  # Variant A: contrastive align ON, topic diversity OFF
  bash -c "N_TOPICS=50 STRUCTURE_ALIGN=0 CONTRASTIVE_ALIGN=1 TOPIC_DIVERSITY_WEIGHT=0 RUN_TAG=${dataset_name}_contrastive_alignw1e-3_tdw0 bash train.sh ${dataset_name}" \
    > logs/train_${dataset_name}_contrastive_alignw1e-3_tdw0_K50.log 2>&1
  
  # Variant B: structure align ON, topic diversity OFF
  bash -c "N_TOPICS=50 STRUCTURE_ALIGN=1 CONTRASTIVE_ALIGN=0 TOPIC_DIVERSITY_WEIGHT=0 RUN_TAG=${dataset_name}_structure_alignw1e-3_tdw0 bash train.sh ${dataset_name}" \
    > logs/train_${dataset_name}_structure_alignw1e-3_tdw0_K50.log 2>&1

  # Variant C: no alignment, topic diversity OFF
  bash -c "N_TOPICS=50 STRUCTURE_ALIGN=0 CONTRASTIVE_ALIGN=0 TOPIC_DIVERSITY_WEIGHT=0 RUN_TAG=${dataset_name}_alignw0_tdw0 bash train.sh ${dataset_name}" \
    > logs/train_${dataset_name}_alignw0_tdw0_K50.log 2>&1

  # Variant D: contrastive align ON, topic diversity ON (reference)
  bash -c "N_TOPICS=50 STRUCTURE_ALIGN=0 CONTRASTIVE_ALIGN=1 TOPIC_DIVERSITY_WEIGHT=1e-3 RUN_TAG=${dataset_name}_contrastive_alignw1e-3_tdw1e-3 bash train.sh ${dataset_name}" \
    > logs/train_${dataset_name}_contrastive_alignw1e-3_tdw1e-3_K50.log 2>&1

  # Variant E: structure align ON, topic diversity ON
  bash -c "N_TOPICS=50 STRUCTURE_ALIGN=1 CONTRASTIVE_ALIGN=0 TOPIC_DIVERSITY_WEIGHT=1e-3 RUN_TAG=${dataset_name}_structure_alignw1e-3_tdw1e-3 bash train.sh ${dataset_name}" \
    > logs/train_${dataset_name}_structure_alignw1e-3_tdw1e-3_K50.log 2>&1

done