#!/usr/bin/env bash
set -euo pipefail

cd /data1021/xiepengyu/scVOTE
conda activate scvote

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0
export SCIMILARITY_MODEL_PATH=/data1021/xiepengyu/scimilarity/models

nohup env \
  EMBED_METHOD=scimilarity \
  SCIMILARITY_MODEL_PATH="$SCIMILARITY_MODEL_PATH" \
  SCIMILARITY_USE_GPU=1 \
  FORCE_REEMBED=1 \
  ADATA_PATH=/data1021/xiepengyu/scVOTE/data/human_PBMC_sca_method_10x_Chromium_v2_A.h5ad \
  bash train.sh human_PBMC_sca_method_10x_Chromium_v2_A \
  > logs/human_PBMC_sca_method_10x_Chromium_v2_A_scimilarity.log 2>&1 &
