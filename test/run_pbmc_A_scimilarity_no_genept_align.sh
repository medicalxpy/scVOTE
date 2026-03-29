#!/usr/bin/env bash
set -euo pipefail

cd /data1021/xiepengyu/scVOTE

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0
export SCIMILARITY_MODEL_PATH=/data1021/xiepengyu/scVOTE/scimilarity/models

nohup env \
  EMBED_METHOD=scimilarity \
  SCIMILARITY_MODEL_PATH="$SCIMILARITY_MODEL_PATH" \
  SCIMILARITY_USE_GPU=1 \
  CONTRASTIVE_ALIGN=0 \
  FORCE_REEMBED=1 \
  ADATA_PATH=/data1021/xiepengyu/scVOTE/data/human_PBMC_sca_method_10x_Chromium_v2_A.h5ad \
  bash train.sh human_PBMC_sca_method_10x_Chromium_v2_A \
  > logs/human_PBMC_sca_method_10x_Chromium_v2_A_scimilarity_no_genept_align.log 2>&1 &
