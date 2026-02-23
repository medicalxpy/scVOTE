#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-/data1021/xiepengyu/scVOTE}
SRC_DIR=${SRC_DIR:-/data1021/zhuyixuan/project/scVOTE/data/benchmark}
RUN_TAG=${RUN_TAG:-benchmark}

N_TOPICS=${N_TOPICS:-50}
EPOCHS=${EPOCHS:-1000}
STRUCTURE_ALIGN=${STRUCTURE_ALIGN:-1}
CONTRASTIVE_ALIGN=${CONTRASTIVE_ALIGN:-0}
FORCE_REEMBED=${FORCE_REEMBED:-1}

cd "$REPO_DIR"
mkdir -p data logs

export RUN_TAG N_TOPICS EPOCHS STRUCTURE_ALIGN CONTRASTIVE_ALIGN FORCE_REEMBED

for ds in kidney lung pancreas_scIB PBMC4k Spleen; do
  cp -f "$SRC_DIR/${ds}.h5ad" "data/${ds}.h5ad"
  bash train.sh "$ds" > "logs/train_${ds}.log" 2>&1
done
