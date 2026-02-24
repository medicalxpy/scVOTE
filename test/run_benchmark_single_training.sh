#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-/data1021/xiepengyu/scVOTE}
SRC_DIR=${SRC_DIR:-/data1021/zhuyixuan/project/scVOTE/data/benchmark}
RUN_TAG=${RUN_TAG:-benchmark}

EPOCHS=${EPOCHS:-1000}
STRUCTURE_ALIGN=${STRUCTURE_ALIGN:-1}
CONTRASTIVE_ALIGN=${CONTRASTIVE_ALIGN:-0}
FORCE_REEMBED=${FORCE_REEMBED:-0}

DATASETS=(kidney lung pancreas_scIB PBMC4k Spleen)

# GPU assignment: gpu0 → K=100, gpu1 → K=50, gpu2 → K=20
declare -A GPU_MAP=( [100]=0 [50]=1 [20]=2 )

cd "$REPO_DIR"
mkdir -p data logs

export EPOCHS STRUCTURE_ALIGN CONTRASTIVE_ALIGN FORCE_REEMBED

# Copy data first
for ds in "${DATASETS[@]}"; do
  cp -f "$SRC_DIR/${ds}.h5ad" "data/${ds}.h5ad"
done

# Per-GPU worker: trains all datasets sequentially for a given K
train_group() {
  local k=$1
  local gpu=$2
  for ds in "${DATASETS[@]}"; do
    echo "[GPU${gpu}] Training ${ds} K=${k} ..."
    CUDA_VISIBLE_DEVICES="$gpu" N_TOPICS="$k" \
      RUN_TAG="${RUN_TAG_PREFIX:-benchmark}_K${k}" \
      bash train.sh "$ds" > "logs/train_${ds}_K${k}.log" 2>&1
    echo "[GPU${gpu}] Done ${ds} K=${k}"
  done
}

# Launch 3 GPU groups in parallel
for k in 100 50 20; do
  train_group "$k" "${GPU_MAP[$k]}" &
  echo "Launched K=${k} on GPU${GPU_MAP[$k]}  (pid=$!)"
done

echo "All groups launched. Monitor with: tail -f logs/train_*.log"
wait
echo "All training complete."
