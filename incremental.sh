#!/usr/bin/env bash
set -euo pipefail

# Run incremental topic alignment/evaluation from the repo root,
# regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -f results/topic_store/topic_store_align.pkl
python incremental_eval.py \
  --results_dir results \
  --datasets PBMC4k_scVI_align PBMC8k_scVI_align PBMC12k_scVI_align \
  --store_path results/topic_store/topic_store_align.pkl \
  --out_dir results/topic_store/merge_eval_align \
  --metric cosine --reg 0.005 --reg_m 0.5 \
  --min_best_ratio 0.2 --min_transport_mass 1e-3 --smoothing 0.5

rm -f results/topic_store/topic_store_align.pkl
python incremental_eval.py \
  --results_dir results \
  --datasets PBMC4k_scVI_align PBMC8k_scVI_align PBMC12k_scVI_align \
  --store_path results/topic_store/topic_store_align.pkl \
  --out_dir results/topic_store/merge_eval_align \
  --metric cosine --reg 0.005 --reg_m 0.5 \
  --min_best_ratio 0.2 --min_transport_mass 1e-3 --smoothing 0.5 \
  --filter_background --sparsity_threshold 0.20

Example usage: add topics incrementally to TopicStore
1) Add by dataset name (artifacts already under results/...)
python train_incremental.py \
  --dataset PBMC8k_scVI_align \
  --from_dataset \
  --results_dir results \
  --store_path results/topic_store/topic_store_align.pkl \
  --metric cosine --reg 0.005 --reg_m 0.5 --min_best_ratio 0.2 --min_transport_mass 1e-3 \
  --filter_background --sparsity_threshold 0.20 --expand_genes

2) Add by explicit artifact paths (e.g., produced by an external get_incremental.py)
python train_incremental.py \
  --dataset CUSTOM_RUN \
  --from_paths \
  --topic_embeddings /path/to/topic_embeddings.pkl \
  --gene_embeddings /path/to/gene_embeddings.pkl \
  --gene_names /path/to/gene_names.pkl \
  --cell_topic /path/to/cell_topic_matrix.pkl \
  --store_path results/topic_store/topic_store_align.pkl \
  --metric cosine --reg 0.005 --reg_m 0.5 --min_best_ratio 0.2 --min_transport_mass 1e-3 \
  --filter_background --sparsity_threshold 0.20 --expand_genes
