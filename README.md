 This guide explains how to train FASTopic on a dataset and how to perform incremental topic alignment/merge across datasets using the provided scripts.

## Prerequisites
- Place your `.h5ad` files under `data/`, for example: `data/PBMC4k.h5ad`.

## Training: train.sh
- Purpose: extracts cell embeddings with VAE, then trains FASTopic with structural alignment enabled.
- Path handling: the script infers the repository root from its own location (no hard-coded absolute paths). You can run it from any directory.
- Usage:
  - `bash train.sh [DATASET_NAME]`
  - Defaults to `DATASET_NAME=PBMC4k` (expects `data/PBMC4k.h5ad`).
- Examples:
  - `bash train.sh PBMC4k`
  - Override hyperparameters via environment variables:
    `N_TOPICS=50 EPOCHS=800 LR=0.01 DT_ALPHA=1 TW_ALPHA=8 THETA_TEMP=5 ALIGN_ALPHA=1e-3 ALIGN_BETA=1e-3 ALIGN_K=1024 CKA_SAMPLE_N=2048 bash train.sh PBMC4k`
- Outputs:
  - scVI embedding: `results/cell_embedding/<DATASET_NAME>_scvi.pkl`
  - FASTopic artifacts under `results/` (e.g., `cell_topic/`, `topic_gene/`, `gene_embedding/`, `topic_embedding/`).

## Incremental Merge/Alignment: incremental.sh
- Purpose: aligns and evaluates topics across multiple trained datasets and saves a global topic store.
- Prerequisite: train each dataset first (defaults expected by the script):
  `bash train.sh PBMC4k`, `bash train.sh PBMC8k`, `bash train.sh PBMC12k` (producing `PBMC*_scVI_align`).
- Usage:
  - `bash incremental.sh`
- Behavior:
  - Removes any existing `results/topic_store/topic_store_align.pkl`.
  - Runs two alignment/evaluation passes; the second adds `--filter_background --sparsity_threshold 0.20`.
- Outputs:
  - Topic store: `results/topic_store/topic_store_align.pkl`
  - Merge evaluation reports: `results/topic_store/merge_eval_align/`

## Notes / Troubleshooting
- Missing data: ensure `data/<DATASET_NAME>.h5ad` exists and matches the name you pass.
- Paths: both scripts resolve paths from their own locations; you can run them from anywhere.
- Performance: GPU strongly recommended; CPU will work but is slower.
