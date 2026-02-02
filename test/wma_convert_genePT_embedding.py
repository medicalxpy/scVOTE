#!/usr/bin/env python3
"""
Visualization script for GenePT cell embeddings to check cluster separation.

This script loads GenePT-aligned embeddings and corresponding AnnData
files, performs UMAP on the embeddings, and visualizes colored by cell
type. It also supports visualizing GenePT cell-topic embeddings from
contrastive runs, and saving barcode files.
"""

import os
import pickle
import glob
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import argparse

# Suffixes and patterns for GenePT artifacts
# GenePT embedding outputs
EMB_SUFFIX = "_genept.pkl"
BARCODE_SUFFIX = "_barcodes.txt"
TOPICS = 50

def _maybe_load_barcodes(npy_file: Path, n_cells: int) -> Optional[List[str]]:
    """Find and load barcodes near npy_file if available."""
    candidates = list(npy_file.parent.glob("*cell_id*.npy"))
    candidates += list(npy_file.parent.glob("*cell_ids*.npy"))
    candidates += list(npy_file.parent.glob("*cell_barcode*.npy"))
    candidates += list(npy_file.parent.glob("*cell_barcodes*.npy"))
    candidates += list(npy_file.parent.glob("*cell_ids*.txt"))

    for cand in candidates:
        try:
            if cand.suffix == ".npy":
                arr = np.load(cand)
                arr = arr.ravel()
                barcodes = [str(x) for x in arr.tolist()]
            else:
                with open(cand, "r") as f:
                    barcodes = [line.strip() for line in f]
            if len(barcodes) == n_cells:
                return barcodes
        except Exception as exc:
            print(f"Failed to read barcodes from {cand}: {exc}")
    return None


def convert_genept_to_pickle(
    src_dir: str = "/data1021/zhuyixuan/project/scVOTE/results/genept_fastopic/genept_cell_emb",
    dest_dir: str = "/data1021/xiepengyu/scVOTE/results/genept_embedding",
    suffix: str = EMB_SUFFIX
):
    """Convert GenePT CSV embeddings to pickle and barcode TXT files.

    The CSV row index is treated as the barcode list.
    """
    os.makedirs(dest_dir, exist_ok=True)
    pattern = os.path.join(src_dir, "**", "*.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))
    if not csv_files:
        print(f"No CSV files found under {src_dir}")
        return

    target_sets = {
        "kidney",
        "lung",
        "Spleen",
        "wang",
        "PBMC4k",
        "PBMC8k",
        "PBMC12k",
    }

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        df = pd.read_csv(csv_path, index_col=0)
        arr = df.to_numpy()
        dataset_base = csv_path.stem
        dataset_base = re.sub(r"(_cell_emb|_cell_embedding|_filtered_cell_emb)$", "", dataset_base)
        dataset_base = re.sub(r"(_genept|_genePT)$", "", dataset_base)
        if dataset_base and dataset_base[0].isdigit():
            dataset_base = f"human_PBMC_batch1_ind{dataset_base}"

        if not (
            dataset_base.startswith("human_PBMC_batch1_ind")
            or dataset_base.startswith("human_PBMC_sca_method")
            or dataset_base in target_sets
        ):
            continue

        out_path = Path(dest_dir) / f"{dataset_base}{suffix}"
        with open(out_path, "wb") as f:
            pickle.dump(arr, f)
        print(f"Saved embeddings to {out_path} with shape {arr.shape}")

        barcodes = df.index.astype(str).tolist()
        barcode_path = Path(dest_dir) / f"{dataset_base}{BARCODE_SUFFIX}"
        with open(barcode_path, "w") as f:
            f.write("\n".join(barcodes))
        print(f"Saved barcodes to {barcode_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenePT embedding utilities")
    parser.add_argument(
        "--src-dir",
        type=str,
        default="/data1021/zhuyixuan/project/scVOTE/results/genept_fastopic/genept_cell_emb/",
        help="Source directory containing *_cell_emb.npy",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="/data1021/xiepengyu/scVOTE/results/genept_embedding",
        help="Destination directory for pickle outputs",
    )

    args, _ = parser.parse_known_args()
    convert_genept_to_pickle(
        src_dir=args.src_dir,
        dest_dir=args.dest_dir
    )
