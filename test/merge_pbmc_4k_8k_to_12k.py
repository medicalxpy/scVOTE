#!/usr/bin/env python3
"""
Merge PBMC4k and PBMC8k into a single PBMC12k AnnData file.

This script expects two .h5ad files (PBMC4k and PBMC8k), aligns genes
by the intersection of their var_names, concatenates cells, and writes
out a combined PBMC12k .h5ad with a batch column.
"""

import argparse
from pathlib import Path

import scanpy as sc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge PBMC4k and PBMC8k into PBMC12k (.h5ad)."
    )
    p.add_argument(
        "--pbmc4k",
        type=str,
        default="data/PBMC4k.h5ad",
        help="Path to PBMC4k .h5ad file (default: data/PBMC4k.h5ad).",
    )
    p.add_argument(
        "--pbmc8k",
        type=str,
        default="data/PBMC8k.h5ad",
        help="Path to PBMC8k .h5ad file (default: data/PBMC8k.h5ad).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/PBMC12k.h5ad",
        help="Output path for merged PBMC12k .h5ad (default: data/PBMC12k.h5ad).",
    )
    p.add_argument(
        "--batch_key",
        type=str,
        default="batch",
        help="Obs column name to store batch labels (default: batch).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    pbmc4k_path = Path(args.pbmc4k)
    pbmc8k_path = Path(args.pbmc8k)
    out_path = Path(args.output)

    print(f"📁 Loading PBMC4k: {pbmc4k_path}")
    adata_4k = sc.read_h5ad(pbmc4k_path)
    print(f"  PBMC4k shape: {adata_4k.shape}")

    print(f"📁 Loading PBMC8k: {pbmc8k_path}")
    adata_8k = sc.read_h5ad(pbmc8k_path)
    print(f"  PBMC8k shape: {adata_8k.shape}")

    # Align genes by intersection of var_names to ensure compatible matrices.
    shared_genes = adata_4k.var_names.intersection(adata_8k.var_names)
    if shared_genes.empty:
        raise ValueError("No shared genes between PBMC4k and PBMC8k.")

    print(f"🔗 Shared genes: {len(shared_genes)}")
    adata_4k = adata_4k[:, shared_genes].copy()
    adata_8k = adata_8k[:, shared_genes].copy()

    # Concatenate along cells, adding a batch column.
    print("🔀 Concatenating PBMC4k and PBMC8k into PBMC12k...")
    merged = adata_4k.concatenate(
        adata_8k,
        batch_key=args.batch_key,
        batch_categories=["PBMC4k", "PBMC8k"],
        join="outer",
    )
    print(f"✅ PBMC12k shape: {merged.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(out_path)
    print(f"💾 Saved PBMC12k to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

