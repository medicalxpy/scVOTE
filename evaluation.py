#!/usr/bin/env python3
"""
Evaluation utilities for scFASTopic training outputs.

This script computes clustering quality metrics (ARI, NMI) on the
cell-topic matrix produced by training, following the logic used in the
"Clustering and Interpretable Evaluation.ipynb" notebook:

1) Build an AnnData with X = cell-topic matrix
2) Copy ground-truth labels from the input .h5ad into obs[label_key]
3) Run neighbors and scan over Louvain resolution values
4) Pick the resolution that maximizes ARI, then report ARI and NMI

Usage example:

python evaluation.py \
  --adata_path data/GSE103322.h5ad \
  --results_dir results \
  --dataset GSE103322_scVI_align \
  --n_topics 50 \
  --label_key cell_type

Alternatively, provide the cell-topic file directly via --cell_topic_file.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DEFAULT_LABEL_CANDIDATES: List[str] = [
    "cell_type",
    "celltype",
    "celltype.l1",
    "celltype.l2",
    "label",
    "labels",
    "cluster",
    "clusters",
]


@dataclass
class EvalConfig:
    adata_path: str
    results_dir: Optional[str] = None
    dataset: Optional[str] = None
    n_topics: Optional[int] = None
    cell_topic_file: Optional[str] = None
    label_key: Optional[str] = None
    res_min: float = 0.0
    res_max: float = 2.0
    res_step: float = 0.1
    out_dir: Optional[str] = None
    seed: int = 0


def _resolve_cell_topic_path(cfg: EvalConfig) -> Path:
    if cfg.cell_topic_file:
        return Path(cfg.cell_topic_file)
    if not (cfg.results_dir and cfg.dataset and cfg.n_topics is not None):
        raise ValueError(
            "Must provide either --cell_topic_file or all of --results_dir, --dataset, --n_topics"
        )
    return Path(cfg.results_dir) / "cell_topic" / f"{cfg.dataset}_cell_topic_matrix_{cfg.n_topics}.pkl"


def _load_cell_topic_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        m = pickle.load(f)
    return np.asarray(m)


def _pick_label_key(adata, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in adata.obs.columns:
        return preferred
    for k in DEFAULT_LABEL_CANDIDATES:
        if k in adata.obs.columns:
            return k
    return None


def _scan_resolutions_for_best_ari(
    adata: "sc.AnnData", label_key: str, res_min: float, res_max: float, res_step: float, seed: int
) -> Tuple[float, float, float]:
    """Returns (best_resolution, best_ari, nmi_at_best)."""
    # Ensure neighbors are present
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X")

    res = res_min
    best_res = res_min
    best_ari = -1.0
    best_nmi = -1.0
    tried: List[Tuple[float, float]] = []

    # Use Louvain if available; otherwise fall back to Leiden
    use_louvain = True
    try:
        import louvain as _  # noqa: F401
    except Exception:
        use_louvain = False

    while res <= res_max + 1e-9:
        if use_louvain:
            sc.tl.louvain(adata, resolution=res, random_state=seed, key_added="_tmp_clust")
        else:
            sc.tl.leiden(adata, resolution=res, random_state=seed, key_added="_tmp_clust")

        ari = float(adjusted_rand_score(adata.obs[label_key], adata.obs["_tmp_clust"]))
        tried.append((res, ari))
        if ari > best_ari:
            best_ari = ari
            best_res = res
        res = res + res_step

    # Refit at best resolution to compute NMI on final clustering
    if use_louvain:
        sc.tl.louvain(adata, resolution=best_res, random_state=seed, key_added="_best_clust")
    else:
        sc.tl.leiden(adata, resolution=best_res, random_state=seed, key_added="_best_clust")
    best_nmi = float(normalized_mutual_info_score(adata.obs[label_key], adata.obs["_best_clust"]))

    return best_res, best_ari, best_nmi


def evaluate(cfg: EvalConfig) -> Dict[str, float]:
    # Load inputs
    cell_topic_path = _resolve_cell_topic_path(cfg)
    if not cell_topic_path.exists():
        raise FileNotFoundError(f"Cell-topic file not found: {cell_topic_path}")
    X = _load_cell_topic_matrix(cell_topic_path)

    adata_orig = sc.read_h5ad(cfg.adata_path)

    # Align counts of cells
    n = min(X.shape[0], adata_orig.n_obs)
    if n <= 0:
        raise ValueError("No overlapping cells between cell-topic matrix and .h5ad")
    X = X[:n]
    labels_df = adata_orig.obs.iloc[:n].copy()

    # Pick label key and build anndata for clustering
    label_key = _pick_label_key(adata_orig, cfg.label_key)
    if label_key is None:
        raise KeyError(
            f"Could not find a label column. Tried: {[cfg.label_key] if cfg.label_key else []} + {DEFAULT_LABEL_CANDIDATES}"
        )

    adata = sc.AnnData(X)
    adata.obs[label_key] = labels_df[label_key].values

    # Compute metrics via resolution scan
    best_res, ari, nmi = _scan_resolutions_for_best_ari(
        adata=adata,
        label_key=label_key,
        res_min=cfg.res_min,
        res_max=cfg.res_max,
        res_step=cfg.res_step,
        seed=cfg.seed,
    )

    metrics = {
        "best_resolution": float(best_res),
        "ARI": float(ari),
        "NMI": float(nmi),
        "n_cells": int(n),
        "n_topics": int(X.shape[1]) if X.ndim == 2 else int(cfg.n_topics or -1),
        "dataset": cfg.dataset if cfg.dataset else "",
    }

    # Persist optional json
    out_dir = Path(cfg.out_dir) if cfg.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = cfg.dataset or cell_topic_path.stem
        nt = cfg.n_topics if cfg.n_topics is not None else metrics["n_topics"]
        out_path = out_dir / f"{ds}_cluster_metrics_{nt}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved evaluation metrics to {out_path}")

    # Console summary
    print("\n📈 Clustering Quality (Louvain/Leiden scan)")
    print("- Dataset:", metrics["dataset"]) if metrics["dataset"] else None
    print(f"- Cells: {metrics['n_cells']}")
    print(f"- Best resolution: {metrics['best_resolution']:.2f}")
    print(f"- ARI: {metrics['ARI']:.4f}")
    print(f"- NMI: {metrics['NMI']:.4f}")

    return metrics


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Compute ARI/NMI on cell-topic matrix (post-training)")
    p.add_argument("--adata_path", required=True, type=str, help="Path to input .h5ad with labels in .obs")
    p.add_argument("--results_dir", type=str, default=None, help="Results root (expects cell_topic/ subdir)")
    p.add_argument("--dataset", type=str, default=None, help="Dataset name used during training")
    p.add_argument("--n_topics", type=int, default=None, help="Number of topics used during training")
    p.add_argument("--cell_topic_file", type=str, default=None, help="Direct path to a cell_topic_matrix .pkl")
    p.add_argument("--label_key", type=str, default=None, help="Label column in .obs (defaults to common names)")
    p.add_argument("--res_min", type=float, default=0.0, help="Min Louvain/Leiden resolution")
    p.add_argument("--res_max", type=float, default=2.0, help="Max Louvain/Leiden resolution")
    p.add_argument("--res_step", type=float, default=0.1, help="Resolution step size")
    p.add_argument("--out_dir", type=str, default=None, help="Optional output dir for JSON metrics")
    p.add_argument("--seed", type=int, default=0, help="Random seed for clustering")
    a = p.parse_args()
    return EvalConfig(
        adata_path=a.adata_path,
        results_dir=a.results_dir,
        dataset=a.dataset,
        n_topics=a.n_topics,
        cell_topic_file=a.cell_topic_file,
        label_key=a.label_key,
        res_min=a.res_min,
        res_max=a.res_max,
        res_step=a.res_step,
        out_dir=a.out_dir,
        seed=a.seed,
    )


def main() -> int:
    cfg = parse_args()
    try:
        evaluate(cfg)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

