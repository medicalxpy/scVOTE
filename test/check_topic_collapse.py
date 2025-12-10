#!/usr/bin/env python3
"""
Quick diagnostics for topic collapse in scFASTopic runs.

Given a trained run (e.g., results/tuning/<RUN_TAG>), this script:
  1) Loads the cell-topic matrix and computes:
       - topic weight distribution (average topic usage across cells)
       - Shannon entropy / effective number of topics
       - dominant topic ratio (max topic weight)
  2) Loads topic embeddings and plots a simple topic hierarchy dendrogram.

The goal is to visually and quantitatively check whether tuning runs
have topic collapse (only a few topics carrying most of the mass).
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy as sch
from scipy.stats import entropy


def _load_cell_topic_matrix(results_dir: Path, dataset: str, n_topics: int) -> np.ndarray:
    """Load cell-topic matrix from a training run."""
    path = results_dir / "cell_topic" / f"{dataset}_cell_topic_matrix_{n_topics}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cell-topic matrix not found: {path}")
    with open(path, "rb") as f:
        mat = pickle.load(f)
    theta = np.asarray(mat, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != n_topics:
        raise ValueError(
            f"Cell-topic matrix has shape {theta.shape}, expected (*, {n_topics})."
        )
    return theta


def _load_topic_embeddings(results_dir: Path, dataset: str, n_topics: int) -> np.ndarray:
    """Load topic embeddings from a training run."""
    path = results_dir / "topic_embedding" / f"{dataset}_topic_embeddings_{n_topics}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Topic embeddings not found: {path}")
    with open(path, "rb") as f:
        mat = pickle.load(f)
    topic_emb = np.asarray(mat, dtype=np.float64)
    if topic_emb.ndim != 2 or topic_emb.shape[0] != n_topics:
        raise ValueError(
            f"Topic embeddings have shape {topic_emb.shape}, expected ({n_topics}, D)."
        )
    return topic_emb


def _compute_topic_weight_metrics(theta: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute topic weights and collapse diagnostics.

    Follows the logic in train_fastopic.train_fastopic_model:
      - sanitize theta
      - row-normalize
      - average over cells to get topic_weights
      - compute Shannon entropy, effective topics, dominant ratio
    """
    theta_sane = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = theta_sane.sum(axis=1, keepdims=True)
    if row_sum.ndim == 1:
        row_sum = row_sum.reshape(-1, 1)
    zero_rows = row_sum <= 0
    if np.any(zero_rows):
        theta_sane[zero_rows[:, 0]] = 1.0 / max(1, theta_sane.shape[1])
        row_sum = theta_sane.sum(axis=1, keepdims=True)
    theta_sane = theta_sane / np.maximum(row_sum, 1e-12)

    topic_weights = theta_sane.mean(axis=0)
    topic_weights = np.clip(topic_weights, 0.0, None)
    topic_weights = topic_weights / np.maximum(topic_weights.sum(), 1e-12)

    shannon_entropy = float(entropy(topic_weights + 1e-12, base=2))
    effective_topics = float(2**shannon_entropy)
    max_topic_weight = float(topic_weights.max() if topic_weights.size else 0.0)
    dominant_ratio = float(max_topic_weight * 100.0)
    return topic_weights, shannon_entropy, effective_topics, dominant_ratio


def _plot_topic_weights(
    topic_weights: np.ndarray,
    dataset: str,
    n_topics: int,
    out_path: Path,
) -> None:
    """Save a bar plot of topic weights."""
    k = topic_weights.shape[0]
    x = np.arange(k)

    plt.figure(figsize=(10, 4))
    plt.bar(x, topic_weights, color="#C8D2D7", edgecolor="#6E8484", linewidth=0.8)
    plt.xlabel("Topic index")
    plt.ylabel("Average topic weight")
    plt.title(f"Topic weight distribution: {dataset} (K={n_topics})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_topic_hierarchy(
    topic_embeddings: np.ndarray,
    dataset: str,
    n_topics: int,
    out_path: Path,
) -> None:
    """Save a simple dendrogram over topic embeddings."""
    # Ward linkage; optimal ordering for nicer plots.
    Z = sch.linkage(topic_embeddings, method="ward", optimal_ordering=True)
    labels = [str(i) for i in range(topic_embeddings.shape[0])]

    plt.figure(figsize=(10, 6))
    sch.dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(f"Topic hierarchy (Ward linkage): {dataset} (K={n_topics})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check topic collapse via topic weights and hierarchy for a scFASTopic run."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root results directory for the run (e.g., results or results/tuning/<run_tag>).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name used in training (e.g., PBMC4k_scVI_align).",
    )
    p.add_argument(
        "--n_topics",
        type=int,
        required=True,
        help="Number of topics used in training.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for diagnostic plots (default: <results_dir>/topic_diagnostics).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "topic_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Using results_dir: {results_dir}")
    print(f"📁 Saving diagnostics to: {out_dir}")

    # Load matrices
    theta = _load_cell_topic_matrix(results_dir, args.dataset, args.n_topics)
    topic_emb = _load_topic_embeddings(results_dir, args.dataset, args.n_topics)

    # Topic weight metrics
    topic_weights, H, eff_k, dom_ratio = _compute_topic_weight_metrics(theta)

    print("\n📊 Topic weight diagnostics")
    print(f"- Dataset: {args.dataset}")
    print(f"- K (topics): {args.n_topics}")
    print(f"- Shannon entropy (bits): {H:.3f}")
    print(f"- Effective topics: {eff_k:.2f}")
    print(f"- Dominant topic ratio: {dom_ratio:.1f}%")
    print(f"- Max topic weight: {topic_weights.max():.4f}")

    # Save topic weight bar plot
    wt_path = out_dir / f"{args.dataset}_K{args.n_topics}_topic_weights.png"
    _plot_topic_weights(topic_weights, args.dataset, args.n_topics, wt_path)
    print(f"💾 Saved topic weight plot: {wt_path}")

    # Save topic hierarchy dendrogram
    hier_path = out_dir / f"{args.dataset}_K{args.n_topics}_topic_hierarchy.png"
    _plot_topic_hierarchy(topic_emb, args.dataset, args.n_topics, hier_path)
    print(f"💾 Saved topic hierarchy plot: {hier_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

