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
from sklearn.manifold import TSNE


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


def _load_gene_embeddings(results_dir: Path, dataset: str, n_topics: int) -> np.ndarray:
    """Load word/gene embeddings from a training run (if available)."""
    path = results_dir / "gene_embedding" / f"{dataset}_gene_embeddings_{n_topics}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Gene embeddings not found: {path}")
    with open(path, "rb") as f:
        mat = pickle.load(f)
    word_emb = np.asarray(mat, dtype=np.float64)
    if word_emb.ndim != 2:
        raise ValueError(
            f"Gene embeddings have shape {word_emb.shape}, expected (V, D)."
        )
    return word_emb


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
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def _plot_tsne_word_topic(
    word_emb: np.ndarray,
    topic_emb: np.ndarray,
    dataset: str,
    n_topics: int,
    out_path: Path,
    max_words: int = 2000,
    normalize: bool = False,
    random_state: int = 0,
) -> None:
    """
    Joint t-SNE visualization of word/gene embeddings (•) and topic embeddings (▲).

    We subsample up to max_words word vectors to keep t-SNE tractable.
    """
    n_words = word_emb.shape[0]
    # Subsample word embeddings if needed
    if n_words > max_words:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_words, size=max_words, replace=False)
        word_emb_sub = word_emb[idx]
    else:
        word_emb_sub = word_emb

    # Stack word + topic embeddings
    X = np.vstack([word_emb_sub, topic_emb])
    if normalize:
        X = _l2_normalize_rows(X)
    n_word_sub = word_emb_sub.shape[0]
    n_total = X.shape[0]

    # Perplexity must be < n_total
    perp = min(30.0, max(5.0, (n_total - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X)

    word_xy = X_2d[:n_word_sub]
    topic_xy = X_2d[n_word_sub:]

    plt.figure(figsize=(6, 6))
    # Words / genes as blue dots
    plt.scatter(
        word_xy[:, 0],
        word_xy[:, 1],
        s=2,
        c="tab:blue",
        alpha=0.4,
        linewidths=0,
        label="Genes",
    )
    # Topics as red triangles
    plt.scatter(
        topic_xy[:, 0],
        topic_xy[:, 1],
        s=40,
        c="tab:red",
        marker="^",
        edgecolor="k",
        linewidths=0.5,
        label="Topics",
    )
    plt.axis("off")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _load_cell_embeddings(path: Path) -> np.ndarray:
    """Load cell/document embeddings from a .pkl file."""
    if not path.exists():
        raise FileNotFoundError(f"Cell embeddings not found: {path}")
    with open(path, "rb") as f:
        mat = pickle.load(f)
    cell_emb = np.asarray(mat, dtype=np.float64)
    if cell_emb.ndim != 2:
        raise ValueError(
            f"Cell embeddings have shape {cell_emb.shape}, expected (N, D)."
        )
    return cell_emb


def _plot_tsne_cell_topic(
    cell_emb: np.ndarray,
    topic_emb: np.ndarray,
    dataset: str,
    n_topics: int,
    out_path: Path,
    max_cells: int = 5000,
    normalize: bool = False,
    random_state: int = 0,
) -> None:
    """
    Joint t-SNE visualization of cell embeddings (•) and topic embeddings (▲).

    This mirrors Fig. 3(c,d) style: many document points plus topic centroids.
    """
    n_cells = cell_emb.shape[0]
    if n_cells > max_cells:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_cells, size=max_cells, replace=False)
        cell_emb_sub = cell_emb[idx]
    else:
        cell_emb_sub = cell_emb

    X = np.vstack([cell_emb_sub, topic_emb])
    if normalize:
        X = _l2_normalize_rows(X)
    n_cell_sub = cell_emb_sub.shape[0]
    n_total = X.shape[0]

    perp = min(30.0, max(5.0, (n_total - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X)

    cell_xy = X_2d[:n_cell_sub]
    topic_xy = X_2d[n_cell_sub:]

    plt.figure(figsize=(6, 6))
    # Cells/documents as purple points
    plt.scatter(
        cell_xy[:, 0],
        cell_xy[:, 1],
        s=2,
        c="tab:purple",
        alpha=0.5,
        linewidths=0,
        label="Cells",
    )
    # Topics as red triangles
    plt.scatter(
        topic_xy[:, 0],
        topic_xy[:, 1],
        s=40,
        c="tab:red",
        marker="^",
        edgecolor="k",
        linewidths=0.5,
        label="Topics",
    )
    plt.axis("off")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_tsne_cell_gene_topic(
    cell_emb: np.ndarray,
    word_emb: np.ndarray,
    topic_emb: np.ndarray,
    dataset: str,
    n_topics: int,
    out_path: Path,
    max_cells: int = 5000,
    max_genes: int = 2000,
    normalize: bool = False,
    random_state: int = 0,
) -> None:
    """
    Joint t-SNE of cell, gene, and topic embeddings on a single 2D plane.

    - Cells: purple points
    - Genes: blue points
    - Topics: red triangles
    """
    # Subsample cells
    n_cells = cell_emb.shape[0]
    if n_cells > max_cells:
        rng = np.random.default_rng(random_state)
        idx_cells = rng.choice(n_cells, size=max_cells, replace=False)
        cell_emb_sub = cell_emb[idx_cells]
    else:
        cell_emb_sub = cell_emb

    # Subsample genes
    n_genes = word_emb.shape[0]
    if n_genes > max_genes:
        rng = np.random.default_rng(random_state + 1)
        idx_genes = rng.choice(n_genes, size=max_genes, replace=False)
        word_emb_sub = word_emb[idx_genes]
    else:
        word_emb_sub = word_emb

    X = np.vstack([cell_emb_sub, word_emb_sub, topic_emb])
    if normalize:
        X = _l2_normalize_rows(X)
    n_cell_sub = cell_emb_sub.shape[0]
    n_word_sub = word_emb_sub.shape[0]
    n_total = X.shape[0]

    perp = min(30.0, max(5.0, (n_total - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X)

    cell_xy = X_2d[:n_cell_sub]
    word_xy = X_2d[n_cell_sub:n_cell_sub + n_word_sub]
    topic_xy = X_2d[n_cell_sub + n_word_sub:]

    plt.figure(figsize=(6, 6))

    # Cells as purple points
    plt.scatter(
        cell_xy[:, 0],
        cell_xy[:, 1],
        s=2,
        c="tab:purple",
        alpha=0.5,
        linewidths=0,
        label="Cells",
    )

    # Genes as blue points
    plt.scatter(
        word_xy[:, 0],
        word_xy[:, 1],
        s=2,
        c="tab:blue",
        alpha=0.4,
        linewidths=0,
        label="Genes",
    )

    # Topics as red triangles
    plt.scatter(
        topic_xy[:, 0],
        topic_xy[:, 1],
        s=40,
        c="tab:red",
        marker="^",
        edgecolor="k",
        linewidths=0.5,
        label="Topics",
    )

    plt.axis("off")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
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
    p.add_argument(
        "--no_tsne",
        action="store_true",
        help="Disable t-SNE visualization of gene/topic embeddings.",
    )
    p.add_argument(
        "--cell_emb",
        type=str,
        default=None,
        help="Optional path to cell/document embeddings (.pkl) for cell-topic t-SNE.",
    )
    p.add_argument(
        "--max_cells_tsne",
        type=int,
        default=5000,
        help="Maximum number of cells to subsample for cell-topic t-SNE (default: 5000).",
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

    # Joint t-SNE of word/topic embeddings
    if not args.no_tsne:
        word_emb = None
        try:
            word_emb = _load_gene_embeddings(results_dir, args.dataset, args.n_topics)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Skipping t-SNE (gene embeddings unavailable): {exc}")
        else:
            # Raw scale
            tsne_path = out_dir / f"{args.dataset}_K{args.n_topics}_tsne_gene_topic.png"
            print("🌀 Running t-SNE (raw) for gene/topic embeddings (this may take a while)...")
            _plot_tsne_word_topic(
                word_emb=word_emb,
                topic_emb=topic_emb,
                dataset=args.dataset,
                n_topics=args.n_topics,
                out_path=tsne_path,
            )
            print(f"💾 Saved t-SNE plot: {tsne_path}")

            # L2-normalized
            tsne_path_norm = out_dir / f"{args.dataset}_K{args.n_topics}_tsne_gene_topic_norm.png"
            print("🌀 Running t-SNE (L2-normalized) for gene/topic embeddings...")
            _plot_tsne_word_topic(
                word_emb=word_emb,
                topic_emb=topic_emb,
                dataset=args.dataset,
                n_topics=args.n_topics,
                out_path=tsne_path_norm,
                normalize=True,
            )
            print(f"💾 Saved t-SNE plot: {tsne_path_norm}")

        # Optional: joint t-SNE of cell/topic embeddings, if cell_emb is provided.
        if args.cell_emb:
            try:
                cell_emb = _load_cell_embeddings(Path(args.cell_emb))
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Skipping t-SNE (cell embeddings unavailable): {exc}")
            else:
                # Raw scale
                tsne_cell_path = out_dir / f"{args.dataset}_K{args.n_topics}_tsne_cell_topic.png"
                print("🌀 Running t-SNE (raw) for cell/topic embeddings (this may take a while)...")
                _plot_tsne_cell_topic(
                    cell_emb=cell_emb,
                    topic_emb=topic_emb,
                    dataset=args.dataset,
                    n_topics=args.n_topics,
                    out_path=tsne_cell_path,
                    max_cells=args.max_cells_tsne,
                )
                print(f"💾 Saved t-SNE plot: {tsne_cell_path}")

                # L2-normalized
                tsne_cell_path_norm = out_dir / f"{args.dataset}_K{args.n_topics}_tsne_cell_topic_norm.png"
                print("🌀 Running t-SNE (L2-normalized) for cell/topic embeddings...")
                _plot_tsne_cell_topic(
                    cell_emb=cell_emb,
                    topic_emb=topic_emb,
                    dataset=args.dataset,
                    n_topics=args.n_topics,
                    out_path=tsne_cell_path_norm,
                    max_cells=args.max_cells_tsne,
                    normalize=True,
                )
                print(f"💾 Saved t-SNE plot: {tsne_cell_path_norm}")

                # If both cell and gene embeddings are available, also run
                # combined cell+gene+topic t-SNE.
                if word_emb is not None:
                    # Raw scale
                    tsne_all_path = out_dir / f"{args.dataset}_K{args.n_topics}_tsne_cell_gene_topic.png"
                    print("🌀 Running t-SNE (raw) for cell/gene/topic embeddings (this may take a while)...")
                    _plot_tsne_cell_gene_topic(
                        cell_emb=cell_emb,
                        word_emb=word_emb,
                        topic_emb=topic_emb,
                        dataset=args.dataset,
                        n_topics=args.n_topics,
                        out_path=tsne_all_path,
                        max_cells=args.max_cells_tsne,
                        max_genes=2000,
                    )
                    print(f"💾 Saved t-SNE plot: {tsne_all_path}")

                    # L2-normalized
                    tsne_all_path_norm = out_dir / f"{args.dataset}_K{args.n_topics}_tsne_cell_gene_topic_norm.png"
                    print("🌀 Running t-SNE (L2-normalized) for cell/gene/topic embeddings...")
                    _plot_tsne_cell_gene_topic(
                        cell_emb=cell_emb,
                        word_emb=word_emb,
                        topic_emb=topic_emb,
                        dataset=args.dataset,
                        n_topics=args.n_topics,
                        out_path=tsne_all_path_norm,
                        max_cells=args.max_cells_tsne,
                        max_genes=2000,
                        normalize=True,
                    )
                    print(f"💾 Saved t-SNE plot: {tsne_all_path_norm}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
