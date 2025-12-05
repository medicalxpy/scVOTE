#!/usr/bin/env python3
import os
import pickle
from typing import Dict, List, Optional, Tuple

import glob
import numpy as np
import scanpy as sc


def _find_latest(pattern: str) -> str:
    """Return the most recent file matching a glob pattern."""
    matches = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No file matching pattern: {pattern}")
    return matches[0]


def _load_topic_gene_matrix(results_dir: str, dataset: str, n_topics: int) -> np.ndarray:
    """Load topic-gene matrix (beta) for a dataset."""
    pat = os.path.join(
        results_dir, "topic_gene", f"{dataset}*topic_gene_matrix*.pkl"
    )
    path = _find_latest(pat)
    with open(path, "rb") as f:
        mat = pickle.load(f)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[0] != n_topics:
        raise ValueError(
            f"Topic-gene matrix for {dataset} has {mat.shape[0]} topics, "
            f"but n_topics={n_topics} was requested."
        )
    return mat


def _load_cell_topic_matrix(results_dir: str, dataset: str, n_topics: int) -> np.ndarray:
    """Load cell-topic matrix (theta) for a dataset."""
    pat = os.path.join(
        results_dir, "cell_topic", f"{dataset}*cell_topic_matrix*.pkl"
    )
    path = _find_latest(pat)
    with open(path, "rb") as f:
        mat = pickle.load(f)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[1] != n_topics:
        raise ValueError(
            f"Cell-topic matrix for {dataset} has {mat.shape[1]} topics, "
            f"but n_topics={n_topics} was requested."
        )
    return mat


def _load_labels(
    adata_path: str,
    batch_key: Optional[str],
    celltype_key: Optional[str],
    expected_n_cells: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load batch and cell-type labels and align length with expected_n_cells."""
    adata = sc.read_h5ad(adata_path)
    # Mirror basic cell filtering from train_fastopic.preprocess_adata
    sc.pp.filter_cells(adata, min_genes=200)

    batch: Optional[np.ndarray] = None
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(
                f"Batch key '{batch_key}' not found in AnnData.obs for {adata_path}"
            )
        b = adata.obs[batch_key].astype(str).to_numpy()
        if b.shape[0] >= expected_n_cells:
            batch = b[:expected_n_cells]
        else:
            pad_n = expected_n_cells - b.shape[0]
            pad = np.array(["unknown_batch"] * pad_n, dtype=object)
            batch = np.concatenate([b, pad], axis=0)

    cell_type: Optional[np.ndarray] = None
    if celltype_key is not None:
        if celltype_key not in adata.obs:
            raise KeyError(
                f"Cell-type key '{celltype_key}' not found in AnnData.obs for {adata_path}"
            )
        ct = adata.obs[celltype_key].astype(str).to_numpy()
        if ct.shape[0] >= expected_n_cells:
            cell_type = ct[:expected_n_cells]
        else:
            pad_n = expected_n_cells - ct.shape[0]
            pad = np.array(["unknown"] * pad_n, dtype=object)
            cell_type = np.concatenate([ct, pad], axis=0)

    return batch, cell_type


def _find_duplicate_topics(beta: np.ndarray, top_n: int = 15) -> Dict[int, List[int]]:
    """Group topics whose top-N genes (by weight) are exactly the same.

    Two topics are considered duplicates if the *set* of indices of their
    top-N genes is identical. N is configurable via top_n (default: 15).
    """
    K, G = beta.shape
    n = int(max(1, min(top_n, G)))

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for k in range(K):
        row = beta[k]
        if G <= n:
            top_idx = np.argsort(row)[-n:]
        else:
            # Use argpartition to get top-n, then sort for deterministic key.
            idx_part = np.argpartition(row, -n)[-n:]
            top_idx = idx_part[np.argsort(row[idx_part])]
        key = tuple(sorted(int(i) for i in top_idx))
        groups.setdefault(key, []).append(k)

    dup_groups: Dict[int, List[int]] = {}
    for indices in groups.values():
        if len(indices) > 1:
            indices_sorted = sorted(indices)
            rep = indices_sorted[0]
            dup_groups[rep] = indices_sorted
    return dup_groups


def _merge_topics(
    theta: np.ndarray,
    beta: np.ndarray,
    dup_groups: Dict[int, List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate topics by summing their theta columns and keeping one beta row."""
    K_old = beta.shape[0]
    keep_mask = np.ones(K_old, dtype=bool)
    for rep, idxs in dup_groups.items():
        for j in idxs:
            if j != rep:
                keep_mask[j] = False

    keep_indices = np.where(keep_mask)[0]
    # Map representative to position in new order
    rep_to_newpos: Dict[int, int] = {old: i for i, old in enumerate(keep_indices)}

    theta_new = np.zeros((theta.shape[0], keep_indices.shape[0]), dtype=theta.dtype)
    beta_new = np.zeros((keep_indices.shape[0], beta.shape[1]), dtype=beta.dtype)

    for old_idx in keep_indices:
        new_pos = rep_to_newpos[old_idx]
        # Start with its own column/row
        theta_new[:, new_pos] = theta[:, old_idx]
        beta_new[new_pos] = beta[old_idx]

        # Add any duplicates assigned to this representative
        group = dup_groups.get(old_idx)
        if group is not None:
            for j in group:
                if j == old_idx:
                    continue
                theta_new[:, new_pos] += theta[:, j]

    # Optionally renormalise theta rows to sum to 1 (guard against zeros).
    row_sum = theta_new.sum(axis=1, keepdims=True)
    nonzero = row_sum[:, 0] > 0
    theta_new[nonzero] = theta_new[nonzero] / row_sum[nonzero]

    return theta_new, beta_new


def run_merge_and_plot(
    *,
    dataset: str,
    results_dir: str,
    adata_path: str,
    n_topics: int,
    out_dir: str,
    tag: str,
    batch_key: Optional[str] = None,
    celltype_key: Optional[str] = None,
    max_cells: Optional[int] = None,
    top_n: int = 15,
) -> None:
    """Merge duplicate topics (identical topic-gene vectors) and plot UMAP pre/post."""
    os.makedirs(out_dir, exist_ok=True)
    sc.settings.figdir = out_dir
    sc.settings.autoshow = False

    beta = _load_topic_gene_matrix(results_dir, dataset, n_topics=n_topics)
    theta = _load_cell_topic_matrix(results_dir, dataset, n_topics=n_topics)

    K_old = beta.shape[0]
    dup_groups = _find_duplicate_topics(beta, top_n=top_n)

    print(f"Original number of topics: {K_old}")
    if not dup_groups:
        print("No duplicate topic-gene vectors found (up to rounding).")
        # Still produce pre-merge UMAP for inspection.
        theta_ = theta
        if max_cells is not None and theta_.shape[0] > max_cells:
            idx = np.random.choice(theta_.shape[0], size=max_cells, replace=False)
            theta_ = theta_[idx]
            effective_n_cells = theta_.shape[0]
        else:
            effective_n_cells = theta_.shape[0]
        batch, cell_type = _load_labels(
            adata_path=adata_path,
            batch_key=batch_key,
            celltype_key=celltype_key,
            expected_n_cells=effective_n_cells,
        )
        adata_pre = sc.AnnData(X=theta_)
        if batch is not None:
            adata_pre.obs["batch"] = batch
        if cell_type is not None:
            adata_pre.obs["cell_type"] = cell_type
        sc.pp.neighbors(adata_pre, use_rep="X", n_neighbors=15, metric="euclidean")
        sc.tl.umap(adata_pre, min_dist=0.3, random_state=0)
        if batch is not None:
            sc.pl.umap(
                adata_pre,
                color="batch",
                show=False,
                save=f"_{tag}_pre_merge_batch.png",
            )
        if cell_type is not None:
            sc.pl.umap(
                adata_pre,
                color="cell_type",
                show=False,
                save=f"_{tag}_pre_merge_celltype.png",
            )
        return

    # Merge duplicates
    theta_merged, beta_merged = _merge_topics(theta, beta, dup_groups)
    K_new = beta_merged.shape[0]
    print(f"Merged number of topics: {K_new}")
    print(f"Merged {K_old - K_new} duplicate topics into {K_new} unique topics.")

    # Optionally downsample cells for UMAP
    def _maybe_subsample(mat: np.ndarray, max_n: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        if max_n is None or mat.shape[0] <= max_n:
            idx = np.arange(mat.shape[0], dtype=int)
            return mat, idx
        idx = np.random.choice(mat.shape[0], size=max_n, replace=False)
        return mat[idx], idx

    theta_pre, idx_pre = _maybe_subsample(theta, max_cells)
    theta_post, idx_post = _maybe_subsample(theta_merged, max_cells)

    # Load labels and align to sampled cells
    batch_all, cell_type_all = _load_labels(
        adata_path=adata_path,
        batch_key=batch_key,
        celltype_key=celltype_key,
        expected_n_cells=theta.shape[0],
    )
    batch_pre = batch_all[idx_pre] if batch_all is not None else None
    batch_post = batch_all[idx_post] if batch_all is not None else None
    cell_type_pre = cell_type_all[idx_pre] if cell_type_all is not None else None
    cell_type_post = cell_type_all[idx_post] if cell_type_all is not None else None

    # Pre-merge UMAP
    adata_pre = sc.AnnData(X=theta_pre)
    if batch_pre is not None:
        adata_pre.obs["batch"] = batch_pre
    if cell_type_pre is not None:
        adata_pre.obs["cell_type"] = cell_type_pre

    sc.pp.neighbors(adata_pre, use_rep="X", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata_pre, min_dist=0.3, random_state=0)
    if batch_pre is not None:
        sc.pl.umap(
            adata_pre,
            color="batch",
            show=False,
            save=f"_{tag}_pre_merge_batch.png",
        )
    if cell_type_pre is not None:
        sc.pl.umap(
            adata_pre,
            color="cell_type",
            show=False,
            save=f"_{tag}_pre_merge_celltype.png",
        )

    # Post-merge UMAP
    adata_post = sc.AnnData(X=theta_post)
    if batch_post is not None:
        adata_post.obs["batch"] = batch_post
    if cell_type_post is not None:
        adata_post.obs["cell_type"] = cell_type_post

    sc.pp.neighbors(adata_post, use_rep="X", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata_post, min_dist=0.3, random_state=0)
    if batch_post is not None:
        sc.pl.umap(
            adata_post,
            color="batch",
            show=False,
            save=f"_{tag}_post_merge_batch.png",
        )
    if cell_type_post is not None:
        sc.pl.umap(
            adata_post,
            color="cell_type",
            show=False,
            save=f"_{tag}_post_merge_celltype.png",
        )


def _parse_args() -> "argparse.Namespace":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Merge topics with identical topic-gene vectors into a single topic "
            "and visualise UMAP before and after merging."
        )
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset label used in training (e.g., PBMC12k_scVI_align or PBMC4k_scVI_align).",
    )
    p.add_argument(
        "--results_dir",
        required=True,
        help="Results root directory for the dataset (e.g., results or results/tuning/<run>).",
    )
    p.add_argument(
        "--adata",
        required=True,
        help="Path to original .h5ad file for this dataset.",
    )
    p.add_argument(
        "--n_topics",
        type=int,
        required=True,
        help="Number of topics used in single training.",
    )
    p.add_argument(
        "--out_dir",
        default="results/incremental_eval/merge_duplicates",
        help="Directory to save UMAP figures.",
    )
    p.add_argument(
        "--tag",
        default="merge_duplicates",
        help="Tag to include in output figure filenames.",
    )
    p.add_argument(
        "--batch_key",
        default=None,
        help="Obs column name to use as batch labels (optional).",
    )
    p.add_argument(
        "--celltype_key",
        default=None,
        help="Obs column name to use as cell type labels (optional).",
    )
    p.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Optional maximum number of cells to sample for plotting.",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Number of top genes per topic used to detect duplicate topics (default: 15).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_merge_and_plot(
        dataset=args.dataset,
        results_dir=args.results_dir,
        adata_path=args.adata,
        n_topics=args.n_topics,
        out_dir=args.out_dir,
        tag=args.tag,
        batch_key=args.batch_key,
        celltype_key=args.celltype_key,
        max_cells=args.max_cells,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
