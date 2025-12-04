import os
import pickle
from typing import List, Optional, Sequence, Tuple

import glob
import numpy as np
import scanpy as sc

from incremental import TopicStore


def _find_single_file(patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        matches = sorted(
            (p for p in glob.glob(pat)), key=lambda p: os.path.getmtime(p), reverse=True
        )
        if matches:
            return matches[0]
    return None


def _load_cell_topic_matrix(results_dir: str, dataset: str) -> np.ndarray:
    """Load cell-topic matrix for a dataset from results_dir."""
    pat = os.path.join(
        results_dir, "cell_topic", f"{dataset}*cell_topic_matrix*.pkl"
    )
    matches = sorted(glob.glob(pat), key=lambda p: os.path.getmtime(p), reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find cell_topic matrix for dataset={dataset} under {results_dir}/cell_topic"
        )
    path = matches[0]
    with open(path, "rb") as f:
        mat = pickle.load(f)
    return np.asarray(mat, dtype=np.float32)


def _load_cell_types(
    adata_path: str,
    label_key: str,
    expected_n_cells: int,
    *,
    min_genes: int = 200,
) -> np.ndarray:
    """Load cell-type labels aligned with training-time cell filtering.

    We mimic the cell-level filtering in train_fastopic.preprocess_adata by
    applying sc.pp.filter_cells(min_genes=200) to the original AnnData object.
    The resulting order is then truncated or padded to match expected_n_cells.
    """
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    if label_key not in adata.obs:
        raise KeyError(
            f"Label key '{label_key}' not found in AnnData.obs for {adata_path}"
        )

    labels = adata.obs[label_key].astype(str).to_numpy()

    if labels.shape[0] >= expected_n_cells:
        return labels[:expected_n_cells]

    # If fewer labels than cells (should be rare), pad with 'unknown'.
    pad_n = expected_n_cells - labels.shape[0]
    pad = np.array(["unknown"] * pad_n, dtype=object)
    return np.concatenate([labels, pad], axis=0)


def _compute_filtered_topics_for_pair(
    dataset_a: str,
    dataset_b: str,
    results_dir_a: str,
    results_dir_b: str,
    *,
    filter_background: bool = True,
    sparsity_threshold: float = 0.20,
    topk_mass_threshold: Optional[float] = None,
    topk: int = 50,
) -> Tuple[List[int], List[int]]:
    """Use TopicStore.add_topics to obtain filtered topic indices for two datasets."""
    store = TopicStore()

    stats_a = store.add_topics(
        dataset_name=dataset_a,
        results_dir=results_dir_a,
        filter_background=filter_background,
        sparsity_threshold=sparsity_threshold,
        topk_mass_threshold=topk_mass_threshold,
        topk=topk,
        expand_genes=True,
        return_coupling=False,
    )
    filtered_a = list(stats_a.get("filtered", []))

    stats_b = store.add_topics(
        dataset_name=dataset_b,
        results_dir=results_dir_b,
        filter_background=filter_background,
        sparsity_threshold=sparsity_threshold,
        topk_mass_threshold=topk_mass_threshold,
        topk=topk,
        expand_genes=True,
        return_coupling=False,
    )
    filtered_b = list(stats_b.get("filtered", []))

    return filtered_a, filtered_b


def plot_batch_effect_umap(
    *,
    dataset_a: str,
    dataset_b: str,
    results_dir_a: str,
    results_dir_b: str,
    n_topics: int,
    out_dir: str,
    tag: str,
    adata_path_a: Optional[str] = None,
    adata_path_b: Optional[str] = None,
    label_key: Optional[str] = None,
    max_cells_per_batch: Optional[int] = None,
    filter_background: bool = True,
    sparsity_threshold: float = 0.20,
    topk_mass_threshold: float = -1.0,
    topk: int = 50,
) -> None:
    """Visualize batch effect before/after topic filtering using cell-topic matrices.

    This function:
      1) Uses TopicStore.add_topics to obtain filtered topic indices for each dataset.
      2) Loads the corresponding cell-topic matrices.
      3) Builds a combined AnnData with a 'batch' column.
      4) Runs UMAP before and after dropping filtered topics, and saves plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    sc.settings.figdir = out_dir
    sc.settings.autoshow = False

    # Compute filtered topics per dataset via TopicStore
    topk_mass_th: Optional[float] = None if topk_mass_threshold is None or topk_mass_threshold <= 0 else float(topk_mass_threshold)
    filtered_a, filtered_b = _compute_filtered_topics_for_pair(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        results_dir_a=results_dir_a,
        results_dir_b=results_dir_b,
        filter_background=filter_background,
        sparsity_threshold=sparsity_threshold,
        topk_mass_threshold=topk_mass_th,
        topk=topk,
    )

    # Union of filtered topics across both batches
    filtered_union = sorted(set(int(i) for i in filtered_a) | set(int(i) for i in filtered_b))
    all_idx = np.arange(n_topics, dtype=int)
    mask_keep = ~np.isin(all_idx, filtered_union)
    keep_indices = all_idx[mask_keep]

    if keep_indices.size == 0:
        raise ValueError(
            "Topic filtering removed all topics; cannot build post-filter UMAP."
        )

    # Load cell-topic matrices for both batches
    theta_a = _load_cell_topic_matrix(results_dir_a, dataset_a)
    theta_b = _load_cell_topic_matrix(results_dir_b, dataset_b)

    if theta_a.shape[1] != n_topics or theta_b.shape[1] != n_topics:
        raise ValueError(
            f"n_topics={n_topics} does not match cell_topic shapes: "
            f"{dataset_a}: {theta_a.shape[1]}, {dataset_b}: {theta_b.shape[1]}"
        )

    # Optional downsampling per batch for speed
    def _maybe_subsample(mat: np.ndarray, max_cells: Optional[int]) -> np.ndarray:
        if max_cells is None or mat.shape[0] <= max_cells:
            return mat
        idx = np.random.choice(mat.shape[0], size=max_cells, replace=False)
        return mat[idx]

    theta_a = _maybe_subsample(theta_a, max_cells_per_batch)
    theta_b = _maybe_subsample(theta_b, max_cells_per_batch)

    # Optional: load cell-type labels per batch (aligned to theta_* row counts).
    cell_types_a: Optional[np.ndarray] = None
    cell_types_b: Optional[np.ndarray] = None
    if adata_path_a and adata_path_b and label_key:
        cell_types_a = _load_cell_types(
            adata_path=adata_path_a,
            label_key=label_key,
            expected_n_cells=theta_a.shape[0],
        )
        cell_types_b = _load_cell_types(
            adata_path=adata_path_b,
            label_key=label_key,
            expected_n_cells=theta_b.shape[0],
        )
    if cell_types_a is not None and cell_types_b is not None:
        cell_types_all = np.concatenate([cell_types_a, cell_types_b], axis=0)
    else:
        cell_types_all = None

    # Build combined AnnData in topic space (pre-filter)
    theta_all = np.vstack([theta_a, theta_b])
    batch_labels = np.array(
        [dataset_a] * theta_a.shape[0] + [dataset_b] * theta_b.shape[0],
        dtype=object,
    )

    adata_pre = sc.AnnData(X=theta_all)
    adata_pre.obs["batch"] = batch_labels
    if cell_types_all is not None:
        adata_pre.obs["cell_type"] = cell_types_all

    # Pre-filter UMAP
    sc.pp.neighbors(adata_pre, use_rep="X", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata_pre, min_dist=0.3, random_state=0)
    sc.pl.umap(
        adata_pre,
        color="batch",
        show=False,
        save=f"_{tag}_pre_topic_filter_batch.png",
    )
    if cell_types_all is not None:
        sc.pl.umap(
            adata_pre,
            color="cell_type",
            show=False,
            save=f"_{tag}_pre_topic_filter_celltype.png",
        )

    # Post-filter UMAP: restrict to kept topic dimensions
    theta_post = theta_all[:, keep_indices]
    adata_post = sc.AnnData(X=theta_post)
    adata_post.obs["batch"] = batch_labels
    if cell_types_all is not None:
        adata_post.obs["cell_type"] = cell_types_all

    sc.pp.neighbors(adata_post, use_rep="X", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata_post, min_dist=0.3, random_state=0)
    sc.pl.umap(
        adata_post,
        color="batch",
        show=False,
        save=f"_{tag}_post_topic_filter_batch.png",
    )
    if cell_types_all is not None:
        sc.pl.umap(
            adata_post,
            color="cell_type",
            show=False,
            save=f"_{tag}_post_topic_filter_celltype.png",
        )


def _parse_args() -> "argparse.Namespace":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Visualize batch effect before/after incremental topic filtering "
            "using cell-topic matrices for two datasets."
        )
    )
    p.add_argument(
        "--dataset_a",
        required=True,
        help="Dataset label for batch A (e.g., PBMC4k_scVI_align).",
    )
    p.add_argument(
        "--dataset_b",
        required=True,
        help="Dataset label for batch B (e.g., PBMC8k_scVI_align).",
    )
    p.add_argument(
        "--results_dir_a",
        required=True,
        help="Results root directory for dataset A (e.g., results/tuning/PBMC4k_structure).",
    )
    p.add_argument(
        "--results_dir_b",
        required=True,
        help="Results root directory for dataset B (e.g., results/tuning/PBMC8k_structure).",
    )
    p.add_argument(
        "--n_topics",
        type=int,
        required=True,
        help="Number of topics used in single training.",
    )
    p.add_argument(
        "--out_dir",
        default="results/incremental_eval/batch_effect",
        help="Directory to save UMAP figures.",
    )
    p.add_argument(
        "--tag",
        default="PBMC12k",
        help="Tag to include in output figure filenames.",
    )
    p.add_argument(
        "--adata_a",
        help="Path to original .h5ad file for dataset A (used to load cell_type labels).",
    )
    p.add_argument(
        "--adata_b",
        help="Path to original .h5ad file for dataset B (used to load cell_type labels).",
    )
    p.add_argument(
        "--label_key",
        default="cell_type",
        help="Column in AnnData.obs to use as cell type labels (default: cell_type).",
    )
    p.add_argument(
        "--max_cells_per_batch",
        type=int,
        default=None,
        help="Optional maximum number of cells to sample per batch for plotting.",
    )
    # Topic filtering knobs (mirroring incremental.TopicStore.add_topics)
    p.add_argument(
        "--no_filter_background",
        action="store_true",
        help="Disable background-topic filtering (for debugging).",
    )
    p.add_argument(
        "--sparsity_threshold",
        type=float,
        default=0.20,
        help="Hoyer sparsity threshold to keep topics (0..1).",
    )
    p.add_argument(
        "--topk_mass_threshold",
        type=float,
        default=-1.0,
        help="Minimum mass in top-k genes (<=0 to disable).",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=50,
        help="k for top-k mass threshold.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    filter_background = not args.no_filter_background

    plot_batch_effect_umap(
        dataset_a=args.dataset_a,
        dataset_b=args.dataset_b,
        results_dir_a=args.results_dir_a,
        results_dir_b=args.results_dir_b,
        n_topics=args.n_topics,
        out_dir=args.out_dir,
        tag=args.tag,
        adata_path_a=args.adata_a,
        adata_path_b=args.adata_b,
        label_key=args.label_key,
        max_cells_per_batch=args.max_cells_per_batch,
        filter_background=filter_background,
        sparsity_threshold=args.sparsity_threshold,
        topk_mass_threshold=args.topk_mass_threshold,
        topk=args.topk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
