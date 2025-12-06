import os
import pickle
from typing import Optional

import glob
import numpy as np
import scanpy as sc

from incremental import TopicStore


def _find_single_file(pattern: str) -> str:
    matches = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No file matching pattern: {pattern}")
    return matches[0]


def _load_cell_topic_matrix(results_dir: str, dataset: str, n_topics: int) -> np.ndarray:
    topic_dir = os.path.join(results_dir, "cell_topic")
    pat_exact = os.path.join(
        topic_dir, f"{dataset}*cell_topic_matrix_{n_topics}.pkl"
    )
    matches_exact = sorted(glob.glob(pat_exact), key=lambda p: os.path.getmtime(p), reverse=True)
    if matches_exact:
        path = matches_exact[0]
    else:
        pat_any = os.path.join(topic_dir, f"{dataset}*cell_topic_matrix*.pkl")
        path = _find_single_file(pat_any)
    with open(path, "rb") as f:
        mat = pickle.load(f)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[1] != n_topics:
        raise ValueError(
            f"Cell-topic matrix for {dataset} has {mat.shape[1]} topics, "
            f"but n_topics={n_topics} was requested."
        )
    return mat


def _load_obs_labels(
    adata_path: str,
    batch_key: str,
    celltype_key: Optional[str],
    expected_n_cells: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load batch and optional cell-type labels aligned with training-time cell filtering.

    We mimic the cell-level filtering in train_fastopic.preprocess_adata by
    applying sc.pp.filter_cells(min_genes=200) to the original AnnData object
    and then truncating/padding to match expected_n_cells.
    """
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=200)

    if batch_key not in adata.obs:
        raise KeyError(
            f"Batch key '{batch_key}' not found in AnnData.obs for {adata_path}"
        )

    batch = adata.obs[batch_key].astype(str).to_numpy()
    if batch.shape[0] >= expected_n_cells:
        batch = batch[:expected_n_cells]
    else:
        pad_n = expected_n_cells - batch.shape[0]
        pad = np.array(["unknown_batch"] * pad_n, dtype=object)
        batch = np.concatenate([batch, pad], axis=0)

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


def plot_batch_effect_umap_single(
    *,
    dataset: str,
    results_dir: str,
    adata_path: str,
    batch_key: str,
    celltype_key: Optional[str],
    n_topics: int,
    out_dir: str,
    tag: str,
    max_cells: Optional[int] = None,
    filter_background: bool = True,
    sparsity_threshold: float = 0.20,
    topk_mass_threshold: float = -1.0,
    topk: int = 50,
    coherence_top_n: int = 20,
    coherence_threshold: float = 0.20,
) -> None:
    """Visualize batch effect before/after topic filtering for a single dataset.

    Uses TopicStore.add_topics to obtain filtered topics for the dataset, and
    then applies the same topic filtering to its cell-topic matrix.
    """
    os.makedirs(out_dir, exist_ok=True)
    sc.settings.figdir = out_dir
    sc.settings.autoshow = False

    # Compute filtered topics via TopicStore (single dataset).
    store = TopicStore()
    topk_mass_th: Optional[float] = None if topk_mass_threshold is None or topk_mass_threshold <= 0 else float(topk_mass_threshold)

    stats = store.add_topics(
        dataset_name=dataset,
        results_dir=results_dir,
        filter_background=filter_background,
        sparsity_threshold=sparsity_threshold,
        topk_mass_threshold=topk_mass_th,
        topk=topk,
        expand_genes=True,
        return_coupling=False,
        coherence_top_n=coherence_top_n,
        coherence_threshold=coherence_threshold,
    )
    filtered = list(stats.get("filtered", []))

    all_idx = np.arange(n_topics, dtype=int)
    drop_mask = np.isin(all_idx, np.array(filtered, dtype=int)) if filtered else np.zeros_like(all_idx, dtype=bool)
    keep_indices = all_idx[~drop_mask]

    if keep_indices.size == 0:
        raise ValueError(
            "Topic filtering removed all topics; cannot build post-filter UMAP."
        )

    # Load cell-topic matrix for the dataset.
    theta = _load_cell_topic_matrix(results_dir, dataset, n_topics=n_topics)

    # Optional global downsampling for speed.
    if max_cells is not None and theta.shape[0] > max_cells:
        idx = np.random.choice(theta.shape[0], size=max_cells, replace=False)
        theta = theta[idx]
        effective_n_cells = theta.shape[0]
    else:
        effective_n_cells = theta.shape[0]

    # Load batch and (optionally) cell-type labels aligned to theta rows.
    batch_labels, cell_types = _load_obs_labels(
        adata_path=adata_path,
        batch_key=batch_key,
        celltype_key=celltype_key,
        expected_n_cells=effective_n_cells,
    )

    # Pre-filter UMAP: all topics.
    adata_pre = sc.AnnData(X=theta)
    adata_pre.obs["batch"] = batch_labels
    if cell_types is not None:
        adata_pre.obs["cell_type"] = cell_types

    sc.pp.neighbors(adata_pre, use_rep="X", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata_pre, min_dist=0.3, random_state=0)
    sc.pl.umap(
        adata_pre,
        color="batch",
        show=False,
        save=f"_{tag}_pre_topic_filter_batch.png",
    )
    if cell_types is not None:
        sc.pl.umap(
            adata_pre,
            color="cell_type",
            show=False,
            save=f"_{tag}_pre_topic_filter_celltype.png",
        )

    # Post-filter UMAP: keep only the non-filtered topics.
    theta_post = theta[:, keep_indices]
    adata_post = sc.AnnData(X=theta_post)
    adata_post.obs["batch"] = batch_labels
    if cell_types is not None:
        adata_post.obs["cell_type"] = cell_types

    sc.pp.neighbors(adata_post, use_rep="X", n_neighbors=15, metric="euclidean")
    sc.tl.umap(adata_post, min_dist=0.3, random_state=0)
    sc.pl.umap(
        adata_post,
        color="batch",
        show=False,
        save=f"_{tag}_post_topic_filter_batch.png",
    )
    if cell_types is not None:
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
            "Visualize batch effect before/after topic filtering "
            "for a single dataset using its cell-topic matrix."
        )
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset label used in training (e.g., PBMC12k_scVI_align).",
    )
    p.add_argument(
        "--results_dir",
        required=True,
        help="Results root directory for the dataset (e.g., results or results/tuning/<run>).",
    )
    p.add_argument(
        "--adata",
        required=True,
        help="Path to original .h5ad file (e.g., data/PBMC12k.h5ad).",
    )
    p.add_argument(
        "--batch_key",
        default="batch",
        help="Column in AnnData.obs that encodes batch labels (default: batch).",
    )
    p.add_argument(
        "--celltype_key",
        default="cell_type",
        help="Optional column in AnnData.obs used as cell type labels (default: cell_type).",
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
        "--max_cells",
        type=int,
        default=None,
        help="Optional maximum number of cells to sample for plotting.",
    )
    # Topic filtering knobs (mirroring TopicStore.add_topics, including coherence).
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
    p.add_argument(
        "--coherence_top_n",
        type=int,
        default=20,
        help="Number of top genes per topic used to compute coherence (default: 20).",
    )
    p.add_argument(
        "--coherence_threshold",
        type=float,
        default=0.20,
        help="Topic coherence threshold; topics with coherence <= this value are considered low-coherence.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    filter_background = not args.no_filter_background
    celltype_key = args.celltype_key if args.celltype_key else None

    plot_batch_effect_umap_single(
        dataset=args.dataset,
        results_dir=args.results_dir,
        adata_path=args.adata,
        batch_key=args.batch_key,
        celltype_key=celltype_key,
        n_topics=args.n_topics,
        out_dir=args.out_dir,
        tag=args.tag,
        max_cells=args.max_cells,
        filter_background=filter_background,
        sparsity_threshold=args.sparsity_threshold,
        topk_mass_threshold=args.topk_mass_threshold,
        topk=args.topk,
        coherence_top_n=args.coherence_top_n,
        coherence_threshold=args.coherence_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
