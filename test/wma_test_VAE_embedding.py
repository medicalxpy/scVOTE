#!/usr/bin/env python3
"""
Visualization script for VAE cell embeddings to check cluster separation.

This script loads VAE embeddings and corresponding AnnData files,
performs UMAP on the embeddings, and visualizes colored by cell type. It
also supports visualizing cell-topic embeddings from contrastive runs.
"""

import os
import pickle
import glob
import re
from itertools import combinations
from pathlib import Path
from typing import Optional, Set, List, Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


TOKEN_LIST = [
    '_alignw0_tdw0',
    '_contrastive_alignw1e-3_tdw0',
    '_contrastive_alignw1e-3_tdw1e-3',
    '_structure_alignw1e-3_tdw0',
    '_structure_alignw1e-3_tdw1e-3'
]

GENESET_DIR = "/data1021/xiepengyu/scVOTE/data/gene_sets"

DEFAULT_LABEL_CANDIDATES: List[str] = [
    "cell_type",
    "celltype",
    "celltype.l1",
    "celltype.l2",
    "label",
    "labels",
    "CellType",  # PBMC sca dataset
]


def _pick_label_key(adata: "ad.AnnData") -> Optional[str]:
    for k in DEFAULT_LABEL_CANDIDATES:
        if k in adata.obs.columns:
            return k
    return None


def extract_top_genes_per_topic(
    topic_gene: np.ndarray,
    gene_names: List[str],
    top_k: int = 100,
):
    """Return a DataFrame of top genes per topic.

    Handles either (topics x genes) or (genes x topics) by inferring orientation
    from the length of gene_names.
    """
    if topic_gene.shape[1] == len(gene_names):
        mat = topic_gene
    elif topic_gene.shape[0] == len(gene_names):
        mat = topic_gene.T
    else:
        print(
            f"Shape mismatch: topic_gene {topic_gene.shape} not compatible with {len(gene_names)} gene names"
        )
        return None

    records = []
    for t_idx in range(mat.shape[0]):
        row = mat[t_idx]
        top_idx = np.argsort(-row)[:top_k]
        for rank, g_idx in enumerate(top_idx, start=1):
            records.append(
                {
                    "topic": f"topic_{t_idx}",
                    "gene": gene_names[g_idx],
                    "score": float(row[g_idx]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(records, columns=["topic", "gene", "score", "rank"])


def _build_gene_topic_df(
    topic_gene: np.ndarray,
    gene_names: List[str],
) -> Optional[pd.DataFrame]:
    """Return gene x topic DataFrame with gene symbols as index."""
    if topic_gene.shape[1] == len(gene_names):
        mat = topic_gene
    elif topic_gene.shape[0] == len(gene_names):
        mat = topic_gene.T
    else:
        print(
            f"Shape mismatch: topic_gene {topic_gene.shape} not compatible with {len(gene_names)} gene names"
        )
        return None
    n_topics = mat.shape[0]
    df = pd.DataFrame(mat.T, columns=[f"topic_{i}" for i in range(n_topics)])
    df.index = pd.Index([str(g).upper() for g in gene_names], name="gene")
    return df


def _load_go_bp_genesets() -> Dict[str, set]:
    path = os.path.join(GENESET_DIR, f"C2_C5_GO_genesets.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GO_BP geneset file not found: {path}")
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError(f"{path} must contain a 'gene' column.")

    name_col: Optional[str] = None
    for cand in ("go_term", "set_name", "category", "term"):
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        raise ValueError(
            f"{path} must contain one of the geneset name columns: 'go_term', 'set_name', 'category', 'term'."
        )

    df[name_col] = df[name_col].astype(str)
    df["gene"] = df["gene"].astype(str).str.upper()

    genesets: Dict[str, set] = {}
    for name, sub in df.groupby(name_col):
        genesets[name] = set(sub["gene"].tolist())
    return genesets


def _gsea_es_from_mask(scores: np.ndarray, hits_mask: np.ndarray, p: float = 1.0) -> float:
    n = scores.shape[0]
    nh = int(hits_mask.sum())
    nm = n - nh
    if nh == 0 or nm == 0:
        return 0.0

    order = np.argsort(-scores)
    scores_sorted = scores[order]
    hits_sorted = hits_mask[order]

    hit_scores = (np.abs(scores_sorted) ** p) * hits_sorted.astype(float)
    sum_hit = hit_scores[hits_sorted].sum()
    if sum_hit == 0:
        return 0.0

    phit = np.cumsum(hit_scores / sum_hit)
    pmiss = np.cumsum((~hits_sorted).astype(float) / float(nm))
    running = phit - pmiss
    es_pos = running.max()
    es_neg = running.min()
    return float(es_pos if abs(es_pos) >= abs(es_neg) else es_neg)


def _gsea_prerank_per_topic(
    gene_topic_df: pd.DataFrame,
    genesets: Dict[str, set],
    *,
    top_n: int = 100,
    min_size: int = 5,
    max_size: int = 500,
    permutation_num: int = 1000,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Run GSEA prerank per topic and return full results sorted by p-value."""
    if gene_topic_df.empty:
        raise ValueError("Gene-topic matrix is empty; cannot compute GSEA.")
    if not genesets:
        raise ValueError("Geneset dictionary is empty; cannot compute GSEA.")

    try:
        import gseapy as gp
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "gseapy is required for p-value-based GSEA. Install with 'pip install gseapy'."
        ) from exc

    df = gene_topic_df.copy()

    df.index = df.index.map(lambda x: str(x).upper())
    df = df.groupby(df.index).mean()

    results: Dict[str, pd.DataFrame] = {}
    for topic in df.columns:
        scores = df[topic]
        rnk = scores.sort_values(ascending=False).head(top_n).reset_index()
        rnk.columns = ["gene", "score"]

        prerank_res = gp.prerank(
            rnk=rnk,
            gene_sets=genesets,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutation_num,
            outdir=None,
            seed=seed,
            verbose=False,
        )

        res = prerank_res.res2d
        if res is None or res.empty:
            continue
        res = res.reset_index().rename(columns={"Term": "pathway"})
        res = res.sort_values(['FDR q-val'], ascending=[True])
        results[topic] = res

    return results


def _enrichr_per_topic(
    gene_topic_df: pd.DataFrame,
    *,
    top_n: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Run Enrichr per topic using gseapy and return full results."""
    if gene_topic_df.empty:
        raise ValueError("Gene-topic matrix is empty; cannot compute Enrichr.")
    try:
        import gseapy as gp
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "gseapy is required for Enrichr. Install with 'pip install gseapy'."
        ) from exc

    df = gene_topic_df.copy()
    df.index = df.index.map(lambda x: str(x).upper())
    df = df.groupby(df.index).mean()

    results: Dict[str, pd.DataFrame] = {}
    for topic in df.columns:
        scores = df[topic]
        gene_list = scores.sort_values(ascending=False).head(top_n).index.tolist()

        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets='CellMarker_2024',
            outdir=None,
            no_plot=True,
        )

        res = enr.results
        if res is None or res.empty:
            continue
        results[topic] = res

    return results


def assign_topic_cell_types(
    cell_topic_path: str,
    adata_path: str,
    barcode_path: Optional[str] = None,
) -> Optional[Tuple[dict, str]]:
    """Assign each topic to the dominant label (from candidate list).

    Returns (topic_to_label, label_key).
    """
    if not os.path.exists(cell_topic_path):
        print(f"Cell-topic file missing: {cell_topic_path}")
        return None
    if not os.path.exists(adata_path):
        print(f"AnnData file missing: {adata_path}")
        return None

    with open(cell_topic_path, "rb") as f:
        cell_topic = pickle.load(f)

    adata = sc.read_h5ad(adata_path)
    if barcode_path and os.path.exists(barcode_path):
        with open(barcode_path, "r") as f:
            barcodes = [line.strip() for line in f]
        adata = adata[barcodes].copy()
    if cell_topic.shape[0] != adata.n_obs:
        print(
            f"Mismatch for cell-topic vs adata cells: {cell_topic.shape[0]} vs {adata.n_obs}"
        )
        return None
    label_key = _pick_label_key(adata)
    if label_key is None:
        print("No label column found in adata.obs; cannot assign topics")
        return None

    # Compute mean weight per label per topic
    ct = adata.obs[label_key].astype(str)
    df = pd.DataFrame(cell_topic, index=ct.index)
    df[label_key] = ct.values
    mean_by_ct = df.groupby(label_key).mean()

    topic_to_ct = {}
    for t_idx in range(cell_topic.shape[1]):
        topic_col = mean_by_ct.iloc[:, t_idx]
        top_ct = topic_col.idxmax()
        topic_to_ct[f"topic_{t_idx}"] = top_ct

    return topic_to_ct, label_key


def visualize_cell_topic(
    embedding_file: str,
    adata_file: str,
    output_dir: str,
    barcode_file: Optional[str] = None,
):
    """Visualize cell-topic embeddings for one file.

    Uses the provided `barcode_file` if available (recommended: VAE barcodes
    from results/cell_embedding), otherwise falls back to a local sibling
    barcode file next to the cell-topic pickle.
    """
    dataset_name = Path(embedding_file).stem
    dataset_base = re.sub(
        r"_(vae_align|vae)_cell_topic_matrix_\d+$", "", dataset_name
    )

    with open(embedding_file, 'rb') as f:
        cell_topic = pickle.load(f)
    print(f"Loaded cell-topic matrix: {cell_topic.shape}")

    # Prefer explicitly provided barcode path (from VAE embeddings directory)
    inferred_barcode = embedding_file.replace('.pkl', '_barcodes.txt')
    barcode_path = barcode_file if barcode_file else inferred_barcode

    if os.path.exists(barcode_path):
        with open(barcode_path, 'r') as f:
            cell_barcodes = [line.strip() for line in f]
        print(f"Loaded {len(cell_barcodes)} cell barcodes for cell-topic matrix")
    else:
        cell_barcodes = None
        print("No barcode file found for cell-topic matrix; using AnnData order")

    adata = sc.read_h5ad(adata_file)
    print(f"Loaded adata for cell-topic: {adata.shape}")

    if cell_barcodes is not None:
        adata_pp = adata[cell_barcodes].copy()
    else:
        adata_pp = adata.copy()

    if adata_pp.shape[0] != cell_topic.shape[0]:
        print(
            f"Warning: adata_pp cells {adata_pp.shape[0]} != cell-topic cells {cell_topic.shape[0]}"
        )
        return None

    if cell_barcodes is not None and list(adata_pp.obs_names) != cell_barcodes:
        print("Warning: cell barcodes do not match AnnData obs_names for cell-topic")
        return None

    adata_emb = ad.AnnData(
        X=cell_topic,
        obs=adata_pp.obs.copy(),
        var=pd.DataFrame(index=[f'topic_{i}' for i in range(cell_topic.shape[1])])
    )

    label_key = _pick_label_key(adata_emb)
    if label_key is None:
        print("No label column found in adata.obs; cannot compute ARI")
        return None
    n_clusters = len(adata_emb.obs[label_key].unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(cell_topic)
    ari = adjusted_rand_score(adata_emb.obs[label_key], clusters)
    print(f"Cell-topic ARI for {dataset_base}: {ari:.4f}")

    adata_emb.obs['kmeans_cluster'] = clusters.astype(str)

    sc.pp.neighbors(adata_emb, n_neighbors=15, use_rep='X')
    sc.tl.umap(adata_emb)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc.pl.umap(adata_emb, color=label_key, ax=ax, show=False, legend_loc='right margin')
    plt.title(f'UMAP of Cell-Topic Embeddings: {dataset_base} ({label_key})')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{dataset_base}_cell_topic_umap.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved cell-topic plot to: {output_path}")
    
    # Also generate a clustered heatmap colored by label
    if label_key in adata_emb.obs.columns:
        plot_cell_topic_heatmap(
            cell_topic=cell_topic,
            cell_types=adata_emb.obs[label_key],
            dataset_base=dataset_base,
            output_dir=output_dir,
            label_name=label_key,
        )
    else:
        print(f"Warning: '{label_key}' not found in obs; skipping heatmap coloring.")

    return dataset_base, ari


def plot_cell_topic_heatmap(
    cell_topic: np.ndarray,
    cell_types: pd.Series,
    dataset_base: str,
    output_dir: str,
    label_name: str,
):
    """Generate a clustered heatmap for cell x topic embeddings.

    - Rows: cells grouped by label (no clustering)
    - Columns: topics (clustered)
    - Row colors: label categories
    """
    # Order cells by label to keep same types together
    order_idx = cell_types.astype(str).sort_values().index
    df = pd.DataFrame(
        cell_topic,
        index=cell_types.index,
        columns=[f"topic_{i}" for i in range(cell_topic.shape[1])],
    ).loc[order_idx]
    ordered_cell_types = cell_types.loc[order_idx].astype(str)

    # Create a consistent palette for cell types
    categories = ordered_cell_types.unique()
    palette = sns.color_palette("tab20", n_colors=len(categories))
    color_map = {cat: palette[i] for i, cat in enumerate(categories)}
    row_colors = ordered_cell_types.map(color_map)

    # Cluster columns (topics), keep rows ordered by label; z-score rows
    cg = sns.clustermap(
        df,
        row_cluster=False,
        col_cluster=True,
        row_colors=row_colors,
        cmap="viridis",
        # z_score=0,
        metric="euclidean",
        method="average",
        figsize=(10, 10),
    )

    # Add legend for cell types
    handles = [Patch(color=color_map[cat], label=cat) for cat in categories]
    cg.ax_heatmap.legend(
        handles=handles,
        title=label_name,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0,
    )

    # Add space on the right to avoid overlap with the legend
    cg.fig.subplots_adjust(right=0.82)

    # Hide row labels
    cg.ax_heatmap.set_yticklabels([])
    cg.ax_heatmap.tick_params(axis='y', which='both', length=0)
    cg.ax_heatmap.set_ylabel('')

    plt.suptitle(f"Cell x Topic Heatmap: {dataset_base}", y=1.02)
    heatmap_path = os.path.join(output_dir, f"{dataset_base}_cell_topic_heatmap.png")
    cg.fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(cg.fig)
    print(f"Saved cell-topic heatmap to: {heatmap_path}")

def visualize_embeddings(embedding_file: str, adata_file: str, output_dir: str = "test"):
    """Visualize embeddings for one file."""
    dataset_name = Path(embedding_file).stem.replace('_vae', '')
    
    # Load embeddings
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        print("Embeddings contain NaN/Inf values; skipping evaluation")
        return None

    # Load cell barcodes
    barcode_file = embedding_file.replace('_vae.pkl', '_barcodes.txt')
    if os.path.exists(barcode_file):
        with open(barcode_file, 'r') as f:
            cell_barcodes = [line.strip() for line in f]
        print(f"Loaded {len(cell_barcodes)} cell barcodes")
    else:
        cell_barcodes = None
        print("No barcode file found")

    # Load original adata
    adata = sc.read_h5ad(adata_file)
    print(f"Loaded adata: {adata.shape}")
    adata_pp = adata[cell_barcodes].copy() # subset to barcodes of embedding

    # Preprocess to get the subset
    if adata_pp.shape[0] != embeddings.shape[0]:
        print(f"Warning: adata_pp cells {adata_pp.shape[0]} != embeddings cells {embeddings.shape[0]}")
        return None
    
    if cell_barcodes is not None:
        if list(adata_pp.obs_names) != cell_barcodes:
            print("Warning: cell barcodes do not match preprocessed obs_names")
            return None
        else:
            print("Cell barcodes match.")

    # Create AnnData for embeddings
    adata_emb = ad.AnnData(
        X=embeddings,
        obs=adata_pp.obs.copy(),
        var=pd.DataFrame(index=[f'latent_{i}' for i in range(embeddings.shape[1])])
    )

    # Perform KMeans clustering and calculate ARI
    


    label_key = _pick_label_key(adata_emb)
    if label_key is None:
        print("No label column found in adata.obs; cannot compute ARI")
        return None
    n_clusters = len(adata_emb.obs[label_key].unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    ari = adjusted_rand_score(adata_emb.obs[label_key], clusters)
    print(f"ARI for {dataset_name}: {ari:.4f}")

    # Add cluster labels to obs for potential visualization
    adata_emb.obs['kmeans_cluster'] = clusters.astype(str)

    # Compute UMAP
    sc.pp.neighbors(adata_emb, n_neighbors=15, use_rep='X')
    sc.tl.umap(adata_emb)

    # Plot UMAP colored by label
    fig, ax = plt.subplots(figsize=(10, 6))
    sc.pl.umap(adata_emb, color=label_key, ax=ax, show=False, legend_loc='right margin')
    plt.title(f'UMAP of VAE Embeddings: {dataset_name} ({label_key})')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{dataset_name}_umap.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {output_path}")

    return dataset_name, ari


def load_top_genes(path: str) -> dict:
    """Load top genes per topic CSV into a dict of topic -> set of genes."""
    df = pd.read_csv(path)
    topics = {}
    for topic, sub in df.groupby("topic"):
        topics[topic] = set(sub["gene"].astype(str).tolist())
    return topics


def compute_gene_set_overlaps(top_gene_files: List[str], output_dir: str):
    """Compute Jaccard overlap between topics across individuals."""
    if not top_gene_files:
        print("No top-gene files found; skipping overlap computation")
        return

    datasets = {}
    for path in top_gene_files:
        dataset = Path(path).stem.replace('_top100_genes_per_topic', '')
        datasets[dataset] = load_top_genes(path)

    records = []
    for (ds_a, topics_a), (ds_b, topics_b) in combinations(datasets.items(), 2):
        for t_a, genes_a in topics_a.items():
            for t_b, genes_b in topics_b.items():
                inter = genes_a & genes_b
                union = genes_a | genes_b
                jaccard = len(inter) / len(union) if union else 0.0
                records.append(
                    {
                        "dataset_a": ds_a,
                        "topic_a": t_a,
                        "dataset_b": ds_b,
                        "topic_b": t_b,
                        "overlap": len(inter),
                        "jaccard": jaccard,
                    }
                )

    if not records:
        print("No overlap records computed")
        return

    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "topic_gene_jaccard_overlaps.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved gene-set overlaps to: {out_path}")

    # Show top 10 overlaps for quick inspection
    top10 = df.sort_values("jaccard", ascending=False).head(10)
    print("Top 10 topic-topic gene-set overlaps:")
    print(top10.to_string(index=False))


def topic_topic_overlap_from_topgenes(
    top_genes_df: pd.DataFrame,
    metric: str = "overlap",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute topic x topic overlap matrix from a top-genes-per-topic DataFrame.

    - metric="overlap": intersection size of top gene sets
    - metric="jaccard": Jaccard index of top gene sets

    Returns (overlap_df, topic_cell_types) where topic_cell_types is a Series
    indexed by topic with values as label (if available, else 'NA').
    """
    if "topic" not in top_genes_df.columns or "gene" not in top_genes_df.columns:
        raise ValueError("top_genes_df must contain 'topic' and 'gene' columns")

    topics = sorted(top_genes_df["topic"].unique(), key=lambda x: (len(x), x))
    gene_sets = {
        t: set(top_genes_df.loc[top_genes_df["topic"] == t, "gene"].astype(str))
        for t in topics
    }

    if label_col in top_genes_df.columns:
        topic_ct = top_genes_df.groupby("topic")[label_col].first().reindex(topics)
    else:
        topic_ct = pd.Series(["NA"] * len(topics), index=topics, name=label_col)

    n = len(topics)
    M = np.zeros((n, n), dtype=float)
    for i, ti in enumerate(topics):
        gi = gene_sets[ti]
        for j, tj in enumerate(topics):
            gj = gene_sets[tj]
            inter = len(gi & gj)
            if metric == "overlap":
                M[i, j] = inter
            elif metric == "jaccard":
                union = len(gi | gj)
                M[i, j] = inter / union if union else 0.0
            else:
                raise ValueError("metric must be 'overlap' or 'jaccard'")

    overlap_df = pd.DataFrame(M, index=topics, columns=topics)
    return overlap_df, topic_ct


def plot_topic_overlap_heatmap(
    overlap_df: pd.DataFrame,
    topic_cell_types: pd.Series,
    dataset: str,
    output_dir: str,
    metric: str = "overlap",
    label_name: str = "label",
):
    """Plot clustered topic x topic overlap heatmap with topic cell-type colors."""
    topics = overlap_df.index.tolist()
    ct = topic_cell_types.reindex(topics).astype(str)

    # Order topics so same cell types are grouped together (no clustering)
    ordered_topics = ct.sort_values().index.tolist()
    overlap_df = overlap_df.loc[ordered_topics, ordered_topics]
    ct = ct.loc[ordered_topics]

    categories = ct.unique()
    palette = sns.color_palette("tab20", n_colors=len(categories))
    color_map = {cat: palette[i] for i, cat in enumerate(categories)}
    row_colors = ct.map(color_map)
    col_colors = row_colors  # same ordering as rows (square matrix)

    # Do not cluster; just display topics ordered by cell type groups
    cg = sns.clustermap(
        overlap_df,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap="magma" if metric == "overlap" else "viridis",
        figsize=(10, 10),
    )

    # Legend for topic labels
    handles = [Patch(color=color_map[cat], label=cat) for cat in categories]
    cg.ax_heatmap.legend(
        handles=handles,
        title=f"topic {label_name}",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0,
    )
    cg.fig.subplots_adjust(right=0.82)

    # Hide tick labels to avoid overlap; keep axes labels
    cg.ax_heatmap.set_xticklabels([])
    cg.ax_heatmap.set_yticklabels([])
    cg.ax_heatmap.tick_params(axis="both", which="both", length=0)
    cg.ax_heatmap.set_xlabel("topics")
    cg.ax_heatmap.set_ylabel("topics")

    title_metric = "Overlap count" if metric == "overlap" else "Jaccard"
    plt.suptitle(f"Topic x Topic Top-Genes {title_metric}: {dataset}", y=1.02)

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(
        output_dir, f"{dataset}_topic_topic_top100_{metric}_heatmap.png"
    )
    cg.fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(cg.fig)
    print(f"Saved topic-topic {metric} heatmap to: {out_png}")


def _parse_dataset_from_run(run_name: str) -> str:
    for token in TOKEN_LIST:
        if token in run_name:
            return run_name.split(token)[0]
    return re.sub(r"_K\d+$", "", run_name)


def _find_matrix_file(base_dir: str, dataset: str, kind: str, n_topics: int = 50) -> Optional[str]:
    candidates = [
        os.path.join(base_dir, f"{dataset}_vae_align_{kind}_matrix_{n_topics}.pkl"),
        os.path.join(base_dir, f"{dataset}_vae_{kind}_matrix_{n_topics}.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    fallback = glob.glob(os.path.join(base_dir, f"*{kind}_matrix_{n_topics}.pkl"))
    return fallback[0] if fallback else None


def _find_gene_names_file(base_dir: str, dataset: str, n_topics: int = 50) -> Optional[str]:
    candidates = [
        os.path.join(base_dir, f"{dataset}_vae_align_gene_names_{n_topics}.pkl"),
        os.path.join(base_dir, f"{dataset}_vae_gene_names_{n_topics}.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    fallback = glob.glob(os.path.join(base_dir, f"*gene_names_{n_topics}.pkl"))
    return fallback[0] if fallback else None


def _is_target_dataset(dataset: str, targets: Optional[Set[str]] = None) -> bool:
    if targets is not None and len(targets) > 0:
        return dataset in targets
    if dataset.startswith("human_PBMC_batch1_ind"):
        return True
    if dataset.startswith("human_PBMC_sca_method"):
        return True
    return dataset in {
        "kidney",
        "lung",
        "Spleen",
        "wang",
        "PBMC4K",
        "PBMC8K",
        "PBMC12K",
    }


def load_targets_from_ablation_script(
    script_path: str,
) -> Tuple[Optional[str], Set[str]]:
    """Load ADATA_DIR and dataset targets from the ablation shell script."""
    if not os.path.exists(script_path):
        print(f"Ablation script not found: {script_path}")
        return None, set()

    adata_dir: Optional[str] = None
    dataset_patterns: List[str] = []
    in_list = False

    with open(script_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ADATA_DIR="):
                match = re.search(r'ADATA_DIR="([^"]+)"', line)
                if match:
                    adata_dir = match.group(1)
                continue
            if line.startswith("DATASET_FILES=("):
                in_list = True
                continue
            if in_list:
                if line == ")":
                    in_list = False
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                line = line.strip().replace('"', "").replace("'", "")
                if line:
                    dataset_patterns.append(line)

    if adata_dir is None:
        print(f"ADATA_DIR not found in {script_path}")

    targets: Set[str] = set()
    for pattern in dataset_patterns:
        if adata_dir:
            pattern = pattern.replace("${ADATA_DIR}", adata_dir).replace(
                "$ADATA_DIR", adata_dir
            )
        for path in glob.glob(pattern):
            if os.path.isfile(path) and path.endswith(".h5ad"):
                targets.add(os.path.basename(path).replace(".h5ad", ""))

    if not targets:
        print(f"No dataset files matched in {script_path}")

    return adata_dir, targets


def main():
    embedding_dir = "/data1021/xiepengyu/scVOTE/results/cell_embedding"
    adata_dir = "/data1021/xiepengyu/scVOTE/data"
    tuning_root = "/data1021/xiepengyu/scVOTE/results/tuning"
    ablation_script = "/data1021/xiepengyu/scVOTE/test/wma_cli_vae_ablation.sh"

    adata_dir_from_script, target_datasets = load_targets_from_ablation_script(
        ablation_script
    )
    if adata_dir_from_script:
        adata_dir = adata_dir_from_script
    if target_datasets:
        print(
            f"Loaded {len(target_datasets)} target datasets from {ablation_script}"
        )

    run_dirs = [
        d
        for d in glob.glob(os.path.join(tuning_root, "*"))
        if os.path.isdir(d)
    ]
    if not run_dirs:
        print(f"No tuning runs found under: {tuning_root}")
        return

    embedding_cache: Dict[str, Tuple[str, float]] = {}

    for run_dir in sorted(run_dirs):
        run_name = os.path.basename(run_dir)
        dataset = _parse_dataset_from_run(run_name)
        if not _is_target_dataset(dataset, target_datasets):
            print(f"Skipping run (not in target list): {run_name}")
            continue
        adata_file = os.path.join(adata_dir, f"{dataset}.h5ad")
        if not os.path.exists(adata_file):
            print(f"AnnData file not found: {adata_file}")
            continue

        output_dir = os.path.join(run_dir, "validation_figures")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Processing run: {run_name} (dataset={dataset}) ===")

        emb_file = os.path.join(embedding_dir, f"{dataset}_vae.pkl")
        if os.path.exists(emb_file):
            if dataset not in embedding_cache:
                result = visualize_embeddings(emb_file, adata_file, output_dir)
                if result:
                    embedding_cache[dataset] = result
            else:
                cached = embedding_cache[dataset]
                csv_path = os.path.join(output_dir, "kmeans_ari_results.csv")
                df_cached = pd.DataFrame([cached], columns=["dataset", "ARI"])
                df_cached.to_csv(csv_path, index=False)
                print(f"Reused cached VAE ARI for {dataset}; wrote {csv_path}")
        else:
            print(f"VAE embedding file not found: {emb_file}")

        # Process cell-topic embeddings for this run
        ct_dir = os.path.join(run_dir, "cell_topic")
        ct_files: List[str] = []
        if os.path.isdir(ct_dir):
            ct_file = _find_matrix_file(ct_dir, dataset, "cell_topic", n_topics=50)
            if ct_file:
                ct_files = [ct_file]
            else:
                ct_files = glob.glob(os.path.join(ct_dir, "*cell_topic_matrix*.pkl"))

        cell_topic_results: List[Tuple[str, float]] = []
        if ct_files:
            vae_barcode_path = os.path.join(embedding_dir, f"{dataset}_barcodes.txt")
            for ct_file in ct_files:
                print(f"Processing cell-topic: {ct_file}")
                ct_result = visualize_cell_topic(
                    ct_file,
                    adata_file,
                    ct_dir,
                    barcode_file=vae_barcode_path,
                )
                if ct_result:
                    cell_topic_results.append(ct_result)
            if cell_topic_results:
                df_ct_ind = pd.DataFrame(cell_topic_results, columns=["dataset", "ARI"])
                csv_ct_ind = os.path.join(ct_dir, "cell_topic_kmeans_ari_results.csv")
                df_ct_ind.to_csv(csv_ct_ind, index=False)
                print(f"Saved cell-topic ARI to: {csv_ct_ind}")
        else:
            print(f"No cell-topic matrix found under: {ct_dir}")

        # Derive top genes per topic for this run
        tg_dir = os.path.join(run_dir, "topic_gene")
        ge_dir = os.path.join(run_dir, "gene_embedding")
        tg_file = _find_matrix_file(tg_dir, dataset, "topic_gene", n_topics=50)
        gene_names_file = _find_gene_names_file(ge_dir, dataset, n_topics=50)

        if tg_file and gene_names_file and os.path.exists(tg_file) and os.path.exists(gene_names_file):
            with open(tg_file, "rb") as f:
                topic_gene = pickle.load(f)
            with open(gene_names_file, "rb") as f:
                gene_names = pickle.load(f)

            print(
                f"Loaded topic-gene matrix: {topic_gene.shape} and {len(gene_names)} gene names for {dataset}"
            )
            top_genes_df = extract_top_genes_per_topic(topic_gene, gene_names, top_k=100)

            if top_genes_df is not None:
                os.makedirs(tg_dir, exist_ok=True)
                top_genes_path = os.path.join(tg_dir, f"{dataset}_top100_genes_per_topic.csv")
                top_genes_df.to_csv(top_genes_path, index=False)
                print(f"Saved top genes per topic to: {top_genes_path}")

                topic_celltype = None
                topic_label_key = None
                if ct_files:
                    vae_barcode_path = os.path.join(embedding_dir, f"{dataset}_barcodes.txt")
                    topic_celltype_result = assign_topic_cell_types(
                        cell_topic_path=ct_files[0],
                        adata_path=adata_file,
                        barcode_path=vae_barcode_path if os.path.exists(vae_barcode_path) else None,
                    )
                    if topic_celltype_result is not None:
                        topic_celltype, topic_label_key = topic_celltype_result

                if topic_celltype:
                    label_col = topic_label_key or "label"
                    top_genes_df[label_col] = top_genes_df["topic"].map(topic_celltype)
                    ct_top_genes_path = os.path.join(
                        tg_dir, f"{dataset}_top100_genes_per_topic_with_celltype.csv"
                    )
                    top_genes_df.to_csv(ct_top_genes_path, index=False)
                    print(f"Saved cell-type-annotated top genes to: {ct_top_genes_path}")

                    overlap_df, topic_ct_series = topic_topic_overlap_from_topgenes(
                        top_genes_df, metric="overlap", label_col=label_col
                    )
                    overlap_csv = os.path.join(
                        tg_dir, f"{dataset}_topic_topic_top100_overlap_matrix.csv"
                    )
                    overlap_df.to_csv(overlap_csv)
                    print(f"Saved topic-topic overlap matrix to: {overlap_csv}")

                    plot_topic_overlap_heatmap(
                        overlap_df=overlap_df,
                        topic_cell_types=topic_ct_series,
                        dataset=dataset,
                        output_dir=tg_dir,
                        metric="overlap",
                        label_name=label_col,
                    )

                # GSEA pathway enrichment per topic (GO_BP genesets)
                try:
                    gene_topic_df = _build_gene_topic_df(topic_gene, gene_names)
                    if gene_topic_df is None:
                        raise ValueError("Failed to build gene-topic DataFrame")
                    go_genesets = _load_go_bp_genesets()
                    gsea_dir = os.path.join(tg_dir, "topic_gsea")
                    os.makedirs(gsea_dir, exist_ok=True)

                    gsea_full = _gsea_prerank_per_topic(
                        gene_topic_df,
                        go_genesets,
                        top_n=100,
                        min_size=5,
                        max_size=500,
                        permutation_num=1000,
                        seed=0,
                    )
                    for topic, res in gsea_full.items():
                        out_path = os.path.join(
                            gsea_dir, f"{dataset}_{topic}_gsea.csv"
                        )
                        res.to_csv(out_path, index=False)
                    print(f"Saved full topic GSEA results under: {gsea_dir}")

                    enrichr_dir = os.path.join(tg_dir, "topic_gsea", "enrichr")
                    os.makedirs(enrichr_dir, exist_ok=True)
                    enrichr_full = _enrichr_per_topic(
                        gene_topic_df,
                        top_n=100,
                    )
                    for topic, res in enrichr_full.items():
                        out_path = os.path.join(
                            enrichr_dir, f"{dataset}_{topic}_enrichr.csv"
                        )
                        res.to_csv(out_path, index=False)
                    print(f"Saved Enrichr results under: {enrichr_dir}")
                except Exception as exc:
                    print(f"GSEA (GO_BP) unavailable for {dataset}: {exc}")
        else:
            print(f"Topic-gene or gene-names file missing for {run_name}; skipping top-gene extraction")

        # Compute gene-set overlaps across individuals for this run
        if os.path.isdir(tg_dir):
            top_gene_files = glob.glob(os.path.join(tg_dir, "*_top100_genes_per_topic.csv"))
            overlap_output_dir = os.path.join(run_dir, "topic_gene_overlap")
            compute_gene_set_overlaps(top_gene_files, overlap_output_dir)


if __name__ == "__main__":
    main()