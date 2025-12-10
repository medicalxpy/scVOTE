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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import fisher_exact
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


ROOT_DIR = Path(__file__).resolve().parent
GENESET_DIR = ROOT_DIR / "data" / "gene_sets"


@dataclass
class EvalConfig:
    adata_path: str
    results_dir: Optional[str] = None
    dataset: Optional[str] = None
    n_topics: Optional[int] = None
    cell_topic_file: Optional[str] = None
    label_key: Optional[str] = None
    tag: Optional[str] = None
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


def _resolve_topic_gene_path(cfg: "EvalConfig") -> Path:
    if not (cfg.results_dir and cfg.dataset and cfg.n_topics is not None):
        raise ValueError(
            "Gene-topic metrics require --results_dir, --dataset and --n_topics to be set"
        )
    return (
        Path(cfg.results_dir)
        / "topic_gene"
        / f"{cfg.dataset}_topic_gene_matrix_{cfg.n_topics}.pkl"
    )


def _load_topic_gene_matrix(cfg: "EvalConfig") -> pd.DataFrame:
    """Load topic_gene_matrix as gene × topic DataFrame with gene symbols as index."""
    tg_path = _resolve_topic_gene_path(cfg)
    if not tg_path.exists():
        raise FileNotFoundError(f"Gene-topic file not found: {tg_path}")
    with open(tg_path, "rb") as f:
        mat = pickle.load(f)
    mat = np.asarray(mat)

    # Infer topics dimension (topics × genes)
    n_topics = mat.shape[0]
    df = pd.DataFrame(mat.T, columns=[f"topic_{i}" for i in range(n_topics)])

    # Try to attach gene names if available
    ge_path = (
        Path(cfg.results_dir)
        / "gene_embedding"
        / f"{cfg.dataset}_gene_names_{cfg.n_topics}.pkl"
    )
    if ge_path.exists():
        try:
            with open(ge_path, "rb") as f:
                gene_names = pickle.load(f)
            if isinstance(gene_names, (list, tuple)) and len(gene_names) == df.shape[0]:
                df.index = pd.Index(
                    [str(g).upper() for g in gene_names], name="gene"
                )
        except Exception:
            # Fall back to numeric index if names fail to load
            pass
    return df


def _pick_label_key(adata, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in adata.obs.columns:
        return preferred
    for k in DEFAULT_LABEL_CANDIDATES:
        if k in adata.obs.columns:
            return k
    return None


def _load_celltype_genesets(base_dataset: str) -> Dict[str, set]:
    """
    Load cell-type marker gene sets for a dataset.

    Expects CSV at data/gene_sets/{dataset}_genesets.csv with columns:
    - cell_type
    - gene
    """
    path = GENESET_DIR / f"{base_dataset}_genesets.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cell-type geneset file not found: {path}")
    df = pd.read_csv(path)
    if "cell_type" not in df.columns or "gene" not in df.columns:
        raise ValueError(
            f"{path} must contain 'cell_type' and 'gene' columns for genesets."
        )
    df["cell_type"] = df["cell_type"].astype(str)
    df["gene"] = df["gene"].astype(str).str.upper()
    genesets: Dict[str, set] = {}
    for ct, sub in df.groupby("cell_type"):
        genesets[ct] = set(sub["gene"].tolist())
    return genesets


def _load_go_bp_genesets(base_dataset: str) -> Dict[str, set]:
    """
    Load GO_BP gene sets for a dataset.

    Expects CSV at data/gene_sets/{dataset}_GO_BP_genesets.csv with columns:
    - go_term (or similar name)
    - gene
    """
    path = GENESET_DIR / f"{base_dataset}_GO_BP_genesets.csv"
    if not path.exists():
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


def _ora_from_gene_topics(
    gene_topic_df: pd.DataFrame,
    genesets: Dict[str, set],
    top_n: int = 50,
) -> Tuple[float, pd.DataFrame]:
    """
    Core ORA computation.

    Returns (ORA_score, ORA_table_df) where:
    - ORA_score is the mean -log10(best FDR per topic)
    - ORA_table_df has columns: ['topic', 'geneset', 'p_raw', 'p_adj']
    """
    if gene_topic_df.empty:
        raise ValueError("Gene-topic matrix is empty; cannot compute ORA.")
    if not genesets:
        raise ValueError("Geneset dictionary is empty; cannot compute ORA.")

    all_genes = [str(g).upper() for g in gene_topic_df.index]
    background_genes = set(all_genes)
    if not background_genes:
        raise ValueError("No background genes available for ORA.")

    topic_names = list(gene_topic_df.columns)
    p_records: List[Tuple[str, str, float]] = []

    for topic in topic_names:
        weights = gene_topic_df[topic].values
        order = np.argsort(-weights)
        top_idx = order[: min(top_n, len(order))]
        top_genes = set(gene_topic_df.index[top_idx].str.upper())
        if not top_genes:
            continue

        for gs_name, gs_genes in genesets.items():
            gs_genes_bg = gs_genes & background_genes
            if not gs_genes_bg:
                continue

            a = len(top_genes & gs_genes_bg)
            b = len(top_genes) - a
            c = len(gs_genes_bg) - a
            d = len(background_genes) - (a + b + c)
            if min(a, b, c, d) < 0:
                continue

            table = np.array([[a, b], [c, d]], dtype=int)
            try:
                _, p = fisher_exact(table, alternative="greater")
            except Exception:
                continue
            if np.isnan(p):
                continue
            p_records.append((topic, gs_name, float(p)))

    if not p_records:
        raise ValueError("ORA produced no valid (topic, geneset) p-values.")

    m = len(p_records)
    rows: List[Tuple[str, str, float, float]] = []
    best_p_by_topic: Dict[str, float] = {}
    for topic, gs_name, p in p_records:
        p_adj = min(p * m, 1.0)
        rows.append((topic, gs_name, p, p_adj))
        if topic not in best_p_by_topic or p_adj < best_p_by_topic[topic]:
            best_p_by_topic[topic] = p_adj

    if not best_p_by_topic:
        raise ValueError("ORA best_p_by_topic is empty.")

    eps = 1e-300
    scores = [-float(np.log10(max(p, eps))) for p in best_p_by_topic.values()]
    ora_score = float(np.mean(scores))
    df = pd.DataFrame(rows, columns=["topic", "geneset", "p_raw", "p_adj"])
    return ora_score, df


def _summarize_significant_pathways(
    df: pd.DataFrame,
    alpha: float = 0.05,
    topic_col: str = "topic",
    fdr_col: str = "p_adj",
) -> Tuple[float, float]:
    """
    Summarize proportion of topics with ≥1 significant pathway and mean
    number of significant pathways per topic.
    """
    if df.empty:
        raise ValueError("ORA table is empty when summarizing pathways.")
    if topic_col not in df.columns or fdr_col not in df.columns:
        raise ValueError(
            f"ORA table must contain columns '{topic_col}' and '{fdr_col}'."
        )

    topics = df[topic_col].unique()
    k = len(topics)
    if k == 0:
        raise ValueError("ORA table contains no topics.")

    mk = df[df[fdr_col] < alpha].groupby(topic_col).size()
    n_topic_sig = (mk >= 1).sum()
    prop = float(n_topic_sig / float(k))
    mk_all = mk.reindex(topics, fill_value=0)
    mean_mk = float(mk_all.mean())
    return prop, mean_mk


def _gsea_es_from_mask(scores: np.ndarray, hits_mask: np.ndarray, p: float = 1.0) -> float:
    """One-dimensional GSEA running-sum statistic for a given hit mask."""
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


def _gsea_from_gene_topics(
    gene_topic_df: pd.DataFrame,
    genesets: Dict[str, List[str]],
    p: float = 1.0,
) -> float:
    """
    GSEA-style score:
    - For each topic, compute max |ES| over all genesets
    - Return mean of these per-topic |ES|
    """
    if gene_topic_df.empty:
        raise ValueError("Gene-topic matrix is empty; cannot compute GSEA.")
    if not genesets:
        raise ValueError("Geneset dictionary is empty; cannot compute GSEA.")

    df = gene_topic_df.copy()
    df.index = df.index.map(lambda x: str(x).upper())
    df = df.groupby(df.index).mean()

    genes = df.index.to_numpy()
    topic_names = list(df.columns)

    hits_masks: Dict[str, np.ndarray] = {}
    for gs_name, gs_genes in genesets.items():
        gs_upper = {str(g).upper() for g in gs_genes}
        mask = np.isin(genes, list(gs_upper))
        if mask.sum() == 0:
            continue
        hits_masks[gs_name] = mask

    if not hits_masks:
        raise ValueError(
            "Genesets and gene_topic matrix have almost no overlap; GSEA undefined."
        )

    best_es_abs_per_topic: List[float] = []
    for topic in topic_names:
        scores = df[topic].to_numpy()
        best_abs = 0.0
        for mask in hits_masks.values():
            es = _gsea_es_from_mask(scores, mask, p=p)
            if abs(es) > best_abs:
                best_abs = abs(es)
        if best_abs > 0:
            best_es_abs_per_topic.append(best_abs)

    if not best_es_abs_per_topic:
        raise ValueError("All topics have zero GSEA enrichment scores.")
    return float(np.mean(best_es_abs_per_topic))


def _compute_tc_extrinsic_from_genesets(
    gene_topic_df: pd.DataFrame,
    genesets: Dict[str, set],
    top_n: int = 10,
    eps: float = 1e-12,
) -> float:
    """
    Extrinsic Topic Coherence based on pathway-style genesets.

    For each topic, we take the top-n genes (by weight) and compute
    NPMI over pathway co-occurrence induced by the genesets (e.g., GO_BP).
    """
    if gene_topic_df.empty:
        raise ValueError("Gene-topic matrix is empty; cannot compute TC.")
    if not genesets:
        raise ValueError("Geneset dictionary is empty; cannot compute TC.")

    # Normalize gene names and collapse duplicates, as in GSEA computation.
    df = gene_topic_df.copy()
    df.index = df.index.map(lambda x: str(x).upper())
    df = df.groupby(df.index).mean()

    gene_names = df.index.to_numpy()
    gene_topic = df.to_numpy().T  # topics × genes
    n_topics, n_genes = gene_topic.shape

    # Build gene -> set[pathway_name] mapping.
    gene_to_paths: Dict[str, set] = {}
    for path_name, gs_genes in genesets.items():
        for g in gs_genes:
            gu = str(g).upper()
            gene_to_paths.setdefault(gu, set()).add(path_name)
    total_pathways = len(genesets)
    if total_pathways <= 0:
        raise ValueError("No pathways available for TC computation.")

    tc_values: List[float] = []

    for t in range(n_topics):
        weights = gene_topic[t, :]

        if top_n >= n_genes:
            top_idx = np.argsort(weights)[::-1]
        else:
            top_idx = np.argpartition(weights, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(weights[top_idx])[::-1]]

        top_genes = gene_names[top_idx]

        npmi_scores: List[float] = []
        for g1, g2 in combinations(top_genes, 2):
            p1 = gene_to_paths.get(str(g1).upper())
            p2 = gene_to_paths.get(str(g2).upper())
            if not p1 or not p2:
                continue

            p_i = len(p1) / float(total_pathways)
            p_j = len(p2) / float(total_pathways)
            inter = p1 & p2
            if not inter:
                continue

            p_ij = len(inter) / float(total_pathways)
            if p_i <= eps or p_j <= eps or p_ij <= eps:
                continue

            pmi = np.log(p_ij / (p_i * p_j) + eps)
            npmi = pmi / (-np.log(p_ij + eps))
            npmi_scores.append(float(npmi))

        tc_values.append(float(np.mean(npmi_scores)) if npmi_scores else np.nan)

    valid_tc = [x for x in tc_values if not np.isnan(x)]
    if not valid_tc:
        return float("nan")
    return float(np.mean(valid_tc))


def _compute_topic_diversity(
    gene_topic_df: pd.DataFrame,
    top_n: int = 10,
) -> float:
    """
    Topic Diversity: fraction of unique genes in top-k sets across all topics.
    """
    if gene_topic_df.empty:
        raise ValueError("Gene-topic matrix is empty; cannot compute TD.")

    gene_topic = gene_topic_df.to_numpy().T  # topics × genes
    n_topics, n_genes = gene_topic.shape

    if n_topics == 0 or n_genes == 0:
        raise ValueError("Gene-topic matrix has zero topics or genes.")

    all_top_idx: List[int] = []

    for t in range(n_topics):
        weights = gene_topic[t, :]

        if top_n >= n_genes:
            idx = np.argsort(weights)[::-1]
        else:
            idx = np.argpartition(weights, -top_n)[-top_n:]
            idx = idx[np.argsort(weights[idx])[::-1]]

        all_top_idx.extend(idx.tolist())

    if not all_top_idx:
        return float("nan")

    all_top_idx_arr = np.asarray(all_top_idx, dtype=int)
    td_value = np.unique(all_top_idx_arr).size / float(n_topics * top_n)
    return float(td_value)


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
    """Run evaluation on a single training output and return metrics."""
    # ---------------------------
    # Clustering quality (ARI / NMI)
    # ---------------------------
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

    # Pick label key and build AnnData for clustering
    label_key = _pick_label_key(adata_orig, cfg.label_key)
    if label_key is None:
        raise KeyError(
            f"Could not find a label column. Tried: "
            f"{[cfg.label_key] if cfg.label_key else []} + {DEFAULT_LABEL_CANDIDATES}"
        )

    adata = sc.AnnData(X)
    adata.obs[label_key] = labels_df[label_key].values

    # Compute ARI / NMI via resolution scan
    best_res, ari, nmi = _scan_resolutions_for_best_ari(
        adata=adata,
        label_key=label_key,
        res_min=cfg.res_min,
        res_max=cfg.res_max,
        res_step=cfg.res_step,
        seed=cfg.seed,
    )

    metrics: Dict[str, Optional[float]] = {
        "best_resolution": float(best_res),
        "ARI": float(ari),
        "NMI": float(nmi),
        "n_cells": int(n),
        "n_topics": int(X.shape[1]) if X.ndim == 2 else int(cfg.n_topics or -1),
        "dataset": cfg.dataset if cfg.dataset else "",
        "tag": cfg.tag or "",
    }

    # ---------------------------
    # Additional metrics (from evaluation-2025.ipynb)
    # - ASW (silhouette, cell-type labels)
    # - ORA (cell-type markers / GO_BP) + SPP
    # - GSEA (cell-type markers / GO_BP)
    # ---------------------------
    base_dataset = Path(cfg.adata_path).stem

    # Gene-topic based metrics (ORA / GSEA)
    gene_topic_df: Optional[pd.DataFrame] = None
    try:
        gene_topic_df = _load_topic_gene_matrix(cfg)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[evaluation.py] Gene-topic matrix unavailable; "
            f"skipping ORA/GSEA metrics: {exc}"
        )

    if gene_topic_df is not None:
        # TC / TD metrics based on genesets (using GO_BP as pathway-style sets).
        try:
            go_genesets_tc = _load_go_bp_genesets(base_dataset)
            tc_extrinsic = _compute_tc_extrinsic_from_genesets(
                gene_topic_df, go_genesets_tc, top_n=10
            )
            metrics["TC_extrinsic_GO_BP_top10"] = (
                None if np.isnan(tc_extrinsic) else float(tc_extrinsic)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluation.py] TC (extrinsic, GO_BP) unavailable: {exc}")
            metrics.setdefault("TC_extrinsic_GO_BP_top10", None)

        try:
            td_value = _compute_topic_diversity(gene_topic_df, top_n=10)
            metrics["TD_top10"] = None if np.isnan(td_value) else float(td_value)
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluation.py] TD (top10) unavailable: {exc}")
            metrics.setdefault("TD_top10", None)

        # ORA + SPP (cell-type marker genes)
        try:
            ct_genesets = _load_celltype_genesets(base_dataset)
            ora_markers, ora_table = _ora_from_gene_topics(
                gene_topic_df, ct_genesets, top_n=50
            )
            prop, mean_mk = _summarize_significant_pathways(
                ora_table, alpha=0.05
            )
            metrics["ORA_markers"] = ora_markers
            metrics["ORA_markers_SPP_prop"] = prop
            metrics["ORA_markers_SPP_mean"] = mean_mk
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluation.py] ORA (cell-type markers) unavailable: {exc}")
            metrics.setdefault("ORA_markers", None)
            metrics.setdefault("ORA_markers_SPP_prop", None)
            metrics.setdefault("ORA_markers_SPP_mean", None)

        # ORA + SPP (GO_BP gene sets)
        try:
            go_genesets = _load_go_bp_genesets(base_dataset)
            ora_go, ora_go_table = _ora_from_gene_topics(
                gene_topic_df, go_genesets, top_n=50
            )
            prop_go, mean_mk_go = _summarize_significant_pathways(
                ora_go_table, alpha=0.05
            )
            metrics["ORA_GO_BP"] = ora_go
            metrics["ORA_GO_BP_SPP_prop"] = prop_go
            metrics["ORA_GO_BP_SPP_mean"] = mean_mk_go
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluation.py] ORA (GO_BP) unavailable: {exc}")
            metrics.setdefault("ORA_GO_BP", None)
            metrics.setdefault("ORA_GO_BP_SPP_prop", None)
            metrics.setdefault("ORA_GO_BP_SPP_mean", None)

        # GSEA (cell-type markers)
        try:
            ct_genesets_gsea = _load_celltype_genesets(base_dataset)
            gsea_markers = _gsea_from_gene_topics(
                gene_topic_df, ct_genesets_gsea, p=1.0
            )
            metrics["GSEA_markers"] = gsea_markers
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluation.py] GSEA (cell-type markers) unavailable: {exc}")
            metrics.setdefault("GSEA_markers", None)

        # GSEA (GO_BP gene sets)
        try:
            go_genesets_gsea = _load_go_bp_genesets(base_dataset)
            gsea_go = _gsea_from_gene_topics(
                gene_topic_df, go_genesets_gsea, p=1.0
            )
            metrics["GSEA_GO_BP"] = gsea_go
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluation.py] GSEA (GO_BP) unavailable: {exc}")
            metrics.setdefault("GSEA_GO_BP", None)

    # Persist optional json (with all metrics)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = cfg.dataset or cell_topic_path.stem
        nt = cfg.n_topics if cfg.n_topics is not None else metrics["n_topics"]
        tag_str = cfg.tag.strip() if cfg.tag else ""
        suffix = f"_{tag_str}" if tag_str else ""
        out_path = out_dir / f"{ds}_cluster_metrics_{nt}{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved evaluation metrics to {out_path}")

    # Console summary
    print("\n📈 Clustering Quality (Louvain/Leiden scan)")
    if metrics["dataset"]:
        print("- Dataset:", metrics["dataset"])
    print(f"- Cells: {metrics['n_cells']}")
    print(f"- Best resolution: {metrics['best_resolution']:.2f}")
    print(f"- ARI: {metrics['ARI']:.4f}")
    print(f"- NMI: {metrics['NMI']:.4f}")

    # Additional metrics summary
    print("\n🔎 Additional metrics")
    ora_markers = metrics.get("ORA_markers")
    if ora_markers is not None:
        print(
            "- ORA (cell-type markers, mean -log10 FDR): "
            f"{ora_markers:.4f}"
        )
    else:
        print("- ORA (cell-type markers): N/A")

    ora_go = metrics.get("ORA_GO_BP")
    if ora_go is not None:
        print(
            "- ORA (GO_BP, mean -log10 FDR): "
            f"{ora_go:.4f}"
        )
    else:
        print("- ORA (GO_BP): N/A")

    spp_prop = metrics.get("ORA_markers_SPP_prop")
    spp_mean = metrics.get("ORA_markers_SPP_mean")
    if spp_prop is not None and spp_mean is not None:
        print(
            f"- SPP (ORA markers): Prop={spp_prop:.3f}, "
            f"Mean={spp_mean:.2f}"
        )
    else:
        print("- SPP (ORA markers): N/A")

    spp_prop_go = metrics.get("ORA_GO_BP_SPP_prop")
    spp_mean_go = metrics.get("ORA_GO_BP_SPP_mean")
    if spp_prop_go is not None and spp_mean_go is not None:
        print(
            f"- SPP (ORA GO_BP): Prop={spp_prop_go:.3f}, "
            f"Mean={spp_mean_go:.2f}"
        )
    else:
        print("- SPP (ORA GO_BP): N/A")

    tc_ext = metrics.get("TC_extrinsic_GO_BP_top10")
    if tc_ext is not None:
        print(f"- TC_extrinsic (GO_BP, top10): {tc_ext:.4f}")
    else:
        print("- TC_extrinsic (GO_BP, top10): N/A")

    td_top10 = metrics.get("TD_top10")
    if td_top10 is not None:
        print(f"- TD (top10): {td_top10:.4f}")
    else:
        print("- TD (top10): N/A")

    gsea_markers = metrics.get("GSEA_markers")
    if gsea_markers is not None:
        print(f"- GSEA (cell-type markers): {gsea_markers:.4f}")
    else:
        print("- GSEA (cell-type markers): N/A")

    gsea_go = metrics.get("GSEA_GO_BP")
    if gsea_go is not None:
        print(f"- GSEA (GO_BP): {gsea_go:.4f}")
    else:
        print("- GSEA (GO_BP): N/A")

    return metrics


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Compute ARI/NMI on cell-topic matrix (post-training)")
    p.add_argument("--adata_path", required=True, type=str, help="Path to input .h5ad with labels in .obs")
    p.add_argument("--results_dir", type=str, default=None, help="Results root (expects cell_topic/ subdir)")
    p.add_argument("--dataset", type=str, default=None, help="Dataset name used during training")
    p.add_argument("--n_topics", type=int, default=None, help="Number of topics used during training")
    p.add_argument("--cell_topic_file", type=str, default=None, help="Direct path to a cell_topic_matrix .pkl")
    p.add_argument("--label_key", type=str, default=None, help="Label column in .obs (defaults to common names)")
    p.add_argument("--tag", type=str, default=None, help="Optional tag for metrics filename disambiguation")
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
        tag=a.tag,
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
