from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom

PROJECT_ROOT = Path(__file__).resolve().parent

_gobp_path_to_genes_cache = None
_gobp_universe_cache = None
_gobp_sparse_cache = None


def _default_pathway_gene_csv() -> Path | None:
    env = os.environ.get("MSIGDB_PATHWAY_GENE_CSV")
    if env:
        p = Path(env)
        if p.exists():
            return p

    candidates = [
        PROJECT_ROOT / "C2_C5_pathway_gene.csv.gz",
        PROJECT_ROOT / "C2_C5_pathway_gene.csv",
        PROJECT_ROOT / "evaluation" / "MsigDB" / "C2_C5_pathway_gene.csv.gz",
        PROJECT_ROOT / "evaluation" / "MsigDB" / "C2_C5_pathway_gene.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def is_gobp(pathway_name: str) -> bool:
    s = str(pathway_name).upper()
    return (
        s.startswith("GOBP_")
        or s.startswith("GO_BP_")
        or s.startswith("GO_BIOLOGICAL_PROCESS")
        or "BIOLOGICAL_PROCESS" in s
    )


def load_gobp_pathway_db(pathway_gene_csv: Path):
    global _gobp_path_to_genes_cache, _gobp_universe_cache

    if _gobp_path_to_genes_cache is not None:
        return _gobp_path_to_genes_cache, _gobp_universe_cache

    df = pd.read_csv(pathway_gene_csv)
    df = df[["pathway_name", "gene_symbol"]].dropna().drop_duplicates()
    df = df[df["pathway_name"].map(is_gobp)]

    path_to_genes = (
        df.groupby("pathway_name")["gene_symbol"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )
    universe = set(df["gene_symbol"].astype(str))

    print(f"[INFO] GO:BP pathways: {len(path_to_genes)}, universe genes: {len(universe)}")

    _gobp_path_to_genes_cache = path_to_genes
    _gobp_universe_cache = universe
    return path_to_genes, universe


def read_gene_topic(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0)

    cols = df.columns.astype(str)
    idx0 = str(df.index[0])

    cols_are_topics = all(c.startswith("topic_") for c in cols[:3])
    idx_is_topic = idx0.startswith("topic_")

    if cols_are_topics and not idx_is_topic:
        gene_topic = df.values.T
        gene_names = df.index.astype(str).values
    else:
        gene_topic = df.values
        gene_names = df.columns.astype(str).values

    return gene_topic, gene_names


def build_gobp_sparse(pathway_gene_csv: Path, min_term_size=10, max_term_size=500):
    global _gobp_sparse_cache
    if _gobp_sparse_cache is not None:
        return _gobp_sparse_cache

    path_to_genes, universe = load_gobp_pathway_db(pathway_gene_csv)
    universe = sorted(universe)
    gene_to_col = {g: i for i, g in enumerate(universe)}

    rows, cols = [], []
    term_sizes = []

    for term, genes in path_to_genes.items():
        genes_in = [g for g in genes if g in gene_to_col]
        k = len(genes_in)
        if k < min_term_size or k > max_term_size:
            continue
        t = len(term_sizes)
        term_sizes.append(k)
        for g in genes_in:
            rows.append(t)
            cols.append(gene_to_col[g])

    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(term_sizes), len(universe)))

    print(f"[INFO] Sparse GO:BP matrix built: terms={A.shape[0]}, genes={A.shape[1]}")

    _gobp_sparse_cache = (A, np.array(term_sizes), gene_to_col)
    return _gobp_sparse_cache


def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def compute_ora_spp_gobp(
    gene_topic: np.ndarray,
    gene_names,
    top_n=10,
    fdr_cutoff=0.05,
    pathway_gene_csv: Path | None = None,
):
    if pathway_gene_csv is None:
        pathway_gene_csv = _default_pathway_gene_csv()
    if pathway_gene_csv is None or not pathway_gene_csv.exists():
        raise FileNotFoundError(
            "Could not find pathway-gene CSV. Set MSIGDB_PATHWAY_GENE_CSV or place "
            "C2_C5_pathway_gene.csv under the repository root."
        )

    A, term_sizes, gene_to_col = build_gobp_sparse(pathway_gene_csv)
    gene_names = np.asarray(gene_names)

    n_topics, _ = gene_topic.shape
    M = A.shape[1]

    topic_scores = []
    n_sig_topics = 0

    for t in range(n_topics):
        w = gene_topic[t]
        idx = np.argsort(w)[-top_n:][::-1]
        top_genes = [gene_names[i] for i in idx if gene_names[i] in gene_to_col]
        n = len(top_genes)

        if n == 0:
            topic_scores.append(np.nan)
            continue

        cols = [gene_to_col[g] for g in top_genes]
        k_vec = np.asarray(A[:, cols].sum(axis=1)).ravel().astype(int)

        pos = k_vec > 0
        if not np.any(pos):
            topic_scores.append(np.nan)
            continue

        K = term_sizes[pos]
        k = k_vec[pos]

        pvals = hypergeom.sf(k - 1, M, K, n)
        fdrs = bh_fdr(pvals)

        sig = fdrs < fdr_cutoff
        if sig.any():
            n_sig_topics += 1
            score = np.mean(-np.log10(fdrs[sig] + 1e-300))
            topic_scores.append(score)
        else:
            topic_scores.append(np.nan)

    valid = [s for s in topic_scores if not np.isnan(s)]
    ora_score = float(np.mean(valid)) if valid else np.nan
    spp_prop = n_sig_topics / max(1, n_topics)

    return ora_score, spp_prop


def eval_one_gene_topic_file_ora_spp(csv_path: Path, top_n=10):
    csv_path = Path(csv_path)
    gene_topic, gene_names = read_gene_topic(csv_path)
    ora, spp = compute_ora_spp_gobp(gene_topic, gene_names, top_n=top_n)
    return {
        "file": csv_path.name,
        "ORA_GO_BP": ora,
        "SPP_Prop_GO_BP": spp,
    }


def parse_gene_topic_filename(csv_path: Path):
    csv_path = Path(csv_path)
    fname = csv_path.name
    stem = csv_path.stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[-2:] != ["gene", "topic"]:
        raise ValueError(f"Unexpected filename format: {fname}")
    model = parts[0]
    dataset = parts[1]
    K = parts[2]
    return model, dataset, K, fname


def main() -> int:
    p = argparse.ArgumentParser(description="Compute ORA+SPP (GO:BP) from *_gene_topic.csv files.")
    p.add_argument(
        "--gene_topic_dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "gene_topic"),
        help="Directory containing *_gene_topic.csv files.",
    )
    p.add_argument(
        "--pathway_gene_csv",
        type=str,
        default=None,
        help="Path to C2_C5_pathway_gene.csv (optional; otherwise auto-detect or env MSIGDB_PATHWAY_GENE_CSV).",
    )
    p.add_argument("--top_n", type=int, default=10, help="Top-N genes per topic.")
    args = p.parse_args()

    gene_topic_dir = Path(args.gene_topic_dir)
    if not gene_topic_dir.exists():
        raise FileNotFoundError(f"gene_topic_dir not found: {gene_topic_dir}")

    if args.pathway_gene_csv:
        os.environ["MSIGDB_PATHWAY_GENE_CSV"] = args.pathway_gene_csv

    rows = []
    for csv_path in sorted(gene_topic_dir.glob("*_gene_topic.csv")):
        try:
            model, dataset, K, fname = parse_gene_topic_filename(csv_path)
            res = eval_one_gene_topic_file_ora_spp(csv_path, top_n=args.top_n)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "K": K,
                    "file": fname,
                    "ORA_GO_BP": res["ORA_GO_BP"],
                    "SPP_Prop_GO_BP": res["SPP_Prop_GO_BP"],
                }
            )
            print(f"[INFO] processed {csv_path.name}")
        except Exception as e:
            print(f"[WARN] skip {csv_path.name}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["dataset", "model", "K", "file", "ORA_GO_BP", "SPP_Prop_GO_BP"]]
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
