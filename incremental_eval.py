#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from incremental import TopicStore


def _find_single_file(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
    return None


def _load_dataset_artifacts(results_dir: str, dataset: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load topic and gene embeddings (and gene_names if present) for a dataset."""
    topic_pat = [os.path.join(results_dir, "topic_embedding", f"{dataset}*topic_embeddings*.pkl")]
    gene_pat = [os.path.join(results_dir, "gene_embedding", f"{dataset}*gene_embeddings*.pkl")]
    names_pat = [os.path.join(results_dir, "gene_embedding", f"{dataset}*gene_names*.pkl")]

    t_path = _find_single_file(topic_pat)
    g_path = _find_single_file(gene_pat)
    n_path = _find_single_file(names_pat)
    if t_path is None or g_path is None:
        raise FileNotFoundError(f"Missing artifacts for {dataset} under {results_dir}")
    with open(t_path, "rb") as f:
        topic_embeddings = np.asarray(pickle.load(f), dtype=np.float32)
    with open(g_path, "rb") as f:
        gene_embeddings = np.asarray(pickle.load(f), dtype=np.float32)
    if n_path is not None:
        with open(n_path, "rb") as f:
            gene_names = list(pickle.load(f))
    else:
        G = int(gene_embeddings.shape[0])
        gene_names = [f"GENE_{i}" for i in range(G)]
    return topic_embeddings, gene_embeddings, gene_names


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return x / nrm


def _cosine_rowwise(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a_n = _l2_normalize_rows(a, eps)
    b_n = _l2_normalize_rows(b, eps)
    return np.sum(a_n * b_n, axis=1)


def run_incremental_eval(
    results_dir: str,
    datasets: List[str],
    store_path: str,
    out_dir: str,
    *,
    reg: float = 0.05,
    reg_m: float = 10.0,
    metric: str = "euclidean",
    smoothing: float = 0.5,
    min_transport_mass: float = 1e-3,
    min_best_ratio: float = 0.5,
    # Background topic filtering
    filter_background: bool = True,
    sparsity_threshold: float = 0.20,
    topk_mass_threshold: float = -1.0,
    topk: int = 50,
) -> None:
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Load or init store
    if os.path.exists(store_path):
        store = TopicStore.load(store_path)
        print(f"Loaded TopicStore with {store.size} topics from {store_path}")
    else:
        store = TopicStore()
        print("Initialized empty TopicStore")

    for idx, ds in enumerate(datasets):
        print(f"\n=== Merge step {idx+1}/{len(datasets)}: {ds} ===")

        # Snapshot store before merge
        pre_store_emb = store.store_embeddings.copy() if store.size > 0 else np.zeros((0, 0), dtype=np.float32)
        pre_size = pre_store_emb.shape[0]
        pre_ids = list(store.topic_ids)

        # Execute merge with coupling returned
        stats = store.add_topics(
            dataset_name=ds,
            results_dir=results_dir,
            reg=reg,
            reg_m=reg_m,
            metric=metric,
            smoothing=smoothing,
            min_transport_mass=min_transport_mass,
            min_best_ratio=min_best_ratio,
            filter_background=filter_background,
            sparsity_threshold=sparsity_threshold,
            topk_mass_threshold=(None if topk_mass_threshold is None or topk_mass_threshold <= 0 else float(topk_mass_threshold)),
            topk=topk,
            return_coupling=True,
        )

        coupling = stats.get("coupling", None)
        if coupling is None:
            print("Warning: coupling not available; similarity measures will be limited.")
        # Report filtering stats if available
        filtered_orig = stats.get("filtered", None)
        try:
            k_total = int(new_tg.shape[0])
        except Exception:
            k_total = None
        if filtered_orig is not None:
            n_filt = len(filtered_orig)
            if k_total is not None:
                print(f"Filtered topics: {n_filt} / {k_total}")
            else:
                print(f"Filtered topics: {n_filt}")

        # Load dataset artifacts again to form aligned new topic-gene vectors (to current store)
        t_emb, g_emb, g_names = _load_dataset_artifacts(results_dir, ds)
        new_tg = np.asarray(t_emb @ g_emb.T, dtype=np.float32)
        # Align to current store gene order without expanding further
        if new_tg.ndim == 1:
            new_tg = new_tg[None, :]
        new_aligned = store._align_genes(new_tg, g_names, expand=False)  # (K, G_store_after)
        new_norm = _l2_normalize_rows(new_aligned)

        # Build barycentric projections. Use the pre-existing portion of the
        # store AFTER the merge (rows 0..pre_size-1), because add_topics may
        # expand gene dimensions before computing the coupling.
        matched_pairs: List[Tuple[int, int]] = stats["matched"]
        added_js: List[int] = stats["added"]
        start_added = pre_size
        store_added_idx = list(range(start_added, start_added + len(added_js)))

        # Compute best ratio per new topic from coupling, and barycenters
        ratios = None
        bary = None
        mass_new = None
        if coupling is not None and pre_size > 0:
            mass_new = coupling.sum(axis=0)
            best_idx = np.argmax(coupling, axis=0)
            best_val = coupling[best_idx, np.arange(coupling.shape[1])]
            total = mass_new + 1e-12
            ratios = best_val / total
            # Barycentric projection for each new topic j over the pre-existing
            # rows of the store AFTER potential gene-dimension expansion.
            store_base = store.store_embeddings[:pre_size]
            bary = np.zeros((coupling.shape[1], store_base.shape[1]), dtype=np.float32)
            base_mean = store_base.mean(axis=0) if pre_size > 0 else np.zeros((store_base.shape[1],), dtype=np.float32)
            for j in range(coupling.shape[1]):
                m = mass_new[j]
                if m <= 1e-8 or pre_size == 0:
                    bary[j] = base_mean
                else:
                    bary[j] = (coupling[:, j][:, None] * store_base).sum(axis=0) / m

        # Collect matched evaluation payload
        if matched_pairs:
            i_store = [i for (i, j) in matched_pairs]
            j_new_orig = [j for (i, j) in matched_pairs]
            # Build mapping from original indices -> filtered (coupling-column) indices
            K_orig = new_norm.shape[0]
            filtered_orig = stats.get("filtered", None)
            if filtered_orig is not None:
                keep_mask = np.ones(K_orig, dtype=bool)
                keep_mask[np.asarray(filtered_orig, dtype=int)] = False
                keep_idx = np.nonzero(keep_mask)[0]
            else:
                keep_idx = np.arange(K_orig)
            orig_to_rel = -np.ones(K_orig, dtype=int)
            for rel, orig in enumerate(keep_idx):
                orig_to_rel[int(orig)] = int(rel)
            j_new_rel = [int(orig_to_rel[j]) for j in j_new_orig]
            store_ids = [pre_ids[i] for i in i_store] if pre_ids else [f"T{i}" for i in i_store]
            # Store vectors after merge (same dim as new_aligned)
            store_after = store.store_embeddings[i_store]
            # New aligned topic vectors are indexed by original indices
            new_vecs = new_norm[j_new_orig]
            # Bary/ratios/mass computed over filtered columns of coupling
            bary_vecs = bary[j_new_rel] if bary is not None else None
            ratio_sel = ratios[j_new_rel] if ratios is not None else None
            mass_sel = mass_new[j_new_rel] if mass_new is not None else None

            # Cosine similarities (if we have the vectors)
            cos_new_bary = _cosine_rowwise(new_vecs, bary_vecs) if (new_vecs is not None and bary_vecs is not None) else None
            cos_new_after = _cosine_rowwise(new_vecs, store_after) if new_vecs is not None else None
            # cos_before_after is omitted here to avoid dimension mismatch when
            # add_topics expands gene dimensions before coupling. It can be
            # added later with an explicit alignment of pre-merge vectors.
            cos_before_after = None

            payload = {
                "dataset": ds,
                "matched_pairs": matched_pairs,
                "store_indices": i_store,
                "new_indices": j_new_orig,
                "store_ids": store_ids,
                "ratio": ratio_sel,
                "mass": mass_sel,
                "store_before": None,
                "bary": bary_vecs,
                "store_after": store_after,
                "new_aligned": new_vecs,
                "cos_new_bary": cos_new_bary,
                "cos_new_after": cos_new_after,
                "cos_before_after": cos_before_after,
                "gene_names": list(store.gene_names),
            }
            out_path = os.path.join(out_dir, f"{ds}_matched_eval.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(payload, f)
            print(f"Saved matched eval payload: {out_path} ({len(matched_pairs)} pairs)")
        else:
            print("No matched pairs in this step.")

        # Collect added evaluation payload
        if added_js:
            new_added_vecs = new_norm[added_js]
            store_added_vecs = store.store_embeddings[store_added_idx]
            payload = {
                "dataset": ds,
                "new_indices": added_js,
                "store_added_indices": store_added_idx,
                "assigned_ids": stats.get("assigned_ids", []),
                "new_aligned": new_added_vecs,
                "store_added": store_added_vecs,
                "gene_names": list(store.gene_names),
            }
            out_path = os.path.join(out_dir, f"{ds}_added_eval.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(payload, f)
            print(f"Saved added eval payload: {out_path} ({len(added_js)} topics)")
        else:
            print("No added topics in this step.")

        # Persist store after this step
        store.save(store_path)
        print(f"Store saved to {store_path} (size={store.size})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental TopicStore merge with per-step evaluation payloads")
    p.add_argument("--results_dir", default="results", help="Root results directory")
    p.add_argument("--datasets", nargs="+", required=True, help="Datasets to merge in order (e.g., PBMC4k_scVI PBMC8k_scVI PBMC12k_scVI)")
    p.add_argument("--store_path", default="results/topic_store/topic_store.pkl", help="Path to save/load TopicStore")
    p.add_argument("--out_dir", default="results/topic_store/merge_eval", help="Directory to save evaluation payloads")
    # Alignment/merge knobs for UOT
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--reg_m", type=float, default=10.0)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--smoothing", type=float, default=0.5)
    p.add_argument("--min_transport_mass", type=float, default=1e-3)
    p.add_argument("--min_best_ratio", type=float, default=0.5)
    # Background topic filtering flags
    p.add_argument("--filter_background", action="store_true", default=True, help="Enable background-topic filtering before UOT (default: on)")
    p.add_argument("--sparsity_threshold", type=float, default=0.20, help="Hoyer sparsity threshold to keep topics (0..1)")
    p.add_argument("--topk_mass_threshold", type=float, default=-1.0, help="Minimum mass in top-k genes (<=0 to disable)")
    p.add_argument("--topk", type=int, default=50, help="k for top-k mass threshold")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_incremental_eval(
        results_dir=args.results_dir,
        datasets=args.datasets,
        store_path=args.store_path,
        out_dir=args.out_dir,
        reg=args.reg,
        reg_m=args.reg_m,
        metric=args.metric,
        smoothing=args.smoothing,
        min_transport_mass=args.min_transport_mass,
        min_best_ratio=args.min_best_ratio,
        filter_background=args.filter_background,
        sparsity_threshold=args.sparsity_threshold,
        topk_mass_threshold=args.topk_mass_threshold,
        topk=args.topk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
