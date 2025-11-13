#!/usr/bin/env python3
"""
Add topics incrementally into a global TopicStore.

Two usage patterns:
- By dataset name: expects artifacts under a results directory laid out like
  results/{topic_embedding,gene_embedding,cell_topic}/<dataset>_*.pkl.
- By explicit artifact paths: pass --topic_embeddings/--gene_embeddings and
  optionally --gene_names/--cell_topic, and this script will stage them into a
  temporary results layout and then merge into the store.

This mirrors incremental_eval.py but focuses on adding topics (no reports).
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
from typing import Optional

import numpy as np

from incremental import TopicStore


def _dump_pickle(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def _stage_artifacts(
    *,
    dataset: str,
    topic_embeddings_path: str,
    gene_embeddings_path: str,
    gene_names_path: Optional[str] = None,
    cell_topic_path: Optional[str] = None,
) -> str:
    """Create a temporary results-like directory with the provided artifacts.

    Returns the root of the staged results directory.
    """
    stage_root = tempfile.mkdtemp(prefix="incr_stage_")
    try:
        # Load provided artifacts
        with open(topic_embeddings_path, "rb") as f:
            topic_embeddings = np.asarray(pickle.load(f), dtype=np.float32)
        with open(gene_embeddings_path, "rb") as f:
            gene_embeddings = np.asarray(pickle.load(f), dtype=np.float32)
        gene_names = None
        if gene_names_path is not None and os.path.exists(gene_names_path):
            with open(gene_names_path, "rb") as f:
                gene_names = list(pickle.load(f))
        cell_topic = None
        if cell_topic_path is not None and os.path.exists(cell_topic_path):
            with open(cell_topic_path, "rb") as f:
                cell_topic = np.asarray(pickle.load(f), dtype=np.float32)

        # Dump into the staged layout using permissive filename patterns
        _dump_pickle(topic_embeddings, os.path.join(stage_root, "topic_embedding", f"{dataset}_topic_embeddings.pkl"))
        _dump_pickle(gene_embeddings, os.path.join(stage_root, "gene_embedding", f"{dataset}_gene_embeddings.pkl"))
        if gene_names is not None:
            _dump_pickle(gene_names, os.path.join(stage_root, "gene_embedding", f"{dataset}_gene_names.pkl"))
        if cell_topic is not None:
            _dump_pickle(cell_topic, os.path.join(stage_root, "cell_topic", f"{dataset}_cell_topic_matrix.pkl"))
    except Exception:
        # Clean staged dir on failure then re-raise
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return stage_root


def main() -> int:
    p = argparse.ArgumentParser(description="Incrementally add topics to TopicStore")
    # Store and dataset bookkeeping
    p.add_argument("--dataset", required=True, help="Dataset name label for the incoming topics")
    p.add_argument("--store_path", default="results/topic_store/topic_store.pkl", help="Path to save/load TopicStore")
    p.add_argument("--results_dir", default="results", help="Results root (when using --from_dataset)")

    # Two input modes
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--from_dataset", action="store_true", help="Load artifacts by dataset name under --results_dir")
    m.add_argument("--from_paths", action="store_true", help="Load artifacts from explicit pickle paths")

    # Explicit paths mode
    p.add_argument("--topic_embeddings", help="Path to topic_embeddings .pkl (required with --from_paths)")
    p.add_argument("--gene_embeddings", help="Path to gene_embeddings .pkl (required with --from_paths)")
    p.add_argument("--gene_names", help="Optional path to gene_names .pkl")
    p.add_argument("--cell_topic", help="Optional path to cell_topic_matrix .pkl (used as weights)")

    # UOT/merge knobs (kept consistent with incremental_eval.py / TopicStore.add_topics)
    p.add_argument("--metric", default="euclidean", help="Distance metric for UOT cost (euclidean|cosine)")
    p.add_argument("--reg", type=float, default=0.05, help="UOT entropy regularization")
    p.add_argument("--reg_m", type=float, default=10.0, help="UOT mass regularization")
    p.add_argument("--smoothing", type=float, default=0.5, help="EMA smoothing for matched topics")
    p.add_argument("--min_transport_mass", type=float, default=1e-3, help="Minimum transported mass to consider as match")
    p.add_argument("--min_best_ratio", type=float, default=0.5, help="Best/total mass ratio threshold for match")
    p.add_argument("--filter_background", action="store_true", help="Enable background topic filtering before alignment")
    p.add_argument("--sparsity_threshold", type=float, default=0.20, help="Hoyer sparsity threshold for filtering")
    p.add_argument("--topk_mass_threshold", type=float, default=-1.0, help="Optional top-k mass threshold (<=0 disables)")
    p.add_argument("--topk", type=int, default=50, help="k for top-k mass threshold if enabled")
    p.add_argument("--expand_genes", action="store_true", help="Allow store to expand its gene set if needed")

    args = p.parse_args()

    os.makedirs(os.path.dirname(args.store_path), exist_ok=True)

    # Load or initialise store
    if os.path.exists(args.store_path):
        store = TopicStore.load(args.store_path)
        print(f"Loaded TopicStore with {store.size} topics from {args.store_path}")
    else:
        store = TopicStore()
        print("Initialized empty TopicStore")

    # Prepare a results directory for this addition
    results_dir = args.results_dir
    stage_dir: Optional[str] = None
    if args.from_paths:
        if not args.topic_embeddings or not args.gene_embeddings:
            raise SystemExit("--from_paths requires --topic_embeddings and --gene_embeddings")
        stage_dir = _stage_artifacts(
            dataset=args.dataset,
            topic_embeddings_path=args.topic_embeddings,
            gene_embeddings_path=args.gene_embeddings,
            gene_names_path=args.gene_names,
            cell_topic_path=args.cell_topic,
        )
        results_dir = stage_dir

    # Execute merge
    stats = store.add_topics(
        dataset_name=args.dataset,
        results_dir=results_dir,
        metric=args.metric,
        reg=args.reg,
        reg_m=args.reg_m,
        smoothing=args.smoothing,
        min_transport_mass=args.min_transport_mass,
        min_best_ratio=args.min_best_ratio,
        filter_background=args.filter_background,
        sparsity_threshold=args.sparsity_threshold,
        topk_mass_threshold=(None if args.topk_mass_threshold is None or args.topk_mass_threshold <= 0 else float(args.topk_mass_threshold)),
        topk=args.topk,
        return_coupling=False,
        expand_genes=args.expand_genes,
    )

    # Persist store
    store.save(args.store_path)

    matched = stats.get("matched", [])
    added = stats.get("added", [])
    assigned_ids = stats.get("assigned_ids", [])
    print("\n=== Incremental add summary ===")
    print(f"Matched: {len(matched)} topics")
    print(f"Added:   {len(added)} topics")
    print(f"Assigned IDs: {assigned_ids[:10]}{'...' if len(assigned_ids) > 10 else ''}")
    print(f"Store size: {store.size}")
    print(f"Saved store to: {args.store_path}")

    # Cleanup staged dir if used
    if stage_dir is not None:
        shutil.rmtree(stage_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

