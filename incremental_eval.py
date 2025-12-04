#!/usr/bin/env python3
import argparse

from incremental_eval import run_incremental_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incremental TopicStore merge with per-step evaluation payloads"
    )
    p.add_argument(
        "--results_dir",
        default="results",
        help="Root results directory",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to merge in order (e.g., PBMC4k_scVI PBMC8k_scVI PBMC12k_scVI)",
    )
    p.add_argument(
        "--store_path",
        default="results/topic_store/topic_store.pkl",
        help="Path to save/load TopicStore",
    )
    p.add_argument(
        "--out_dir",
        default="results/topic_store/merge_eval",
        help="Directory to save evaluation payloads",
    )
    # Alignment/merge knobs for UOT
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--reg_m", type=float, default=10.0)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--smoothing", type=float, default=0.5)
    p.add_argument("--min_transport_mass", type=float, default=1e-3)
    p.add_argument("--min_best_ratio", type=float, default=0.5)
    # Background topic filtering flags
    p.add_argument(
        "--filter_background",
        action="store_true",
        default=True,
        help="Enable background-topic filtering before UOT (default: on)",
    )
    p.add_argument(
        "--sparsity_threshold",
        type=float,
        default=0.20,
        help="Hoyer sparsity threshold to keep topics (0..1)",
    )
    p.add_argument(
        "--topk_mass_threshold",
        type=float,
        default=-1.0,
        help="Minimum mass in top-k genes (<=0 to disable)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=50,
        help="k for top-k mass threshold",
    )
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

