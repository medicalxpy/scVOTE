#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


RUN_DIR_RE = re.compile(r"^(?P<name>.+)_K(?P<k>\d+)$")
CELL_TOPIC_RE = re.compile(r"^(?P<dataset>.+)_cell_topic_matrix_(?P<k>\d+)\.pkl$")


def _load_eval_module(repo_root: Path):
    eval_py = repo_root / "evaluation.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"Missing evaluation.py under repo root: {repo_root}")
    spec = importlib.util.spec_from_file_location("scvote_evaluation_py", eval_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {eval_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iter_run_dirs(results_root: Path, *, include_tuning: bool) -> List[Path]:
    run_dirs: List[Path] = []
    if results_root.exists():
        for p in results_root.iterdir():
            if not p.is_dir():
                continue
            if p.name in {
                "cell_embedding",
                "cell_topic",
                "topic_gene",
                "gene_embedding",
                "topic_embedding",
                "gpu_stats",
                "eval_cache",
                "evaluation",
                "incremental_eval",
                "visualization",
                "tuning",
            }:
                continue
            if RUN_DIR_RE.match(p.name):
                run_dirs.append(p)

    if include_tuning:
        tuning_root = results_root / "tuning"
        if tuning_root.exists():
            for p in tuning_root.iterdir():
                if p.is_dir():
                    run_dirs.append(p)
    return sorted(run_dirs, key=lambda x: x.name)


def _infer_k_from_run_dir(run_dir: Path) -> Optional[int]:
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        return None
    return int(m.group("k"))


def _infer_method_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    if "_structure_" in name:
        return "structure"
    if "_contrastive_" in name:
        return "contrastive"
    if "_baseline_" in name:
        return "baseline"
    if "_structure_contrastive_" in name:
        return "structure_contrastive"
    return ""


def _infer_base_dataset_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    for tag in ["_structure_contrastive_", "_structure_", "_contrastive_", "_baseline_"]:
        if tag in name:
            return name.split(tag, 1)[0]
    m = RUN_DIR_RE.match(name)
    return m.group("name") if m else name


def _resolve_adata_path(repo_root: Path, base_dataset: str) -> Path:
    data_dir = repo_root / "data"
    direct = data_dir / f"{base_dataset}.h5ad"
    if direct.exists():
        return direct

    candidates = list(data_dir.glob("*.h5ad"))
    base_lower = base_dataset.lower()
    for p in candidates:
        if p.stem.lower() == base_lower:
            return p
    raise FileNotFoundError(f"Could not find .h5ad for dataset='{base_dataset}' under {data_dir}")


def _find_cell_topic_file(run_dir: Path, expected_k: Optional[int]) -> Tuple[str, int, Path]:
    cell_topic_dir = run_dir / "cell_topic"
    if not cell_topic_dir.exists():
        raise FileNotFoundError(f"Missing cell_topic/ under: {run_dir}")

    matches: List[Tuple[str, int, Path]] = []
    for p in cell_topic_dir.glob("*_cell_topic_matrix_*.pkl"):
        m = CELL_TOPIC_RE.match(p.name)
        if not m:
            continue
        ds = m.group("dataset")
        k = int(m.group("k"))
        matches.append((ds, k, p))

    if not matches:
        raise FileNotFoundError(f"No cell_topic_matrix found under: {cell_topic_dir}")

    if expected_k is not None:
        for ds, k, p in matches:
            if k == expected_k:
                return ds, k, p
    return matches[0]


def _run_one(
    *,
    repo_root: Path,
    eval_mod,
    run_dir: Path,
    preferred_label_key: Optional[str],
    res_min: float,
    res_max: float,
    res_step: float,
    seed: int,
    write_json: bool,
) -> Dict[str, object]:
    expected_k = _infer_k_from_run_dir(run_dir)
    method = _infer_method_from_run_dir(run_dir)
    base_dataset = _infer_base_dataset_from_run_dir(run_dir)
    adata_path = _resolve_adata_path(repo_root, base_dataset)

    dataset_id, n_topics, cell_topic_file = _find_cell_topic_file(run_dir, expected_k)
    out_dir = (run_dir / "evaluation") if write_json else None

    cfg = eval_mod.EvalConfig(
        adata_path=str(adata_path),
        results_dir=str(run_dir),
        dataset=dataset_id,
        n_topics=n_topics,
        cell_topic_file=str(cell_topic_file),
        label_key=preferred_label_key,
        tag=run_dir.name,
        res_min=res_min,
        res_max=res_max,
        res_step=res_step,
        out_dir=str(out_dir) if out_dir is not None else None,
        seed=seed,
    )

    metrics = eval_mod.evaluate(cfg)

    row: Dict[str, object] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "base_dataset": base_dataset,
        "method": method,
        "expected_k": expected_k,
        "dataset_id": dataset_id,
        "n_topics": n_topics,
        "adata_path": str(adata_path),
    }
    row.update(metrics)
    row.update({f"cfg_{k}": v for k, v in asdict(cfg).items() if k not in {"adata_path", "results_dir"}})
    row["status"] = "ok"
    row["error"] = ""
    return row


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate all existing training runs under results/ and write a consolidated CSV."
    )
    p.add_argument("--results_root", type=str, default="results", help="Results root directory.")
    p.add_argument(
        "--out_csv",
        type=str,
        default="results/evaluation/all_runs_metrics.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--label_key",
        type=str,
        default=None,
        help="Preferred label key (auto-detected if not present).",
    )
    p.add_argument("--include_tuning", action="store_true", help="Also scan results/tuning/*.")
    p.add_argument("--res_min", type=float, default=0.0)
    p.add_argument("--res_max", type=float, default=2.0)
    p.add_argument("--res_step", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_json", action="store_true", help="Do not write per-run JSON metrics.")
    p.add_argument("--dry_run", action="store_true", help="Only list discovered runs and exit.")
    a = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_mod = _load_eval_module(repo_root)
    results_root = (repo_root / a.results_root).resolve()
    out_csv = (repo_root / a.out_csv).resolve()

    run_dirs = _iter_run_dirs(results_root, include_tuning=a.include_tuning)
    if not run_dirs:
        print(f"[eval_results_to_csv] No run dirs found under: {results_root}")
        return 0

    if a.dry_run:
        print(f"[eval_results_to_csv] Found {len(run_dirs)} run dirs under: {results_root}")
        for rd in run_dirs:
            print(f"- {rd}")
        return 0

    rows: List[Dict[str, object]] = []
    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"[eval_results_to_csv] ({i}/{len(run_dirs)}) evaluating: {run_dir.name}")
        try:
            row = _run_one(
                repo_root=repo_root,
                eval_mod=eval_mod,
                run_dir=run_dir,
                preferred_label_key=a.label_key,
                res_min=a.res_min,
                res_max=a.res_max,
                res_step=a.res_step,
                seed=a.seed,
                write_json=not a.no_json,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "base_dataset": _infer_base_dataset_from_run_dir(run_dir),
                "method": _infer_method_from_run_dir(run_dir),
                "expected_k": _infer_k_from_run_dir(run_dir),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[eval_results_to_csv] Wrote: {out_csv}")
    print("[eval_results_to_csv] Status counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
