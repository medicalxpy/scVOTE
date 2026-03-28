#!/usr/bin/env python3
"""
Extract unique gene symbols from the local Reactome pathway tree.

Reads data/reactome_pathway_tree_flat.csv, splits the semicolon-delimited
`genes` column, normalizes tokens (trim + uppercase), deduplicates, and
writes data/gene_sets/reactome_human_unique_genes.csv with a single
`gene_symbol` column.

This is the local filtered Reactome set and not a strict HGNC-only list.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_gene_universe(src: Path, dst: Path) -> None:
    genes: set[str] = set()
    total_rows = 0
    nonempty_rows = 0

    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total_rows += 1
            raw = row.get("genes", "").strip()
            if not raw:
                continue
            nonempty_rows += 1
            for token in raw.split(";"):
                token = token.strip().upper()
                if token:
                    genes.add(token)

    sorted_genes = sorted(genes)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene_symbol"])
        for gene in sorted_genes:
            writer.writerow([gene])

    print(f"Source: {src}")
    print(f"Total rows: {total_rows}")
    print(f"Non-empty genes: {nonempty_rows}")
    print(f"Unique tokens: {len(sorted_genes)}")
    print(f"Output: {dst}")
    print(f"Preview (first 5): {sorted_genes[:5]}")
    print(f"Preview (last 5): {sorted_genes[-5:]}")


def parse_args() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Build Reactome unique-gene CSV from local pathway tree."
    )
    _ = parser.add_argument(
        "--src",
        type=Path,
        default=repo / "data" / "reactome_pathway_tree_flat.csv",
        help="Source pathway-tree CSV (default: data/reactome_pathway_tree_flat.csv)",
    )
    _ = parser.add_argument(
        "--dst",
        type=Path,
        default=repo / "data" / "gene_sets" / "reactome_human_unique_genes.csv",
        help="Output unique-gene CSV (default: data/gene_sets/reactome_human_unique_genes.csv)",
    )
    parsed = parser.parse_args()

    src_obj: object = getattr(parsed, "src", None)
    dst_obj: object = getattr(parsed, "dst", None)
    if not isinstance(src_obj, Path) or not isinstance(dst_obj, Path):
        raise TypeError("Expected argparse to produce Path values for --src and --dst")
    return src_obj, dst_obj


def main() -> None:
    src, dst = parse_args()

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    build_gene_universe(src, dst)


if __name__ == "__main__":
    _ = main()
