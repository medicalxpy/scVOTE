import argparse
import csv
import gzip
import io
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPResponse
from pathlib import Path
from typing import TextIO, cast
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_REACTOME_URL = "https://reactome.org/download/current/NCBI2Reactome.txt"
DEFAULT_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data/gene_sets/reactome_human_official_genes.csv"
CSV_HEADER = ["ncbi_gene_id", "gene_symbol", "pathway_sources"]


@contextmanager
def open_text_stream(src: str | Path) -> Iterator[TextIO]:
    source = str(src)
    parsed = urlparse(source)
    is_url = parsed.scheme in {"http", "https"}
    is_gzip = source.endswith(".gz")

    if is_url:
        with cast(HTTPResponse, urlopen(source)) as response:
            if is_gzip:
                with gzip.open(response, "rt", encoding="utf-8") as handle:
                    yield handle
            else:
                with io.TextIOWrapper(response, encoding="utf-8") as handle:
                    yield handle
        return

    path = Path(source)
    if is_gzip:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
        return

    with path.open("r", encoding="utf-8") as handle:
        yield handle


def load_human_reactome_pathway_sources(reactome_src: str | Path) -> dict[str, list[str]]:
    gene_to_sources: dict[str, set[str]] = defaultdict(set)

    with open_text_stream(reactome_src) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue

            gene_id, pathway_id, _url, pathway_name, _evidence, species = (field.strip() for field in row[:6])
            if species != "Homo sapiens":
                continue

            gene_to_sources[gene_id].add(f"{pathway_id}|{pathway_name}")

    return {gene_id: sorted(pathway_sources) for gene_id, pathway_sources in gene_to_sources.items()}


def load_gene_symbols(gene_info_src: str | Path) -> dict[str, str]:
    gene_symbols: dict[str, str] = {}

    with open_text_stream(gene_info_src) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue

            tax_id, gene_id, symbol = row[:3]
            if tax_id != "9606":
                continue

            gene_symbols[gene_id] = symbol

    return gene_symbols


def gene_sort_key(gene_id: str) -> tuple[int, int | str]:
    if gene_id.isdigit():
        return (0, int(gene_id))
    return (1, gene_id)


def build_official_gene_list(reactome_src: str | Path, gene_info_src: str | Path, dst: str | Path) -> None:
    gene_to_sources = load_human_reactome_pathway_sources(reactome_src)
    gene_symbols = load_gene_symbols(gene_info_src)

    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)

        for gene_id in sorted(gene_to_sources, key=gene_sort_key):
            writer.writerow(
                [
                    gene_id,
                    gene_symbols.get(gene_id, ""),
                    " ; ".join(gene_to_sources[gene_id]),
                ]
            )


class Args(argparse.Namespace):
    reactome_src: str
    gene_info_src: str
    dst: str

    def __init__(self) -> None:
        super().__init__()
        self.reactome_src = DEFAULT_REACTOME_URL
        self.gene_info_src = DEFAULT_GENE_INFO_URL
        self.dst = str(DEFAULT_OUTPUT_PATH)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Build the official Reactome human gene list CSV.")
    _ = parser.add_argument("--reactome-src", default=DEFAULT_REACTOME_URL)
    _ = parser.add_argument("--gene-info-src", default=DEFAULT_GENE_INFO_URL)
    _ = parser.add_argument("--dst", default=str(DEFAULT_OUTPUT_PATH))
    args = Args()
    _ = parser.parse_args(namespace=args)
    return args


def main() -> None:
    args = parse_args()
    build_official_gene_list(args.reactome_src, args.gene_info_src, args.dst)


if __name__ == "__main__":
    _ = main()
