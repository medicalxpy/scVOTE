import csv
import gzip
import importlib.util
import io
import tempfile
import unittest
from importlib.abc import Loader
from pathlib import Path
from types import ModuleType
from types import TracebackType
from typing import Protocol, cast, override
from unittest.mock import patch


class BuildOfficialGeneList(Protocol):
    def __call__(self, reactome_src: str | Path, gene_info_src: str | Path, dst: str | Path) -> None: ...


def load_build_official_gene_list() -> BuildOfficialGeneList:
    module = load_builder_module()
    module_path = Path(__file__).with_name("build_reactome_official_gene_list.py")
    return get_build_official_gene_list(module, module_path)


def load_builder_module() -> ModuleType:
    module_path = Path(__file__).with_name("build_reactome_official_gene_list.py")
    spec = importlib.util.spec_from_file_location("build_reactome_official_gene_list", module_path)
    if spec is None:
        raise ImportError(f"Could not create a module spec for {module_path}")

    loader = spec.loader
    if not isinstance(loader, Loader):
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def get_build_official_gene_list(module: ModuleType, module_path: Path) -> BuildOfficialGeneList:
    builder = module.__dict__.get("build_official_gene_list")
    if not callable(builder):
        raise AttributeError(f"build_official_gene_list is missing from {module_path}")
    return cast(BuildOfficialGeneList, builder)


class BuildReactomeOfficialGeneListTest(unittest.TestCase):
    def test_builds_official_gene_list_with_aggregated_human_pathway_sources(self) -> None:
        build_official_gene_list = load_build_official_gene_list()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reactome_path = tmpdir_path / "NCBI2Reactome.txt"
            gene_info_path = tmpdir_path / "Homo_sapiens.gene_info.gz"
            output_path = tmpdir_path / "reactome_human_official_genes.csv"

            _ = reactome_path.write_text(
                "\n".join(
                    [
                        "1\tR-HSA-2\thttps://example/2\tPathway B\tIEA\tHomo sapiens",
                        "1\tR-HSA-1\thttps://example/1\tPathway A\tTAS\tHomo sapiens",
                        "1\tR-HSA-1\thttps://example/1\tPathway A\tTAS\tHomo sapiens",
                        "2\tR-HSA-3\thttps://example/3\tPathway C\tTAS\tHomo sapiens",
                        "3\tR-MMU-4\thttps://example/4\tMouse Pathway\tTAS\tMus musculus",
                        "999\tR-HSA-5\thttps://example/5\tPathway D\tTAS\tHomo sapiens",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with gzip.open(gene_info_path, "wt", encoding="utf-8") as handle:
                _ = handle.write("#tax_id\tGeneID\tSymbol\n")
                _ = handle.write("9606\t1\tA1BG\n")
                _ = handle.write("9606\t2\tA2M\n")

            build_official_gene_list(reactome_path, gene_info_path, output_path)

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(
            rows,
            [
                ["ncbi_gene_id", "gene_symbol", "pathway_sources"],
                ["1", "A1BG", "R-HSA-1|Pathway A ; R-HSA-2|Pathway B"],
                ["2", "A2M", "R-HSA-3|Pathway C"],
                ["999", "", "R-HSA-5|Pathway D"],
            ],
        )

    def test_trims_trailing_whitespace_from_reactome_pathway_names(self) -> None:
        build_official_gene_list = load_build_official_gene_list()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reactome_path = tmpdir_path / "NCBI2Reactome.txt"
            gene_info_path = tmpdir_path / "Homo_sapiens.gene_info.gz"
            output_path = tmpdir_path / "reactome_human_official_genes.csv"

            _ = reactome_path.write_text(
                (
                    "1\tR-HSA-114608\thttps://example/114608\tPlatelet degranulation \tTAS\tHomo sapiens\n"
                    "1\tR-HSA-6798695\thttps://example/6798695\tNeutrophil degranulation\tTAS\tHomo sapiens\n"
                ),
                encoding="utf-8",
            )

            with gzip.open(gene_info_path, "wt", encoding="utf-8") as handle:
                _ = handle.write("#tax_id\tGeneID\tSymbol\n")
                _ = handle.write("9606\t1\tA1BG\n")

            build_official_gene_list(reactome_path, gene_info_path, output_path)

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(
            rows[1],
            [
                "1",
                "A1BG",
                "R-HSA-114608|Platelet degranulation ; R-HSA-6798695|Neutrophil degranulation",
            ],
        )

    def test_builds_official_gene_list_from_url_sources_without_full_response_buffering(self) -> None:
        module = load_builder_module()
        build_official_gene_list = get_build_official_gene_list(
            module,
            Path(__file__).with_name("build_reactome_official_gene_list.py"),
        )

        reactome_bytes = (
            "1\tR-HSA-2\thttps://example/2\tPathway B\tIEA\tHomo sapiens\n"
            "1\tR-HSA-1\thttps://example/1\tPathway A\tTAS\tHomo sapiens\n"
            "1\tR-HSA-1\thttps://example/1\tPathway A\tTAS\tHomo sapiens\n"
            "2\tR-HSA-3\thttps://example/3\tPathway C\tTAS\tHomo sapiens\n"
            "3\tR-MMU-4\thttps://example/4\tMouse Pathway\tTAS\tMus musculus\n"
            "999\tR-HSA-5\thttps://example/5\tPathway D\tTAS\tHomo sapiens\n"
        ).encode("utf-8")

        gene_info_buffer = io.BytesIO()
        with gzip.open(gene_info_buffer, "wt", encoding="utf-8") as handle:
            _ = handle.write("#tax_id\tGeneID\tSymbol\n")
            _ = handle.write("9606\t1\tA1BG\n")
            _ = handle.write("9606\t2\tA2M\n")
        gene_info_bytes = gene_info_buffer.getvalue()

        class FakeStreamingResponse(io.BytesIO):
            @override
            def __enter__(self) -> "FakeStreamingResponse":
                return cast("FakeStreamingResponse", super().__enter__())

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                _ = super().__exit__(exc_type, exc, tb)

            @override
            def read(self, size: int | None = -1) -> bytes:
                if size == -1:
                    raise AssertionError("response.read() without a size would buffer the full response")
                return super().read(size)

        url_payloads = {
            "https://example/reactome.txt": reactome_bytes,
            "https://example/Homo_sapiens.gene_info.gz": gene_info_bytes,
        }

        def fake_urlopen(url: str) -> FakeStreamingResponse:
            return FakeStreamingResponse(url_payloads[url])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "reactome_human_official_genes.csv"

            with patch.object(module, "urlopen", side_effect=fake_urlopen):
                build_official_gene_list(
                    "https://example/reactome.txt",
                    "https://example/Homo_sapiens.gene_info.gz",
                    output_path,
                )

            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(
            rows,
            [
                ["ncbi_gene_id", "gene_symbol", "pathway_sources"],
                ["1", "A1BG", "R-HSA-1|Pathway A ; R-HSA-2|Pathway B"],
                ["2", "A2M", "R-HSA-3|Pathway C"],
                ["999", "", "R-HSA-5|Pathway D"],
            ],
        )


if __name__ == "__main__":
    _ = unittest.main()
