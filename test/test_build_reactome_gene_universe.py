import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Callable, cast


REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "test" / "build_reactome_gene_universe.py"


def load_builder() -> Callable[[Path, Path], None]:
    if not MODULE_PATH.exists():
        raise FileNotFoundError(f"Missing module under test: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("build_reactome_gene_universe", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {MODULE_PATH}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_gene_universe", None)
    if not callable(builder):
        raise AttributeError("build_gene_universe is missing or not callable")
    return cast(Callable[[Path, Path], None], builder)


class BuildReactomeGeneUniverseTest(unittest.TestCase):
    def test_build_gene_universe_writes_uppercase_deduplicated_csv(self):
        build_gene_universe = load_builder()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "input.csv"
            dst = tmp / "output.csv"

            _ = src.write_text(
                (
                    "pathway_id,pathway_name,min_depth,parent_ids,child_ids,node_role,genes\n"
                    + "R-HSA-1,Pathway A,0,ROOT,,ROOT,TP53;brca1; TP53 ;\n"
                    + "R-HSA-2,Pathway B,1,R-HSA-1,,LEAF,BRAF;EGFR\n"
                    + "R-HSA-3,Pathway C,1,R-HSA-1,,LEAF,\n"
                ),
                encoding="utf-8",
            )

            build_gene_universe(src, dst)

            with open(dst, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))

        self.assertEqual(rows[0], ["gene_symbol"])
        self.assertEqual(rows[1:], [["BRAF"], ["BRCA1"], ["EGFR"], ["TP53"]])


if __name__ == "__main__":
    _ = unittest.main()
