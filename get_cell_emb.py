#!/usr/bin/env python3
"""
scFASTopic Cell Embedding Extraction Script

This version computes cell embeddings with scvi-tools. It filters and
optionally subsamples the input AnnData object, trains an SCVI model, and
stores the latent representation as a pickle file.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("medium")

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    input_data: str
    dataset_name: str = "dataset"
    output_dir: str = "results/cell_embedding"
    max_cells: Optional[int] = None
    n_top_genes: Optional[int] = 0
    # Gene-level preprocessing (keep in sync with train_fastopic.py)
    filter_genept: bool = True
    gene_list_path: Optional[str] = "data/gene_list/C2_C5_GO.csv"
    batch_key: Optional[str] = None
    labels_key: Optional[str] = None
    n_latent: int = 30
    n_hidden: int = 256
    n_layers: int = 2
    dropout_rate: float = 0.1
    gene_likelihood: str = "zinb"
    learning_rate: float = 1e-3
    max_epochs: int = 1000
    batch_size: int = 2048
    early_stopping: bool = True
    early_stopping_patience: int = 20
    check_val_every_n_epoch: int = 1
    seed: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def _copy_counts(matrix):
    if sp is not None and sp.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _load_genept_genes() -> Optional[Set[str]]:
    """Load GenePT gene set (keys of the embedding dict)."""
    try:
        genept_path = "GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle"
        import pickle

        with open(genept_path, "rb") as f:
            genept_dict = pickle.load(f)
        return set(genept_dict.keys())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load GenePT gene list: %s", e)
        return None


def _load_gene_list(gene_list_path: str, verbose: bool = False) -> Optional[Set[str]]:
    """Load a gene list from CSV and return as a set of symbols."""
    try:
        df = pd.read_csv(gene_list_path)
    except FileNotFoundError:
        if verbose:
            logger.warning("Gene list file not found: %s", gene_list_path)
        return None
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.warning("Could not load gene list from %s: %s", gene_list_path, e)
        return None

    if df.empty:
        if verbose:
            logger.warning("Gene list CSV is empty: %s", gene_list_path)
        return None

    if "gene_symbol" in df.columns:
        series = df["gene_symbol"]
    else:
        series = df.iloc[:, 0]

    genes = {str(g).strip() for g in series.dropna().astype(str)}
    if verbose:
        logger.info("Loaded %d genes from gene list: %s", len(genes), gene_list_path)
    return genes


def preprocess_for_scvi(adata: ad.AnnData, config: EmbeddingConfig) -> ad.AnnData:
    if config.verbose:
        logger.info("Raw data: %d cells × %d genes", adata.n_obs, adata.n_vars)

    adata = adata.copy()

    # Gene list filtering (e.g., C2_C5_GO), to keep genes
    # consistent with train_fastopic preprocessing.
    if config.gene_list_path:
        gene_list = _load_gene_list(config.gene_list_path, verbose=config.verbose)
        if gene_list:
            current_genes = [str(g) for g in adata.var_names]
            mask = [g in gene_list for g in current_genes]
            n_keep = sum(mask)
            if n_keep > 0:
                adata = adata[:, mask].copy()
                if config.verbose:
                    logger.info(
                        "Gene-list filtering: kept %d/%d genes from %s",
                        n_keep,
                        len(current_genes),
                        config.gene_list_path,
                    )
            elif config.verbose:
                logger.warning(
                    "Gene-list filtering skipped: no overlap between adata genes and %s",
                    config.gene_list_path,
                )

    # Basic QC filtering on X / obs / var
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # GenePT gene filtering (same logic as train_fastopic.py)
    if config.filter_genept:
        genept_genes = _load_genept_genes()
        if genept_genes is not None:
            current_genes = [str(g) for g in adata.var_names]
            mask = [g in genept_genes for g in current_genes]
            n_keep = sum(mask)
            if n_keep > 0:
                adata = adata[:, mask].copy()
                if config.verbose:
                    logger.info(
                        "GenePT filtering: kept %d/%d genes shared with GenePT",
                        n_keep,
                        len(current_genes),
                    )
            elif config.verbose:
                logger.warning("No genes shared with GenePT; skip GenePT filtering")

    if config.max_cells is not None and adata.n_obs > config.max_cells:
        rng = np.random.default_rng(config.seed)
        subset = rng.choice(adata.n_obs, size=config.max_cells, replace=False)
        adata = adata[subset].copy()
        if config.verbose:
            logger.info("Cells after subsampling: %d", adata.n_obs)

    # Derive a raw-counts layer aligned with the *current* adata shape.
    # We assume that adata.X already contains raw counts (for all datasets
    # we've normalised the files so that X holds counts). We still honour
    # an existing layers["counts"] if present.
    if "counts" in adata.layers:
        base_counts = adata.layers["counts"]
    else:
        base_counts = adata.X

    counts = _copy_counts(base_counts)

    # Ensure layers["counts"] matches (n_obs, n_vars)
    if counts.shape != adata.shape:
        raise ValueError(
            f"Counts matrix shape {counts.shape} does not match "
            f"AnnData shape {adata.shape} after filtering."
        )

    adata.layers["counts"] = counts
    adata.X = counts.copy()

    if config.n_top_genes:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.n_top_genes,
            flavor="seurat_v3",
            layer="counts",
        )
        adata = adata[:, adata.var.highly_variable].copy()
        if config.verbose:
            logger.info("Selected %d highly variable genes", adata.n_vars)

    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=1)

    if config.verbose:
        logger.info("Post-preprocessing: %d cells × %d genes", adata.n_obs, adata.n_vars)

    return adata


def train_scvi_model(adata: ad.AnnData, config: EmbeddingConfig) -> scvi.model.SCVI:
    setup_kwargs = {"layer": "counts"}
    if config.batch_key:
        setup_kwargs["batch_key"] = config.batch_key
    if config.labels_key:
        setup_kwargs["labels_key"] = config.labels_key

    scvi.settings.seed = config.seed
    scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)

    model = scvi.model.SCVI(
        adata,
        n_latent=config.n_latent,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout_rate=config.dropout_rate,
        gene_likelihood=config.gene_likelihood,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "auto"
    train_kwargs = {
        "max_epochs": config.max_epochs,
        "batch_size": config.batch_size,
        "plan_kwargs": {"lr": config.learning_rate},
        "accelerator": accelerator,
        "devices": "auto",
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "early_stopping": config.early_stopping,
        "datasplitter_kwargs": {"num_workers": 4},
    }
    if config.early_stopping:
        train_kwargs["early_stopping_patience"] = config.early_stopping_patience

    if config.verbose:
        logger.info(
            "Training SCVI model (epochs=%d, latent=%d, batch_size=%d)...",
            config.max_epochs,
            config.n_latent,
            config.batch_size,
        )

    model.train(**train_kwargs)
    return model


def extract_scvi_embeddings(adata: ad.AnnData, config: EmbeddingConfig) -> np.ndarray:
    model = train_scvi_model(adata, config)
    latent = model.get_latent_representation()
    return latent.astype(np.float32)


def save_embeddings(cell_embeddings: np.ndarray, config: EmbeddingConfig) -> str:
    filename = f"{config.dataset_name}_scvi.pkl"
    path = Path(config.output_dir) / filename
    with path.open("wb") as f:
        import pickle

        pickle.dump(cell_embeddings, f)

    if config.verbose:
        logger.info("Cell embeddings saved to: %s", path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="scFASTopic cell embeddings via scVI")
    parser.add_argument("--input_data", required=True, help="Input .h5ad file path")
    parser.add_argument("--dataset_name", default="dataset", help="Dataset name")
    parser.add_argument("--output_dir", default="results/cell_embedding", help="Output directory")
    parser.add_argument("--max_cells", type=int, help="Max number of cells")
    parser.add_argument("--n_top_genes", type=int, default=0, help="Number of highly variable genes (0 to disable)")
    parser.add_argument(
        "--gene_list_path",
        type=str,
        default="data/gene_list/C2_C5_GO.csv",
        help="CSV file with gene list to keep (same as train_fastopic, default: data/gene_list/C2_C5_GO.csv)",
    )
    parser.add_argument(
        "--no_genept_filter",
        action="store_true",
        help="Disable GenePT gene filtering during embedding preprocessing",
    )
    parser.add_argument("--batch_key", help="Batch column in AnnData")
    parser.add_argument("--labels_key", help="Labels column in AnnData")
    parser.add_argument("--n_latent", type=int, default=30, help="Latent dimension for scVI")
    parser.add_argument("--n_hidden", type=int, default=128, help="Hidden layer size for scVI")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers for scVI")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout for scVI")
    parser.add_argument("--gene_likelihood", default="zinb", help="Gene likelihood for scVI")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Training learning rate")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Training batch size")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validation check frequency")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = EmbeddingConfig(
        input_data=args.input_data,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_cells=args.max_cells,
        n_top_genes=args.n_top_genes,
         # Gene-level preprocessing
        filter_genept=not args.no_genept_filter,
        gene_list_path=args.gene_list_path,
        batch_key=args.batch_key,
        labels_key=args.labels_key,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout_rate=args.dropout_rate,
        gene_likelihood=args.gene_likelihood,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        seed=args.seed,
        verbose=args.verbose,
    )

    if config.verbose:
        logger.info("Loading data: %s", config.input_data)
    adata = sc.read_h5ad(config.input_data)

    adata_preprocessed = preprocess_for_scvi(adata, config)

    cell_embeddings = extract_scvi_embeddings(adata_preprocessed, config)

    saved_file = save_embeddings(cell_embeddings, config)

    logger.info("=== Cell embedding extraction completed ===")
    logger.info("Dataset: %s", config.dataset_name)
    logger.info("Cells: %d", cell_embeddings.shape[0])
    logger.info("Embedding dim: %d", cell_embeddings.shape[1])
    logger.info("Method: scvi")
    logger.info("Saved to: %s", saved_file)


if __name__ == "__main__":
    main()
