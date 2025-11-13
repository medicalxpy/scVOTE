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
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
import scvi
import torch

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
    n_top_genes: Optional[int] = 2000
    batch_key: Optional[str] = None
    labels_key: Optional[str] = None
    n_latent: int = 30
    n_hidden: int = 256
    n_layers: int = 2
    dropout_rate: float = 0.1
    gene_likelihood: str = "zinb"
    learning_rate: float = 1e-3
    max_epochs: int = 1000
    batch_size: int = 256
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


def preprocess_for_scvi(adata: ad.AnnData, config: EmbeddingConfig) -> ad.AnnData:
    if config.verbose:
        logger.info("Raw data: %d cells × %d genes", adata.n_obs, adata.n_vars)

    adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    if config.max_cells is not None and adata.n_obs > config.max_cells:
        rng = np.random.default_rng(config.seed)
        subset = rng.choice(adata.n_obs, size=config.max_cells, replace=False)
        adata = adata[subset].copy()
        if config.verbose:
            logger.info("Cells after subsampling: %d", adata.n_obs)

    if "counts" in adata.layers:
        counts = _copy_counts(adata.layers["counts"])
    elif adata.raw is not None:
        counts = _copy_counts(adata.raw.X)
    else:
        counts = _copy_counts(adata.X)

    if sp is not None and sp.issparse(counts):
        adata.layers["counts"] = counts
        adata.X = counts.copy()
    else:
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
    parser.add_argument("--output_dir", default="scFastopic/results/cell_embedding", help="Output directory")
    parser.add_argument("--max_cells", type=int, help="Max number of cells")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--batch_key", help="Batch column in AnnData")
    parser.add_argument("--labels_key", help="Labels column in AnnData")
    parser.add_argument("--n_latent", type=int, default=3072, help="Latent dimension for scVI")
    parser.add_argument("--n_hidden", type=int, default=128, help="Hidden layer size for scVI")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers for scVI")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout for scVI")
    parser.add_argument("--gene_likelihood", default="zinb", help="Gene likelihood for scVI")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Training learning rate")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
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
