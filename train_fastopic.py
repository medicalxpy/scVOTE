#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
import pickle
import time
from typing import Optional, List, Dict, Any, Set
import warnings

warnings.filterwarnings('ignore')

import scanpy as sc


def save_matrices(matrices, dataset_name, n_topics, output_dir):
    """Save matrices into their designated subdirectories."""
    base_output_dir = Path(output_dir)
    
    # Map matrix types to subdirectories
    matrix_subdirs = {
        'cell_topic_matrix': 'cell_topic',
        'topic_gene_matrix': 'topic_gene', 
        'gene_embeddings': 'gene_embedding',
        'topic_embeddings': 'topic_embedding',
        'gene_names': 'gene_embedding',
    }
    
    saved_files = []
    for matrix_name, matrix in matrices.items():
        # Create subdirectory
        subdir = matrix_subdirs.get(matrix_name, matrix_name)
        matrix_output_dir = base_output_dir / subdir
        matrix_output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{dataset_name}_{matrix_name}_{n_topics}.pkl"
        filepath = matrix_output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(matrix, f)
        
        saved_files.append(str(filepath))
        print(f"💾 Saved {matrix_name}: {filepath}")
    
    return saved_files
def validate_matrices(matrices):
    """Validate matrix shapes and content.

    Allows non-array artifacts like `gene_names` (list[str]). Only checks
    `.size` for NumPy arrays/torch tensors.
    """
    try:
        for name, matrix in matrices.items():
            if matrix is None:
                print(f"⚠️ Warning: {name} is None")
                return False
            # Special-case: gene_names is a list of strings
            if name == 'gene_names':
                if not isinstance(matrix, (list, tuple)):
                    print(f"⚠️ Warning: gene_names should be a list/tuple, got {type(matrix)}")
                    return False
                if len(matrix) == 0:
                    print("⚠️ Warning: gene_names is empty")
                    return False
                continue
            # NumPy arrays or torch tensors
            try:
                size = matrix.size  # numpy array / torch tensor
            except Exception:
                # Fallback: try to convert to numpy array for size check
                try:
                    arr = np.asarray(matrix)
                    size = arr.size
                except Exception as _:
                    print(f"⚠️ Warning: {name} has unsupported type {type(matrix)}")
                    return False
            if size == 0:
                print(f"⚠️ Warning: {name} is empty")
                return False
        return True
    except Exception as e:
        print(f"❌ Matrix validation error: {e}")
        return False
from dataclasses import dataclass


@dataclass
class FastopicConfig:
    embedding_file: Optional[str] = None
    adata_path: Optional[str] = None
    dataset: str = "PBMC"
    output_dir: str = "results"
    n_topics: int = 20
    epochs: int = 100
    learning_rate: float = 0.01
    DT_alpha: float = 1.0
    TW_alpha: float = 1.0
    theta_temp: float = 2.0
    verbose: bool = True
    seed: int = 42
    filter_genept: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    # Structural alignment (Laplacian + CKA)
    align_enable: bool = True
    align_alpha: float = 1e-3
    align_beta: float = 1e-3
    align_knn_k: int = 48
    align_cka_sample_n: int = 2048
    align_max_kernel_genes: int = 4096
    # Legacy GenePT contrastive loss weight
    genept_loss_weight: float = 0.0
    # HVG selection for single training (0 disables)
    n_top_genes: int = 0
    # Optional gene list filter for single training
    gene_list_path: Optional[str] = "data/gene_list/C2_C5_GO.csv"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train scFASTopic with pre-extracted cell embeddings')
    
    # Input files
    parser.add_argument('--embedding_file', type=str, required=True,
                       help='Path to cell embeddings pkl file')
    parser.add_argument('--adata_path', type=str, required=True,
                       help='Path to original adata file (.h5ad)')
    parser.add_argument('--dataset', type=str, default='PBMC',
                       help='Dataset name')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    # Model options
    parser.add_argument('--n_topics', type=int, default=20,
                       help='Number of topics')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    
    # FASTopic hyperparameters
    parser.add_argument('--DT_alpha', type=float, default=1.0,
                       help='Dirichlet-tree alpha parameter')
    parser.add_argument('--TW_alpha', type=float, default=1.0,
                       help='Topic-word alpha parameter')
    parser.add_argument('--theta_temp', type=float, default=2.0,
                       help='Temperature parameter')
    # HVG selection (0 disables; apply during single training preprocessing)
    parser.add_argument('--n_top_genes', type=int, default=0,
                       help='Select top-N HVGs for training (0 to disable)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--no_genept_filter', action='store_true',
                       help='Disable GenePT gene filtering')
    parser.add_argument(
        '--gene_list_path',
        type=str,
        default='data/gene_list/C2_C5_GO.csv',
        help='CSV file with gene list to keep (default: data/gene_list/C2_C5_GO.csv)',
    )
    
    # Structural alignment options
    parser.add_argument('--no_align', action='store_true',
                       help='Disable structural alignment (Laplacian + CKA)')
    parser.add_argument(
        '--structure',
        action='store_true',
        help='Enable structural (Laplacian + CKA) alignment explicitly',
    )
    parser.add_argument('--align_alpha', type=float, default=1e-3,
                       help='Weight for Laplacian alignment loss')
    parser.add_argument('--align_beta', type=float, default=1e-3,
                       help='Weight for CKA alignment loss')
    parser.add_argument('--align_knn_k', type=int, default=48,
                       help='k for cosine kNN graph on reference embeddings')
    parser.add_argument('--align_cka_sample_n', type=int, default=2048,
                       help='Subsample size for CKA computation')
    parser.add_argument('--align_max_kernel_genes', type=int, default=4096,
                       help='Cap for kernel template size to control memory')
    
    # Legacy GenePT contrastive alignment weight (kept for compatibility; default 0)
    parser.add_argument('--genept_loss_weight', type=float, default=0.0,
                       help='Weight for legacy GenePT contrastive alignment loss')
    parser.add_argument(
        '--contrastive',
        action='store_true',
        help='Enable GenePT contrastive alignment loss (uses a default weight if not set)',
    )
    
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> FastopicConfig:
    # Structural alignment: base flag from --no_align, allow overriding via --structure
    align_enable = not args.no_align
    if getattr(args, "structure", False):
        align_enable = True

    # GenePT contrastive alignment: base weight from --genept_loss_weight,
    # optionally enable via --contrastive with a small default weight.
    genept_loss_weight = args.genept_loss_weight
    if getattr(args, "contrastive", False) and genept_loss_weight <= 0.0:
        genept_loss_weight = 1e-3

    return FastopicConfig(
        embedding_file=args.embedding_file,
        adata_path=args.adata_path,
        dataset=args.dataset,
        output_dir=args.output_dir,
        n_topics=args.n_topics,
        epochs=args.epochs,
        learning_rate=args.lr,
        DT_alpha=args.DT_alpha,
        TW_alpha=args.TW_alpha,
        theta_temp=args.theta_temp,
        verbose=not args.quiet,
        seed=args.seed,
        filter_genept=not args.no_genept_filter,
        patience=args.patience,
        align_enable=align_enable,
        align_alpha=args.align_alpha,
        align_beta=args.align_beta,
        align_knn_k=args.align_knn_k,
        align_cka_sample_n=args.align_cka_sample_n,
        align_max_kernel_genes=args.align_max_kernel_genes,
        genept_loss_weight=genept_loss_weight,
        n_top_genes=args.n_top_genes,
        gene_list_path=args.gene_list_path,
    )


def load_genept_genes():
    """Load GenePT gene set."""
    try:
        genept_path = 'GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
        with open(genept_path, 'rb') as f:
            genept_dict = pickle.load(f)
        return set(genept_dict.keys())
    except Exception as e:
        print(f"⚠️ Could not load GenePT gene list: {e}")
        return None


def load_gene_list(
    gene_list_path: str,
    verbose: bool = False,
) -> Optional[Set[str]]:
    """Load a gene list from CSV and return as a set of symbols.

    The CSV is expected to contain either a column named ``gene_symbol`` or
    a single unnamed column with gene symbols.
    """
    try:
        df = pd.read_csv(gene_list_path)
    except FileNotFoundError:
        if verbose:
            print(f"⚠️ Gene list file not found: {gene_list_path}")
        return None
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not load gene list from {gene_list_path}: {e}")
        return None

    if df.empty:
        if verbose:
            print(f"⚠️ Gene list CSV is empty: {gene_list_path}")
        return None

    if "gene_symbol" in df.columns:
        series = df["gene_symbol"]
    else:
        series = df.iloc[:, 0]

    genes = {str(g).strip() for g in series.dropna().astype(str)}
    if verbose:
        print(f"🧬 Loaded {len(genes)} genes from gene list: {gene_list_path}")
    return genes


def preprocess_adata(
    adata_path: str,
    verbose: bool = False,
    filter_genept: bool = True,
    n_top_genes: int = 0,
    gene_list_path: Optional[str] = None,
):
    """
    Extract counts from adata and preprocess.

    Args:
        adata_path: Path to single-cell data (.h5ad).
        verbose: Whether to print details.
        filter_genept: Whether to filter to genes shared with GenePT.
        n_top_genes: Number of HVGs to keep (0 disables HVG filter).
        gene_list_path: Optional CSV path; if provided, restrict genes to
            those present in the gene list (intersection with adata.var_names).

    Returns:
        expression_matrix: Preprocessed expression matrix (cells x genes).
        gene_names: List of gene names.
    """
    if verbose:
        print(f"📁 Loading adata: {adata_path}")
    
    # Load data
    adata = sc.read_h5ad(adata_path)

    # Gene list filtering (e.g., C2_C5_GO)
    if gene_list_path:
        gene_list = load_gene_list(gene_list_path, verbose=verbose)
        if gene_list:
            current_genes = [str(g) for g in adata.var_names]
            mask = [g in gene_list for g in current_genes]
            n_keep = sum(mask)
            if n_keep > 0:
                adata = adata[:, mask].copy()
                if verbose:
                    print(
                        f"🧬 Gene-list filtering: kept {n_keep}/"
                        f"{len(current_genes)} genes from {gene_list_path}"
                    )
            elif verbose:
                print(
                    f"⚠️ Gene-list filtering skipped: no overlap between "
                    f"adata genes and list {gene_list_path}"
                )
    
    if verbose:
        print(f"Original shape: {adata.shape}")
    
    # Simple filtering
    # Filter low-quality cells (n_genes < 200)
    sc.pp.filter_cells(adata, min_genes=200)
    
    # Filter lowly expressed genes (min_cells >= 3)
    sc.pp.filter_genes(adata, min_cells=3)
    
    if verbose:
        print(f"After filtering: {adata.shape}")
    
    # GenePT gene filtering
    if filter_genept:
        genept_genes = load_genept_genes()
        if genept_genes is not None:
            # Find genes shared with GenePT
            current_genes = [str(g) for g in adata.var_names]
            mask = [g in genept_genes for g in current_genes]
            n_keep = sum(mask)

            if n_keep > 0:
                adata = adata[:, mask].copy()
                if verbose:
                    print(f"🧬 GenePT filtering: kept {n_keep}/{len(current_genes)} genes")
            elif verbose:
                print("⚠️ No genes shared with GenePT; skip filtering")
    
    # Optional HVG selection (default disabled)
    if n_top_genes and n_top_genes > 0:
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                flavor="seurat_v3",
            )
            adata = adata[:, adata.var["highly_variable"]].copy()
            if verbose:
                print(f"🔎 HVG selection: kept top {adata.n_vars} genes")
        except Exception as e:
            if verbose:
                print(f"⚠️ HVG selection failed ({e}); proceeding without HVG filter")

    if verbose:
        print(f"Final shape: {adata.shape}")
    
    # Normalize total counts to 1 per cell
    sc.pp.normalize_total(adata, target_sum=1)
    
    # log1p transform
    sc.pp.log1p(adata)
    
    # Get processed matrix (keep sparse to avoid OOM on large datasets)
    try:
        if sp.issparse(adata.X):
            expression_matrix = adata.X.copy()
        else:
            expression_matrix = adata.X
    except Exception:
        # Fallback: best-effort dense conversion
        expression_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    
    gene_names = adata.var_names.tolist()
    
    if verbose:
        print(f"✅ Preprocessing complete: {expression_matrix.shape}")
        print(f"✅ Gene count: {len(gene_names)}")
    
    return expression_matrix, gene_names


def load_embeddings_and_expression(
    embedding_file: str,
    adata_path: str,
    verbose: bool = False,
    filter_genept: bool = True,
    n_top_genes: int = 0,
    gene_list_path: Optional[str] = None,
):
    """
    Load cell embeddings and the preprocessed expression matrix.

    Args:
        embedding_file: Path to cell embeddings (.pkl).
        adata_path: Path to original adata (.h5ad).
        verbose: Whether to print details.
        filter_genept: Whether to filter to genes shared with GenePT.
        n_top_genes: Number of HVGs to keep (0 disables HVG filter).
        gene_list_path: Optional CSV path; if provided, restrict genes to
            those present in the gene list.

    Returns:
        cell_embeddings: Array of cell embeddings.
        expression_matrix: Preprocessed expression matrix.
        gene_names: List of gene names.
    """
    if verbose:
        print("📥 Loading embeddings and preprocessing expression data")
        print("="*60)
    
    # Load cell embeddings
    if verbose:
        print(f"📁 Loading cell embeddings: {embedding_file}")
    
    with open(embedding_file, 'rb') as f:
        cell_embeddings = pickle.load(f)
    
    if verbose:
        print(f"✅ Cell embeddings: {cell_embeddings.shape}")
    
    # Preprocess adata
    expression_matrix, gene_names = preprocess_adata(
        adata_path=adata_path,
        verbose=verbose,
        filter_genept=filter_genept,
        n_top_genes=n_top_genes,
        gene_list_path=gene_list_path,
    )
    
    # Ensure matching cell counts
    n_cells_emb = cell_embeddings.shape[0]
    n_cells_exp = expression_matrix.shape[0]
    
    if n_cells_emb != n_cells_exp:
        min_cells = min(n_cells_emb, n_cells_exp)
        if verbose:
            print(f"⚠️ Cell count mismatch (embedding: {n_cells_emb}, expression: {n_cells_exp})")
            print(f"Using first {min_cells} cells")
        
        cell_embeddings = cell_embeddings[:min_cells]
        expression_matrix = expression_matrix[:min_cells]
    
    return cell_embeddings, expression_matrix, gene_names



def train_fastopic_model(
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    gene_names: List[str],
    config: FastopicConfig,
    verbose: bool = False,
):
    """
    Train the scFASTopic model.

    Args:
        cell_embeddings: Cell embeddings array.
        expression_matrix: Preprocessed expression matrix.
        gene_names: List of gene names.
        config: Training/config parameters.
        verbose: Verbose output flag.

    Returns:
        results: Training result dictionary.
        training_time: Elapsed training time (seconds).
    """
    if verbose:
        print("\n🤖 Training scFASTopic model")
        print("="*60)

    # Reset GPU peak memory stats so we can measure usage for this run.
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            # If reset is not supported, we just skip GPU memory tracking.
            if verbose:
                print("⚠️ Could not reset CUDA peak memory stats; GPU usage tracking disabled.")
    
    # Use the FASTopic implementation
    from fastopic import FASTopic
    
    model = FASTopic(
        num_topics=config.n_topics,
        device="cuda" if torch.cuda.is_available() else "cpu",
        DT_alpha=config.DT_alpha,
        TW_alpha=config.TW_alpha,
        theta_temp=config.theta_temp,
        align_enable=config.align_enable,
        align_alpha=config.align_alpha,
        align_beta=config.align_beta,
        align_knn_k=config.align_knn_k,
        align_cka_sample_n=config.align_cka_sample_n,
        align_max_kernel_genes=config.align_max_kernel_genes,
        genept_loss_weight=config.genept_loss_weight,
        verbose=verbose,
        log_interval=10,
        low_memory=True,
        low_memory_batch_size=2048
    )
    
    # Train model
    start_time = time.time()
    if verbose:
        print(f"🔥 Training with {config.n_topics} topics for {config.epochs} epochs...")
    
    # Convert expression matrix to sparse BOW input
    expression_bow = sp.csr_matrix(expression_matrix)
    
    # Standard training
    top_words, train_theta = model.fit_transform_sc(
        cell_embeddings=cell_embeddings,
        gene_names=gene_names,
        expression_bow=expression_bow,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
        min_delta=config.min_delta
    )

    training_time = time.time() - start_time

    # Measure peak GPU memory usage (if available)
    gpu_max_mem_allocated_mb: Optional[float] = None
    gpu_max_mem_reserved_mb: Optional[float] = None
    if torch.cuda.is_available():
        try:
            gpu_max_mem_allocated_mb = float(
                torch.cuda.max_memory_allocated() / (1024**2)
            )
            gpu_max_mem_reserved_mb = float(
                torch.cuda.max_memory_reserved() / (1024**2)
            )
        except Exception:
            gpu_max_mem_allocated_mb = None
            gpu_max_mem_reserved_mb = None

    # Collect result matrices
    beta = model.get_beta()  # topic-gene matrix
    theta = train_theta      # cell-topic matrix
    
    # Compute evaluation metrics
    from scipy.stats import entropy
    
    # Shannon entropy (topic distribution uniformity)
    # Sanitize theta to avoid NaN/Inf in evaluation
    theta_sane = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    # Row-normalize so each cell sums to 1; fallback to uniform for empty rows
    row_sum = theta_sane.sum(axis=1, keepdims=True)
    if row_sum.ndim == 1:
        row_sum = row_sum.reshape(-1, 1)
    zero_rows = (row_sum <= 0)
    if np.any(zero_rows):
        theta_sane[zero_rows[:, 0]] = 1.0 / max(1, theta_sane.shape[1])
        row_sum = theta_sane.sum(axis=1, keepdims=True)
    theta_sane = theta_sane / np.maximum(row_sum, 1e-12)

    topic_weights = theta_sane.mean(axis=0)
    # Normalize to a probability distribution; guard against tiny negatives / precision issues
    topic_weights = np.clip(topic_weights, 0.0, None)
    topic_weights = topic_weights / np.maximum(topic_weights.sum(), 1e-12)
    shannon_entropy = entropy(topic_weights + 1e-12, base=2)
    
    # Effective number of topics
    effective_topics = 2**shannon_entropy
    
    # Dominant topic ratio
    max_topic_weight = topic_weights.max() if topic_weights.size else 0.0
    dominant_topic_ratio = max_topic_weight * 100
    
    results = {
        'beta': beta,
        'theta': theta,
        'top_words': top_words,
        'shannon_entropy': shannon_entropy,
        'effective_topics': effective_topics,
        'dominant_topic_ratio': dominant_topic_ratio,
        'gpu_max_mem_allocated_mb': gpu_max_mem_allocated_mb,
        'gpu_max_mem_reserved_mb': gpu_max_mem_reserved_mb,
    }

    if verbose:
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📊 Shannon Entropy: {shannon_entropy:.3f}")
        print(f"🎯 Effective Topics: {effective_topics:.1f}")
        print(f"👑 Dominant Topic: {dominant_topic_ratio:.1f}%")
        if gpu_max_mem_allocated_mb is not None:
            print(
                f"💾 GPU peak memory (allocated): "
                f"{gpu_max_mem_allocated_mb:.1f} MB"
            )
        if gpu_max_mem_reserved_mb is not None:
            print(
                f"💾 GPU peak memory (reserved): "
                f"{gpu_max_mem_reserved_mb:.1f} MB"
            )

    return model, results, training_time


def save_all_matrices(
    model,
    results: dict,
    config: FastopicConfig,
    verbose: bool = False,
):
    """Save all matrices to disk."""
    if verbose:
        print("\n💾 Saving matrices")
        print("="*60)
    
    # Prepare matrices to persist (the core four)
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    matrices = {
        'cell_topic_matrix': results['theta'],
        'topic_gene_matrix': results['beta'],
        'gene_embeddings': _to_numpy(model.word_embeddings),
        'topic_embeddings': _to_numpy(model.topic_embeddings),
    }

    # Also persist gene names used for this run to enable
    # downstream topic-gene alignment across datasets.
    try:
        gene_names = getattr(model, 'vocab', None)
        if gene_names is None:
            raise AttributeError('model.vocab not available')
        matrices['gene_names'] = list(gene_names)
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not capture gene_names for persistence: {e}")
    
    # Validate matrices
    if not validate_matrices(matrices):
        raise ValueError("Matrix validation failed")
    
    # Save matrices
    saved_files = save_matrices(
        matrices=matrices,
        dataset_name=config.dataset,
        n_topics=config.n_topics,
        output_dir=config.output_dir
    )

    return saved_files




def main():
    """Main entry point."""
    print("🚀 scFASTopic Training Pipeline")
    print("="*80)
    
    # Parse arguments
    args = parse_args()
    
    config = config_from_args(args)

    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if config.verbose:
        print(f"📊 Configuration:")
        print(f"  Dataset: {config.dataset}")
        print(f"  Topics: {config.n_topics}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Early stopping patience: {config.patience}")
        print(f"  Structure alignment (Laplacian+CKA): {config.align_enable}")
        print(f"  GenePT contrastive weight: {config.genept_loss_weight}")
        print(f"  GenePT gene filtering: {config.filter_genept}")
        print(f"  HVG n_top_genes: {config.n_top_genes}")
        print(f"  Gene list path: {config.gene_list_path}")
        print(f"  Embedding file: {config.embedding_file}")
        print(f"  Adata file: {config.adata_path}")
    
    try:
        # Step 1: Load embeddings and preprocess expression matrix
        cell_embeddings, expression_matrix, gene_names = load_embeddings_and_expression(
            embedding_file=config.embedding_file,
            adata_path=config.adata_path,
            verbose=config.verbose,
            filter_genept=config.filter_genept,
            n_top_genes=config.n_top_genes,
            gene_list_path=config.gene_list_path,
        )
        
        # Step 2: Train model
        model, results, training_time = train_fastopic_model(
            cell_embeddings, expression_matrix, gene_names, config, config.verbose
        )

        # Step 3: Save matrices
        saved_files = save_all_matrices(
            model, results, config, config.verbose
        )

        # Step 4: Persist GPU memory stats (if available) for downstream scripts
        gpu_alloc = results.get("gpu_max_mem_allocated_mb")
        gpu_reserved = results.get("gpu_max_mem_reserved_mb")
        if gpu_alloc is not None or gpu_reserved is not None:
            gpu_stats = {
                "dataset": config.dataset,
                "n_topics": config.n_topics,
                "gpu_max_mem_allocated_mb": gpu_alloc,
                "gpu_max_mem_reserved_mb": gpu_reserved,
            }
            gpu_stats_dir = Path(config.output_dir) / "gpu_stats"
            gpu_stats_dir.mkdir(parents=True, exist_ok=True)
            gpu_stats_path = gpu_stats_dir / f"{config.dataset}_gpu_stats_{config.n_topics}.json"
            try:
                with open(gpu_stats_path, "w", encoding="utf-8") as f:
                    json.dump(gpu_stats, f, ensure_ascii=False, indent=2)
                if config.verbose:
                    print(f"💾 Saved GPU stats to: {gpu_stats_path}")
            except Exception as e:
                print(f"⚠️ Failed to save GPU stats to {gpu_stats_path}: {e}")

        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}/")
        
        print(f"\n🎯 Final Results:")
        print(f"  Shannon Entropy: {results['shannon_entropy']:.3f}")
        print(f"  Effective Topics: {results['effective_topics']:.1f}")
        print(f"  Training Time: {training_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
