#!/usr/bin/env python3
"""
scFASTopic Results Visualization

Automatically infer the result type from file paths and visualize them.
Supported result types:
- cell embeddings
- cell topic matrix
- gene embeddings
- topic embeddings
- topic gene matrix
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import umap
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import warnings
warnings.filterwarnings('ignore')

class ResultVisualizer:
    """Results visualizer."""
    
    def __init__(self, output_dir: str = "visualization", adata_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.adata_path = adata_path
        self.adata = None
        
        # Supported result types (primarily identified via directory names)
        self.supported_types = {
            'cell_embedding',
            'cell_topic', 
            'gene_embedding',
            'topic_embedding',
            'topic_gene'
        }
        
        # Load adata if a path is provided
        if self.adata_path and os.path.exists(self.adata_path):
            self.load_adata()
    
    def identify_result_type(self, file_path: str) -> str:
        """Infer result type based on directory path."""
        path_obj = Path(file_path)
        
        # Check parent directory names
        parent_dirs = [p.name for p in path_obj.parents] + [path_obj.parent.name]
        
        # Map directory names to result types
        dir_type_mapping = {
            'cell_embedding': 'cell_embedding',
            'cell_topic': 'cell_topic', 
            'topic_gene': 'topic_gene',
            'gene_embedding': 'gene_embedding',
            'topic_embedding': 'topic_embedding'
        }
        
        # Check directory names
        for dir_name in parent_dirs:
            if dir_name in dir_type_mapping:
                return dir_type_mapping[dir_name]
        
        # Fallback to filename-based detection if directory detection fails
        file_name = path_obj.stem.lower()
        
        # Check keywords in filename in priority order
        priority_order = [
            ('cell_topic', ['cell_topic']),
            ('topic_gene', ['topic_gene']),
            ('gene_embedding', ['gene_embedding', 'gene_emb']),
            ('topic_embedding', ['topic_embedding', 'topic_emb']),
            ('cell_embedding', ['cell_embedding'])
        ]
        
        for result_type, keywords in priority_order:
            for keyword in keywords:
                if keyword in file_name:
                    return result_type
        
        return 'unknown'
    
    def preprocess_adata(self, adata_path: str, verbose: bool = True):
        """
        Load and preprocess an AnnData object (consistent with train_fastopic.py).

        Args:
            adata_path: Path to single-cell data (.h5ad).
            verbose: Whether to print details.

        Returns:
            adata: Preprocessed AnnData.
        """
        if verbose:
            print(f"📁 Loading adata: {adata_path}")
        
        # Load data
        adata = sc.read_h5ad(adata_path)
        
        if verbose:
            print(f"Original shape: {adata.shape}")
        
        # Backup cell_type information (before preprocessing)
        cell_type_backup = None
        if 'cell_type' in adata.obs.columns:
            cell_type_backup = adata.obs['cell_type'].copy()
            if verbose:
                print(f"✅ Found cell_type: {len(cell_type_backup.unique())} types")
                print(f"   Types: {list(cell_type_backup.unique())}")
        
        # Simple filtering (consistent with train_fastopic.py)
        # Filter low-quality cells (n_genes < 200)
        sc.pp.filter_cells(adata, min_genes=200)
        
        # Filter lowly expressed genes (min_cells >= 3)
        sc.pp.filter_genes(adata, min_cells=3)
        
        if verbose:
            print(f"After filtering: {adata.shape}")
        
        # Restore cell_type info (matching filtered cells)
        if cell_type_backup is not None:
            # Get indices of retained cells after filtering
            remaining_cells = adata.obs.index
            adata.obs['cell_type'] = cell_type_backup.loc[remaining_cells]
            if verbose:
                print(f"✅ Restored cell_type: {len(adata.obs['cell_type'].unique())} types")
        
        # Normalize total counts to 1 per cell
        sc.pp.normalize_total(adata, target_sum=1)
        
        # log1p transform
        sc.pp.log1p(adata)
        
        if verbose:
            print(f"✅ Preprocessing done: {adata.shape[0]} cells, {adata.shape[1]} genes")
        
        return adata
    
    def load_adata(self):
        """Load and preprocess adata."""
        try:
            self.adata = self.preprocess_adata(self.adata_path, verbose=True)
            
            # Check whether cell_type info exists
            if 'cell_type' in self.adata.obs.columns:
                print(f"✅ Cell type info available for coloring")
            else:
                print("⚠️ 'cell_type' column not found in adata.obs")
                print(f"   Available obs columns: {list(self.adata.obs.columns)}")
                
        except Exception as e:
            print(f"❌ Failed to load adata: {e}")
            self.adata = None
    
    def load_data(self, file_path: str) -> Any:
        """Load pickle data."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded: {file_path}")
            print(f"   Type: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"   Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"❌ Failed to load: {file_path}, error: {e}")
            return None
    
    def load_result(self, file_path: str) -> Dict[str, Any]:
        """Load a single result file and annotate it with type."""
        result_type = self.identify_result_type(file_path)
        data = self.load_data(file_path)
        
        if data is None:
            return None
        
        return {
            'type': result_type,
            'data': data,
            'file_path': file_path
        }
    
    def load_results(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple result files in batch."""
        results = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"📁 Loading file: {file_path}")
                result = self.load_result(file_path)
                if result:
                    print(f"   Type: {result['type']}")
                    results.append(result)
            else:
                print(f"❌ File not found: {file_path}")
        
        return results
    
    def visualize_results(self, results: List[Dict[str, Any]]):
        """Visualize results (placeholder; plug in plotting as needed)."""
        print(f"\n🎨 Visualizing {len(results)} result files")
        
        for result in results:
            result_type = result['type']
            data = result['data']
            file_path = result['file_path']
            
            print(f"\n📊 {result_type}: {file_path}")
            print(f"   Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            # Implement concrete plotting logic
            if result_type == 'cell_embedding':
                self.plot_cell_embedding_umap(data, file_path)
            elif result_type == 'cell_topic':
                self.plot_cell_topic_umap(data, file_path)
            elif result_type == 'gene_embedding':
                # self.plot_gene_embeddings(data, file_path)
                pass
            elif result_type == 'topic_embedding':
                self.plot_topic_embedding_umap(data, file_path)
            elif result_type == 'topic_gene':
                # self.plot_topic_gene_matrix(data, file_path)
                pass
    
    def plot_cell_topic_umap(self, cell_topic_matrix: np.ndarray, file_path: str):
        """Plot UMAP for cell-topic matrix, colored by cell type."""
        print(f"\n🎨 Plotting Cell Topic UMAP: {file_path}")
        
        # Check data dimensions
        n_cells, n_topics = cell_topic_matrix.shape
        print(f"   Cells: {n_cells}, Topics: {n_topics}")
        
        # Check adata and cell_type availability
        if self.adata is None:
            print("⚠️ No adata provided; using default color")
            cell_types = None
        elif 'cell_type' not in self.adata.obs.columns:
            print("⚠️ No 'cell_type' info in adata; using default color")
            cell_types = None
        else:
            cell_types = self.adata.obs['cell_type'].values
            # Ensure cell counts match
            if len(cell_types) != n_cells:
                print(f"⚠️ Cell count mismatch: cell_topic({n_cells}) vs adata({len(cell_types)})")
                cell_types = None
        
        # Run UMAP for dimensionality reduction
        print("🔄 Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(cell_topic_matrix)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        if cell_types is not None:
            # Color by cell type
            unique_types = np.unique(cell_types)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
            
            for i, cell_type in enumerate(unique_types):
                mask = cell_types == cell_type
                plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                           c=[colors[i]], label=cell_type, alpha=0.7, s=20)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            title = f"Cell Topic UMAP (colored by cell type)\n{n_cells} cells, {n_topics} topics"
        else:
            # Use default color
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                       c='skyblue', alpha=0.7, s=20)
            title = f"Cell Topic UMAP\n{n_cells} cells, {n_topics} topics"
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save image
        file_stem = Path(file_path).stem
        output_file = self.output_dir / f"{file_stem}_umap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved UMAP: {output_file}")
        
        return umap_coords
    
    def plot_cell_embedding_umap(self, cell_embeddings: np.ndarray, file_path: str):
        """Plot UMAP for cell embeddings, colored by cell type."""
        print(f"\n🎨 Plotting Cell Embedding UMAP: {file_path}")
        
        # Check data dimensions
        n_cells, embedding_dim = cell_embeddings.shape
        print(f"   Cells: {n_cells}, Embedding dim: {embedding_dim}")
        
        # Check adata and cell_type availability
        if self.adata is None:
            print("⚠️ No adata provided; using default color")
            cell_types = None
        elif 'cell_type' not in self.adata.obs.columns:
            print("⚠️ No 'cell_type' info in adata; using default color")
            cell_types = None
        else:
            cell_types = self.adata.obs['cell_type'].values
            # Ensure cell counts match
            if len(cell_types) != n_cells:
                print(f"⚠️ Cell count mismatch: cell_embedding({n_cells}) vs adata({len(cell_types)})")
                cell_types = None
        
        # Run UMAP for dimensionality reduction
        print("🔄 Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(cell_embeddings)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        if cell_types is not None:
            # Color by cell type
            unique_types = np.unique(cell_types)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
            
            for i, cell_type in enumerate(unique_types):
                mask = cell_types == cell_type
                plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                           c=[colors[i]], label=cell_type, alpha=0.7, s=20)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            title = f"Cell Embedding UMAP (colored by cell type)\n{n_cells} cells, {embedding_dim}D embeddings"
        else:
            # Use default color
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                       c='skyblue', alpha=0.7, s=20)
            title = f"Cell Embedding UMAP\n{n_cells} cells, {embedding_dim}D embeddings"
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save image
        file_stem = Path(file_path).stem
        output_file = self.output_dir / f"{file_stem}_umap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved UMAP: {output_file}")
        
        return umap_coords
    
    def plot_topic_embedding_umap(self, topic_embeddings: np.ndarray, file_path: str):
        """Plot UMAP for topic embeddings, colored by topic ID."""
        print(f"\n🎨 Plotting Topic Embedding UMAP: {file_path}")
        
        # Check data dimensions
        n_topics, embedding_dim = topic_embeddings.shape
        print(f"   Topics: {n_topics}, Embedding dim: {embedding_dim}")
        
        # Run UMAP for dimensionality reduction
        print("🔄 Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_topics-1), min_dist=0.1)
        umap_coords = reducer.fit_transform(topic_embeddings)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Color by topic ID
        topic_ids = np.arange(n_topics)
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_topics)))
        
        # If more than 20 topics, use a continuous colormap
        if n_topics > 20:
            colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
            scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                c=topic_ids, cmap='viridis', alpha=0.8, s=80)
            plt.colorbar(scatter, label='Topic ID')
        else:
            # For <= 20 topics, use discrete colors and a legend
            for i in range(n_topics):
                plt.scatter(umap_coords[i, 0], umap_coords[i, 1], 
                           c=[colors[i]], label=f'Topic {i}', alpha=0.8, s=80)
            
            # Only show the legend when topic count is modest
            if n_topics <= 12:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Annotate topic IDs
        for i in range(n_topics):
            plt.annotate(f'T{i}', (umap_coords[i, 0], umap_coords[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.7)
        
        title = f"Topic Embedding UMAP\n{n_topics} topics, {embedding_dim}D embeddings"
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save image
        file_stem = Path(file_path).stem
        output_file = self.output_dir / f"{file_stem}_umap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved UMAP: {output_file}")
        
        return umap_coords

def main():
    parser = argparse.ArgumentParser(description="scFASTopic results visualization")
    parser.add_argument("files", nargs="+", help="Result file paths")
    parser.add_argument("--output_dir", default="visualization", help="Output directory")
    parser.add_argument("--adata_path", help="Path to adata file (for cell type info)")
    parser.add_argument("--no_plot", action="store_true", help="Load only without plotting")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultVisualizer(output_dir=args.output_dir, adata_path=args.adata_path)
    
    # Load results in batch
    results = visualizer.load_results(args.files)
    
    if not results:
        print("❌ No files loaded successfully")
        return
    
    # Print loading summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    type_counts = {}
    for result in results:
        result_type = result['type']
        type_counts[result_type] = type_counts.get(result_type, 0) + 1
    
    for result_type, count in type_counts.items():
        print(f"📊 {result_type}: {count} files")
    
    # Visualize if requested
    if not args.no_plot:
        visualizer.visualize_results(results)

if __name__ == "__main__":
    main()
