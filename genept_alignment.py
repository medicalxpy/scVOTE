"""
GenePT embedding alignment utilities.
Provides gene-name normalization, matching, and alignment of target genes
to GenePT embeddings.
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

class GenePTAligner:
    """GenePT embedding aligner."""
    
    def __init__(self, genept_path: str):
        """
        Initialize the GenePT aligner.

        Args:
            genept_path: Path to the GenePT embedding file.
        """
        self.genept_path = genept_path
        self.genept_dict = None
        self.genept_genes = None
        self._load_genept_data()
        
    def _load_genept_data(self):
        """Load GenePT embedding data."""
        print(f"📥 Loading GenePT embeddings: {self.genept_path}")
        with open(self.genept_path, 'rb') as f:
            self.genept_dict = pickle.load(f)
        self.genept_genes = set(self.genept_dict.keys())
        print(f"✅ Loaded embeddings for {len(self.genept_genes):,} genes")
        
    def normalize_gene_name(self, gene_name: str) -> str:
        """
        Normalize a gene name.

        Args:
            gene_name: Original gene symbol.

        Returns:
            Normalized gene symbol.
        """
        # Basic cleanup
        normalized = gene_name.strip().upper()
        
        # Handle common gene-name formatting differences
        # Examples: LOC genes, ORF genes often look different in GenePT
        if normalized.startswith('LOC') and normalized[3:].isdigit():
            # LOC genes typically don't have counterparts in GenePT
            return normalized
            
        if 'ORF' in normalized:
            # Handle ORF format, e.g., C8ORF48 -> C8orf48
            normalized = normalized.replace('ORF', 'orf')
            
        return normalized
    
    def find_gene_matches(self, target_genes: List[str]) -> Dict[str, Optional[str]]:
        """
        Find GenePT matches for a target gene list.

        Args:
            target_genes: List of target genes.

        Returns:
            Mapping {target_gene: genept_gene_or_None}.
        """
        matches = {}
        genept_upper = {gene.upper(): gene for gene in self.genept_genes}
        
        for gene in target_genes:
            # Exact match
            if gene in self.genept_dict:
                matches[gene] = gene
                continue
                
            # Case-insensitive match
            normalized = self.normalize_gene_name(gene)
            if normalized in genept_upper:
                matches[gene] = genept_upper[normalized]
                continue
                
            # Other matching strategies
            found_match = None
            
            # Try removing version suffix (e.g., .1, .2)
            if '.' in gene:
                base_gene = gene.split('.')[0]
                if base_gene in self.genept_dict:
                    found_match = base_gene
                elif self.normalize_gene_name(base_gene) in genept_upper:
                    found_match = genept_upper[self.normalize_gene_name(base_gene)]
            
            # Try hyphen handling
            if not found_match and '-' in gene:
                # Some genes may differ by hyphen usage
                alt_name = gene.replace('-', '')
                if alt_name in genept_upper:
                    found_match = genept_upper[alt_name]
            
            matches[gene] = found_match
            
        return matches
    
    def get_alignment_stats(self, gene_matches: Dict[str, Optional[str]]) -> Dict[str, int]:
        """
        Compute alignment statistics.

        Args:
            gene_matches: Gene matching results.

        Returns:
            Stats dictionary.
        """
        total_genes = len(gene_matches)
        matched_genes = sum(1 for match in gene_matches.values() if match is not None)
        unmatched_genes = total_genes - matched_genes
        
        return {
            'total_genes': total_genes,
            'matched_genes': matched_genes,
            'unmatched_genes': unmatched_genes,
            'match_rate': matched_genes / total_genes if total_genes > 0 else 0
        }
    
    def extract_aligned_embeddings(self, target_genes: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract aligned GenePT embeddings.

        Args:
            target_genes: Target gene list.

        Returns:
            (aligned_embeddings, aligned_gene_names)
            aligned_embeddings: shape (n_matched_genes, embedding_dim)
            aligned_gene_names: list of corresponding gene names
        """
        gene_matches = self.find_gene_matches(target_genes)
        stats = self.get_alignment_stats(gene_matches)
        
        print(f"🔍 Gene alignment stats:")
        print(f"  Total genes: {stats['total_genes']:,}")
        print(f"  Matched genes: {stats['matched_genes']:,}")
        print(f"  Unmatched genes: {stats['unmatched_genes']:,}")
        print(f"  Match rate: {stats['match_rate']:.1%}")
        
        # Extract matched embeddings
        aligned_embeddings = []
        aligned_gene_names = []
        
        for target_gene in target_genes:
            genept_gene = gene_matches[target_gene]
            if genept_gene is not None:
                embedding = self.genept_dict[genept_gene]
                # Ensure numpy arrays
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                aligned_embeddings.append(embedding)
                aligned_gene_names.append(target_gene)  # Keep original gene name
        
        if aligned_embeddings:
            aligned_embeddings = np.vstack(aligned_embeddings)
        else:
            aligned_embeddings = np.empty((0, 3072))  # GenePT embedding dim is 3072
            
        return aligned_embeddings, aligned_gene_names
    
    def get_unmatched_genes(self, target_genes: List[str]) -> List[str]:
        """
        Get the list of unmatched genes.

        Args:
            target_genes: Target gene list.

        Returns:
            Unmatched genes.
        """
        gene_matches = self.find_gene_matches(target_genes)
        return [gene for gene, match in gene_matches.items() if match is None]
    
    def show_unmatched_analysis(self, target_genes: List[str]):
        """
        Display analysis of unmatched genes.

        Args:
            target_genes: Target gene list.
        """
        unmatched = self.get_unmatched_genes(target_genes)
        
        if not unmatched:
            print("🎉 All genes matched!")
            return
            
        print(f"\n❌ Unmatched genes analysis (n={len(unmatched)}):")
        
        # Analyze unmatched genes by type
        loc_genes = [g for g in unmatched if g.startswith('LOC')]
        orf_genes = [g for g in unmatched if 'ORF' in g.upper()]
        version_genes = [g for g in unmatched if '.' in g]
        other_genes = [g for g in unmatched if g not in loc_genes + orf_genes + version_genes]
        
        if loc_genes:
            print(f"  LOC genes ({len(loc_genes)}): {loc_genes[:10]}{'...' if len(loc_genes) > 10 else ''}")
        if orf_genes:
            print(f"  ORF genes ({len(orf_genes)}): {orf_genes[:10]}{'...' if len(orf_genes) > 10 else ''}")
        if version_genes:
            print(f"  Versioned genes ({len(version_genes)}): {version_genes[:10]}{'...' if len(version_genes) > 10 else ''}")
        if other_genes:
            print(f"  Other genes ({len(other_genes)}): {other_genes[:10]}{'...' if len(other_genes) > 10 else ''}")


def load_filtered_genes_from_training_result(datasetname) -> List[str]:
    """
    Get the filtered gene list used in training.
    Compute the intersection between the Wang dataset and GenePT genes.

    Returns:
        Filtered gene list.
    """
    import scanpy as sc
    
    # Load raw data
    adata = sc.read_h5ad(f'/root/autodl-tmp/scFastopic/data/{datasetname}.h5ad')
    wang_genes = list(adata.var_names)
    
    # Load GenePT data and get common genes
    genept_path = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
    with open(genept_path, 'rb') as f:
        genept_dict = pickle.load(f)
    genept_genes = set(genept_dict.keys())
    
    # Return intersection while preserving original order
    filtered_genes = [gene for gene in wang_genes if gene in genept_genes]
    
    print(f"📋 Genes used in training:")
    print(f"  Original gene count: {len(wang_genes)}")
    print(f"  Filtered gene count: {len(filtered_genes)}")
    print(f"  Removed genes: {len(wang_genes) - len(filtered_genes)}")
    
    return filtered_genes


def align_genept_for_notebook(topic_gene_matrix: np.ndarray, 
                               target_genes: List[str],
                               genept_path: str = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle') -> Tuple[np.ndarray, List[str], Dict]:
    """
    Main GenePT alignment function for notebooks.

    Args:
        topic_gene_matrix: Topic-gene matrix, shape (n_topics, n_genes).
        target_genes: Target gene list; length should equal topic_gene_matrix.shape[1].
        genept_path: Path to the GenePT embedding file.

    Returns:
        (aligned_genept_embeddings, aligned_gene_names, alignment_info)
        aligned_genept_embeddings: shape (n_aligned_genes, embedding_dim)
        aligned_gene_names: aligned gene names
        alignment_info: alignment statistics
    """
    print("🧬 Starting GenePT embedding alignment...")
    
    # Validate input dimensions
    if len(target_genes) != topic_gene_matrix.shape[1]:
        raise ValueError(f"Gene count mismatch: target_genes={len(target_genes)}, topic_gene_matrix.shape[1]={topic_gene_matrix.shape[1]}")
    
    # Create aligner
    aligner = GenePTAligner(genept_path)
    
    # Perform alignment
    aligned_embeddings, aligned_gene_names = aligner.extract_aligned_embeddings(target_genes)
    
    # Collect statistics
    gene_matches = aligner.find_gene_matches(target_genes)
    alignment_info = aligner.get_alignment_stats(gene_matches)
    alignment_info['unmatched_genes'] = aligner.get_unmatched_genes(target_genes)
    
    # Show alignment summary
    print(f"\n🎯 GenePT alignment completed!")
    print(f"  Input genes: {len(target_genes)}")
    print(f"  Matched genes: {len(aligned_gene_names)}")
    print(f"  GenePT embedding shape: {aligned_embeddings.shape}")
    print(f"  Match rate: {alignment_info['match_rate']:.1%}")
    
    if alignment_info['unmatched_genes']:
        print(f"  Unmatched examples: {alignment_info['unmatched_genes'][:5]}...")
    
    return aligned_embeddings, aligned_gene_names, alignment_info


def create_genept_aligned_dataframe(topic_gene_matrix: np.ndarray,
                                  target_genes: List[str],
                                  use_genept_embedding: bool = True) -> pd.DataFrame:
    """
    Create a gene DataFrame with GenePT alignment information.

    Args:
        topic_gene_matrix: Topic-gene matrix.
        target_genes: Gene name list.
        use_genept_embedding: Whether to substitute topic_gene weights with GenePT embeddings.

    Returns:
        Aligned gene DataFrame.
    """
    if use_genept_embedding:
        # Use GenePT embeddings
        aligned_embeddings, aligned_genes, info = align_genept_for_notebook(topic_gene_matrix, target_genes)
        
        # Keep topic_gene rows corresponding to matched genes only
        gene_indices = [i for i, gene in enumerate(target_genes) if gene in aligned_genes]
        filtered_topic_gene = topic_gene_matrix[:, gene_indices]
        
        # Create DataFrame (topic-gene weights; gene order aligned with GenePT)
        data = pd.DataFrame(filtered_topic_gene.T, index=aligned_genes)
        
        print(f"✅ Created GenePT-aligned DataFrame: {data.shape}")
        return data
    else:
        # Use original topic_gene matrix
        data = pd.DataFrame(topic_gene_matrix.T, index=target_genes)
        print(f"✅ Created original DataFrame: {data.shape}")
        return data


if __name__ == "__main__":
    # Test code
    genept_path = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
    
    # Get genes used in training
    filtered_genes = load_filtered_genes_from_training_result()
    
    # Simulate topic_gene matrix
    mock_topic_gene = np.random.rand(20, len(filtered_genes))
    
    # Test main alignment function
    aligned_embeddings, aligned_gene_names, alignment_info = align_genept_for_notebook(
        mock_topic_gene, filtered_genes
    )
    
    # Test DataFrame creation
    aligned_df = create_genept_aligned_dataframe(mock_topic_gene, filtered_genes, use_genept_embedding=True)
    print(f"\n📊 DataFrame created successfully: {aligned_df.shape}")
    
    print(f"\n🎉 All tests passed!")
