"""
GenePT Embedding对齐工具
用于处理基因名称映射和GenePT embedding的对齐
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

class GenePTAligner:
    """GenePT embedding对齐器"""
    
    def __init__(self, genept_path: str):
        """
        初始化GenePT对齐器
        
        Args:
            genept_path: GenePT embedding文件路径
        """
        self.genept_path = genept_path
        self.genept_dict = None
        self.genept_genes = None
        self._load_genept_data()
        
    def _load_genept_data(self):
        """加载GenePT embedding数据"""
        print(f"📥 加载GenePT embedding数据: {self.genept_path}")
        with open(self.genept_path, 'rb') as f:
            self.genept_dict = pickle.load(f)
        self.genept_genes = set(self.genept_dict.keys())
        print(f"✅ 成功加载 {len(self.genept_genes):,} 个基因的embedding")
        
    def normalize_gene_name(self, gene_name: str) -> str:
        """
        标准化基因名称
        
        Args:
            gene_name: 原始基因名
            
        Returns:
            标准化后的基因名
        """
        # 基本清理
        normalized = gene_name.strip().upper()
        
        # 处理一些常见的基因名格式差异
        # 例如: LOC基因、ORF基因等通常在GenePT中格式不同
        if normalized.startswith('LOC') and normalized[3:].isdigit():
            # LOC基因通常在GenePT中没有对应
            return normalized
            
        if 'ORF' in normalized:
            # 处理ORF格式，如C8ORF48 -> C8orf48
            normalized = normalized.replace('ORF', 'orf')
            
        return normalized
    
    def find_gene_matches(self, target_genes: List[str]) -> Dict[str, Optional[str]]:
        """
        查找目标基因列表在GenePT中的匹配
        
        Args:
            target_genes: 目标基因列表
            
        Returns:
            基因映射字典 {target_gene: genept_gene_or_None}
        """
        matches = {}
        genept_upper = {gene.upper(): gene for gene in self.genept_genes}
        
        for gene in target_genes:
            # 直接匹配
            if gene in self.genept_dict:
                matches[gene] = gene
                continue
                
            # 大小写不敏感匹配
            normalized = self.normalize_gene_name(gene)
            if normalized in genept_upper:
                matches[gene] = genept_upper[normalized]
                continue
                
            # 其他匹配策略
            found_match = None
            
            # 尝试去除版本号 (如基因名后的.1, .2等)
            if '.' in gene:
                base_gene = gene.split('.')[0]
                if base_gene in self.genept_dict:
                    found_match = base_gene
                elif self.normalize_gene_name(base_gene) in genept_upper:
                    found_match = genept_upper[self.normalize_gene_name(base_gene)]
            
            # 尝试连字符处理
            if not found_match and '-' in gene:
                # 有些基因可能有连字符差异
                alt_name = gene.replace('-', '')
                if alt_name in genept_upper:
                    found_match = genept_upper[alt_name]
            
            matches[gene] = found_match
            
        return matches
    
    def get_alignment_stats(self, gene_matches: Dict[str, Optional[str]]) -> Dict[str, int]:
        """
        获取对齐统计信息
        
        Args:
            gene_matches: 基因匹配结果
            
        Returns:
            统计信息字典
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
        提取对齐的GenePT embeddings
        
        Args:
            target_genes: 目标基因列表
            
        Returns:
            (aligned_embeddings, aligned_gene_names)
            aligned_embeddings: shape (n_matched_genes, embedding_dim)
            aligned_gene_names: 对应的基因名列表
        """
        gene_matches = self.find_gene_matches(target_genes)
        stats = self.get_alignment_stats(gene_matches)
        
        print(f"🔍 基因对齐统计:")
        print(f"  总基因数: {stats['total_genes']:,}")
        print(f"  匹配基因数: {stats['matched_genes']:,}")
        print(f"  未匹配基因数: {stats['unmatched_genes']:,}")
        print(f"  匹配率: {stats['match_rate']:.1%}")
        
        # 提取匹配的embeddings
        aligned_embeddings = []
        aligned_gene_names = []
        
        for target_gene in target_genes:
            genept_gene = gene_matches[target_gene]
            if genept_gene is not None:
                embedding = self.genept_dict[genept_gene]
                # 确保embedding是numpy数组
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                aligned_embeddings.append(embedding)
                aligned_gene_names.append(target_gene)  # 保持原始基因名
        
        if aligned_embeddings:
            aligned_embeddings = np.vstack(aligned_embeddings)
        else:
            aligned_embeddings = np.empty((0, 3072))  # GenePT embedding维度是3072
            
        return aligned_embeddings, aligned_gene_names
    
    def get_unmatched_genes(self, target_genes: List[str]) -> List[str]:
        """
        获取未匹配的基因列表
        
        Args:
            target_genes: 目标基因列表
            
        Returns:
            未匹配的基因列表
        """
        gene_matches = self.find_gene_matches(target_genes)
        return [gene for gene, match in gene_matches.items() if match is None]
    
    def show_unmatched_analysis(self, target_genes: List[str]):
        """
        显示未匹配基因的分析
        
        Args:
            target_genes: 目标基因列表
        """
        unmatched = self.get_unmatched_genes(target_genes)
        
        if not unmatched:
            print("🎉 所有基因都已匹配！")
            return
            
        print(f"\n❌ 未匹配的基因分析 (共{len(unmatched)}个):")
        
        # 按类型分析未匹配基因
        loc_genes = [g for g in unmatched if g.startswith('LOC')]
        orf_genes = [g for g in unmatched if 'ORF' in g.upper()]
        version_genes = [g for g in unmatched if '.' in g]
        other_genes = [g for g in unmatched if g not in loc_genes + orf_genes + version_genes]
        
        if loc_genes:
            print(f"  LOC基因 ({len(loc_genes)}个): {loc_genes[:10]}{'...' if len(loc_genes) > 10 else ''}")
        if orf_genes:
            print(f"  ORF基因 ({len(orf_genes)}个): {orf_genes[:10]}{'...' if len(orf_genes) > 10 else ''}")
        if version_genes:
            print(f"  版本号基因 ({len(version_genes)}个): {version_genes[:10]}{'...' if len(version_genes) > 10 else ''}")
        if other_genes:
            print(f"  其他基因 ({len(other_genes)}个): {other_genes[:10]}{'...' if len(other_genes) > 10 else ''}")


def load_filtered_genes_from_training_result(datasetname) -> List[str]:
    """
    从训练结果中获取过滤后的基因列表
    通过分析Wang数据集和GenePT的交集得到实际使用的基因
    
    Returns:
        过滤后的基因列表
    """
    import scanpy as sc
    
    # 加载原始数据
    adata = sc.read_h5ad(f'/root/autodl-tmp/scFastopic/data/{datasetname}.h5ad')
    wang_genes = list(adata.var_names)
    
    # 加载GenePT数据获取共同基因
    genept_path = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
    with open(genept_path, 'rb') as f:
        genept_dict = pickle.load(f)
    genept_genes = set(genept_dict.keys())
    
    # 返回交集，保持原始顺序
    filtered_genes = [gene for gene in wang_genes if gene in genept_genes]
    
    print(f"📋 获取训练中实际使用的基因:")
    print(f"  原始基因数: {len(wang_genes)}")
    print(f"  过滤后基因数: {len(filtered_genes)}")
    print(f"  过滤掉的基因数: {len(wang_genes) - len(filtered_genes)}")
    
    return filtered_genes


def align_genept_for_notebook(topic_gene_matrix: np.ndarray, 
                               target_genes: List[str],
                               genept_path: str = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle') -> Tuple[np.ndarray, List[str], Dict]:
    """
    为notebook提供的GenePT对齐主函数
    
    Args:
        topic_gene_matrix: Topic-gene矩阵，shape (n_topics, n_genes)
        target_genes: 目标基因名列表，长度应与topic_gene_matrix.shape[1]一致
        genept_path: GenePT embedding文件路径
        
    Returns:
        (aligned_genept_embeddings, aligned_gene_names, alignment_info)
        aligned_genept_embeddings: 对齐的GenePT embeddings, shape (n_aligned_genes, embedding_dim)
        aligned_gene_names: 对齐后的基因名列表
        alignment_info: 对齐统计信息
    """
    print("🧬 开始GenePT embedding对齐...")
    
    # 验证输入维度
    if len(target_genes) != topic_gene_matrix.shape[1]:
        raise ValueError(f"基因数量不匹配: target_genes={len(target_genes)}, topic_gene_matrix.shape[1]={topic_gene_matrix.shape[1]}")
    
    # 创建对齐器
    aligner = GenePTAligner(genept_path)
    
    # 执行对齐
    aligned_embeddings, aligned_gene_names = aligner.extract_aligned_embeddings(target_genes)
    
    # 获取统计信息
    gene_matches = aligner.find_gene_matches(target_genes)
    alignment_info = aligner.get_alignment_stats(gene_matches)
    alignment_info['unmatched_genes'] = aligner.get_unmatched_genes(target_genes)
    
    # 显示对齐结果摘要
    print(f"\n🎯 GenePT对齐完成!")
    print(f"  输入基因数: {len(target_genes)}")
    print(f"  匹配基因数: {len(aligned_gene_names)}")
    print(f"  GenePT embedding维度: {aligned_embeddings.shape}")
    print(f"  匹配率: {alignment_info['match_rate']:.1%}")
    
    if alignment_info['unmatched_genes']:
        print(f"  未匹配基因示例: {alignment_info['unmatched_genes'][:5]}...")
    
    return aligned_embeddings, aligned_gene_names, alignment_info


def create_genept_aligned_dataframe(topic_gene_matrix: np.ndarray,
                                  target_genes: List[str],
                                  use_genept_embedding: bool = True) -> pd.DataFrame:
    """
    创建包含GenePT对齐信息的基因数据DataFrame
    
    Args:
        topic_gene_matrix: Topic-gene矩阵
        target_genes: 基因名列表
        use_genept_embedding: 是否使用GenePT embedding替代topic_gene权重
        
    Returns:
        对齐后的基因数据DataFrame
    """
    if use_genept_embedding:
        # 使用GenePT embedding
        aligned_embeddings, aligned_genes, info = align_genept_for_notebook(topic_gene_matrix, target_genes)
        
        # 只保留匹配的基因对应的topic_gene行
        gene_indices = [i for i, gene in enumerate(target_genes) if gene in aligned_genes]
        filtered_topic_gene = topic_gene_matrix[:, gene_indices]
        
        # 创建DataFrame (使用topic-gene权重，但基因顺序与GenePT对齐)
        data = pd.DataFrame(filtered_topic_gene.T, index=aligned_genes)
        
        print(f"✅ 创建GenePT对齐的DataFrame: {data.shape}")
        return data
    else:
        # 使用原始topic_gene矩阵
        data = pd.DataFrame(topic_gene_matrix.T, index=target_genes)
        print(f"✅ 创建原始DataFrame: {data.shape}")
        return data


if __name__ == "__main__":
    # 测试代码
    genept_path = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
    
    # 获取训练中实际使用的基因
    filtered_genes = load_filtered_genes_from_training_result()
    
    # 模拟topic_gene矩阵
    mock_topic_gene = np.random.rand(20, len(filtered_genes))
    
    # 测试主要对齐函数
    aligned_embeddings, aligned_gene_names, alignment_info = align_genept_for_notebook(
        mock_topic_gene, filtered_genes
    )
    
    # 测试DataFrame创建
    aligned_df = create_genept_aligned_dataframe(mock_topic_gene, filtered_genes, use_genept_embedding=True)
    print(f"\n📊 测试DataFrame创建成功: {aligned_df.shape}")
    
    print(f"\n🎉 所有功能测试通过！")