#!/usr/bin/env bash
# Handy CLI recipes for the scVOTE / FASTopic project.
# This file is meant as a copy-paste cheat sheet, not something
# you run end-to-end in one go.

###############################################################################
# 0. Common environment setup (102 server)
###############################################################################

# cd /data1021/xiepengyu/scVOTE
# export PATH="/data1021/xiepengyu/miniconda3/envs/scvote/bin:$PATH"
# mkdir -p logs
#
# Embedding files are written to results/cell_embedding/<dataset>_vae.pkl
#
# Evaluation now also computes ORA+SPP (GO_BP, BH top10). If you want it to use a
# MsigDB pathway-gene mapping CSV, set:
#   export MSIGDB_PATHWAY_GENE_CSV=/path/to/C2_C5_pathway_gene.csv
# Otherwise it uses `data/gene_sets/C2_C5_pathway_gene.csv.gz` if present, and
# falls back to `data/gene_sets/<dataset>_GO_BP_genesets.csv`.


###############################################################################
# 1. Merge PBMC4k and PBMC8k into PBMC12k (AnnData)
###############################################################################

# python test/merge_pbmc_4k_8k_to_12k.py \
#   --pbmc4k data/PBMC4k.h5ad \
#   --pbmc8k data/PBMC8k.h5ad \
#   --output data/PBMC12k.h5ad \
#   --batch_key batch


###############################################################################
# 2. Single-dataset training (structure alignment; 50 topics)
###############################################################################

# 训练结果默认写到 results/<dataset>_{structure|contrastive|baseline}_K<N_TOPICS>/：
# - 例：PBMC4k + structure + K=50 → results/PBMC4k_structure_K50/
# - 例：PBMC4k + contrastive + K=50 → results/PBMC4k_contrastive_K50/
# - 可选：TOPIC_DIVERSITY_WEIGHT=1e-3（topic diversity 正则，0 表示关闭）
# - 兼容：如需写到 results/tuning/<RUN_TAG>/，设置 RUN_TAG 即可
#
nohup bash -c 'N_TOPICS=50 STRUCTURE_ALIGN=0 CONTRASTIVE_ALIGN=1 TOPIC_DIVERSITY_WEIGHT=1e-3 bash train.sh PBMC4k' \
  > logs/train_PBMC4k_contrastive_K50.log 2>&1 &
#
# 对应训练产物的 dataset 名一般为：
# - 有 alignment：<dataset>_vae_align（例如 PBMC4k_vae_align）
# - 无 alignment：<dataset>_vae（例如 PBMC4k_vae）

# PBMC4k (structure)
# nohup bash -c 'N_TOPICS=50 STRUCTURE_ALIGN=1 CONTRASTIVE_ALIGN=0 bash train.sh PBMC4k' \
#   > logs/train_PBMC4k_structure.log 2>&1 &

# PBMC8k (structure)
# nohup bash -c 'N_TOPICS=50 STRUCTURE_ALIGN=1 CONTRASTIVE_ALIGN=0 bash train.sh PBMC8k' \
#   > logs/train_PBMC8k_structure.log 2>&1 &

# PBMC12k (structure)
# nohup bash -c 'N_TOPICS=50 STRUCTURE_ALIGN=1 CONTRASTIVE_ALIGN=0 bash train.sh PBMC12k' \
#   > logs/train_PBMC12k_structure.log 2>&1 &

# 对比：如需 contrastive 版本，把 STRUCTURE_ALIGN=0 CONTRASTIVE_ALIGN=1 即可：
# nohup bash -c 'N_TOPICS=50 STRUCTURE_ALIGN=0 CONTRASTIVE_ALIGN=1 bash train.sh PBMC4k' \
#   > logs/train_PBMC4k_contrastive.log 2>&1 &

###############################################################################
# 2b. Grid training (Spleen/lung/PBMC4k/kidney × structure/contrastive × K=20/50/100)
###############################################################################

# One-shot launcher (starts jobs via nohup):
# nohup bash -c 'bash test/train_grid.sh' > logs/train_grid.log 2>&1 &


###############################################################################
# 3. Incremental alignment + topic filter（老 pipeline）
###############################################################################

# 使用 PBMC4k_vae_align / PBMC8k_vae_align / PBMC12k_vae_align 做增量 TopicStore 对齐，
# 第二次调用带 --filter_background，会使用“稀疏度 + coherence” 的 topic filter。

# nohup bash incremental.sh > logs/incremental_align.log 2>&1 &


###############################################################################
# 4. Batch effect UMAP：PBMC12k 单数据集（带 batch 信息）
###############################################################################

# 使用 PBMC12k 的 cell-topic matrix，按 topic filter 前/后对比 batch effect。
# 假设：
#   - 单训练结果在 results/tuning/PBMC12k_structure/
#   - 原始数据在 data/PBMC12k.h5ad
#   - obs 中 batch 列为 batch，cell type 列为 cell_type

# python -m incremental_eval.batch_effect_single \
#   --dataset PBMC12k_vae_align \
#   --results_dir results/tuning/PBMC12k_structure \
#   --adata data/PBMC12k.h5ad \
#   --batch_key batch \
#   --celltype_key cell_type \
#   --n_topics 50 \
#   --out_dir results/incremental_eval/batch_effect \
#   --tag PBMC12k_structure \
#   --max_cells 8000 \
#   --sparsity_threshold 0.20 \
#   --coherence_top_n 20 \
#   --coherence_threshold 0.20


###############################################################################
# 5. Batch effect UMAP：PBMC4k vs PBMC8k（两批视角，可选）
###############################################################################

# 使用 PBMC4k / PBMC8k 的 cell-topic matrix，把两个 batch 合并到同一 UMAP 上，
# 按 dataset_a/dataset_b 上色，观察 filter 前/后 batch 混合情况。

# python -m incremental_eval.batch_effect \
#   --dataset_a PBMC4k_vae_align \
#   --dataset_b PBMC8k_vae_align \
#   --results_dir_a results/tuning/PBMC4k_structure \
#   --results_dir_b results/tuning/PBMC8k_structure \
#   --adata_a data/PBMC4k.h5ad \
#   --adata_b data/PBMC8k.h5ad \
#   --label_key cell_type \
#   --n_topics 50 \
#   --out_dir results/incremental_eval/batch_effect \
#   --tag PBMC4k_8k_structure \
#   --max_cells_per_batch 5000 \
#   --sparsity_threshold 0.20 \
#   --topk_mass_threshold -1.0 \
#   --topk 50


###############################################################################
# 6. 合并“相同 top-N gene”的 topics，并做 UMAP 对比
###############################################################################

# 对单数据集（如 PBMC12k_vae_align），如果发现有若干 topics 的 top-N gene 完全相同，
# 可以用下面命令把它们合并为一个 topic，并对比合并前/后的 UMAP：

# 示例：PBMC12k，使用 top 15 genes 判定重复 topic
# python -m incremental_eval.merge_duplicate_topics \
#   --dataset PBMC12k_vae_align \
#   --results_dir results/tuning/PBMC12k_structure \
#   --adata data/PBMC12k.h5ad \
#   --n_topics 50 \
#   --out_dir results/incremental_eval/merge_duplicates \
#   --tag PBMC12k_structure_top15 \
#   --top_n 15 \
#   --max_cells 8000

# 如果想更宽松或更严格，可以调整 --top_n：
#   - --top_n 10  # top10 完全相同才算 duplicate
#   - --top_n 20  # top20 完全相同才算 duplicate


###############################################################################
# 7. 直接调用单独脚本（如 cell embedding、evaluation、可视化）
###############################################################################

# （1）仅提取 cell embedding（一般不需要手动调，用 train.sh 即可）
# python get_cell_emb.py \
#   --input_data data/PBMC4k.h5ad \
#   --dataset_name PBMC4k \
#   --output_dir results/cell_embedding \
#   --n_latent 128 \
#   --gene_list_path data/gene_list/C2_C5_GO.csv \
#   --batch_key batch \
#   --labels_key cell_type \
#   --early_stopping \
#   --verbose

# （2）只跑 evaluation（已训练好后，除了 ARI/NMI，还会输出 ORA/GSEA、TC_extrinsic、TD 等指标）
# python evaluation.py \
#   --adata_path data/PBMC4k.h5ad \
#   --results_dir results \
#   --dataset PBMC4k_vae_align \
#   --n_topics 50 \
#   --label_key cell_type \
#   --out_dir results/evaluation

# （3）只跑可视化脚本
# python visualization.py \
#   --adata_path data/PBMC4k.h5ad \
#   --results_dir results \
#   --dataset PBMC4k_vae_align \
#   --n_topics 50 \
#   --out_dir results/visualization


###############################################################################
# 8. Topic collapse 诊断（topic weights + hierarchy）
###############################################################################

# 对某个单次训练 / tuning 结果，检查是否存在 topic collapse：
#   - 统计 topic weights 的 Shannon entropy / 有效 topic 数
#   - 画出 topic weights 分布和 topic hierarchy dendrogram
#
# 示例：PBMC4k，查看 tuning 结果（RUN_TAG=PBMC4k_structure）
python test/check_topic_collapse.py \
  --results_dir results/tuning/PBMC4k_contrastive_K50 \
  --dataset PBMC4k_vae_align \
  --n_topics 50 \
  --cell_emb results/cell_embedding/PBMC4k_vae.pkl 


###############################################################################
# End of cheat sheet
###############################################################################
