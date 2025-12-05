from __future__ import annotations

import dataclasses
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import os
import pickle
import glob


# -----------------------------
# Topic Store for global topics
# -----------------------------

@dataclass
class TopicMeta:
    """Lightweight metadata for a topic in the store."""
    first_dataset: str
    last_dataset: str
    match_count: int = 1


class TopicStore:
    """A minimal global store that accumulates topic representations.

    Representation = topic-gene vector computed as (topic_embeddings @ gene_embeddings.T).
    All rows are L2-normalised on insertion/update for consistency across datasets.
    """

    def __init__(
        self,
        store_embeddings: Optional[np.ndarray] = None,
        topic_ids: Optional[List[str]] = None,
        meta: Optional[List[TopicMeta]] = None,
        gene_names: Optional[List[str]] = None,
    ) -> None:
        if store_embeddings is None:
            self.store_embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            arr = np.asarray(store_embeddings, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("store_embeddings must be 2-D")
            self.store_embeddings = self._l2_normalize(arr)
        self.topic_ids: List[str] = topic_ids or []
        self.meta: List[TopicMeta] = meta or []
        self.gene_names: List[str] = list(gene_names) if gene_names is not None else []
        if len(self.topic_ids) != len(self.meta) or (
            self.store_embeddings.size > 0
            and self.store_embeddings.shape[0] != len(self.topic_ids)
        ):
            raise ValueError("Embeddings, topic_ids and meta length mismatch in TopicStore")

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return x / nrm

    @property
    def size(self) -> int:
        return self.store_embeddings.shape[0] if self.store_embeddings.size else 0

    @property
    def dim(self) -> int:
        return self.store_embeddings.shape[1] if self.store_embeddings.size else 0

    def add_topics(
        self,
        dataset_name: str,
        *,
        results_dir: str = "results",
        data_dir: str = "data",
        # ID and normalisation
        id_prefix: str = "T",
        normalize: bool = True,
        # UOT controls (always used)
        reg: float = 0.05,
        reg_m: float = 10.0,
        metric: str = "euclidean",
        smoothing: float = 0.5,
        min_transport_mass: float = 1e-3,
        min_best_ratio: float = 0.5,
        # Background topic filtering (pre-UOT)
        filter_background: bool = True,
        sparsity_threshold: float = 0.20,
        topk_mass_threshold: Optional[float] = None,
        topk: int = 50,
        # Topic coherence (pre-UOT, used together with sparsity)
        coherence_top_n: int = 20,
        coherence_threshold: float = 0.20,
        # Gene handling
        expand_genes: bool = True,
        # Debug/analysis
        return_coupling: bool = False,
    ) -> dict:
        """Add topics for a dataset using UOT alignment on topic-gene vectors.

        Steps:
          1) Load topic embeddings and gene embeddings from results_dir.
          2) Build topic-gene = topic_embeddings @ gene_embeddings.T.
          3) Align columns to the store's gene order (expanding store if enabled).
          4) UOT alignment vs. store. Matched -> EMA update; unmatched -> append.

        Returns dict: matched, added, assigned_ids, store_size[, coupling]
        """

        # Load from results dir by dataset name
        topic_pat = [
            os.path.join(results_dir, "topic_embedding", f"{dataset_name}*topic_embeddings*.pkl"),
        ]
        gene_pat = [
            os.path.join(results_dir, "gene_embedding", f"{dataset_name}*gene_embeddings*.pkl"),
        ]
        gene_names_pat = [
            os.path.join(results_dir, "gene_embedding", f"{dataset_name}*gene_names*.pkl"),
        ]
        cell_topic_pat = [
            os.path.join(results_dir, "cell_topic", f"{dataset_name}*cell_topic_matrix*.pkl"),
        ]
        def _find_single_file(patterns: List[str]) -> Optional[str]:
            for pat in patterns:
                matches = sorted(glob.glob(pat))
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return matches[0]
            return None
        # Always load embeddings (topic/gene) and derive topic-gene via dot product
        topic_path = _find_single_file(topic_pat)
        if topic_path is None:
            raise FileNotFoundError(
                f"Cannot find topic embeddings for dataset={dataset_name} under {results_dir}/topic_embedding"
            )
        with open(topic_path, "rb") as f:
            topic_embeddings = pickle.load(f)
        topic_embeddings = np.asarray(topic_embeddings, dtype=np.float32)

        gene_path = _find_single_file(gene_pat)
        if gene_path is None:
            raise FileNotFoundError(
                f"Cannot find gene embeddings for dataset={dataset_name} under {results_dir}/gene_embedding"
            )
        with open(gene_path, "rb") as f:
            gene_embeddings = pickle.load(f)
        gene_embeddings = np.asarray(gene_embeddings, dtype=np.float32)

        # Try to load persisted gene_names from training; fallback later based on shapes
        gene_names_path = _find_single_file(gene_names_pat)
        new_gene_names: Optional[List[str]] = None
        if gene_names_path is not None:
            with open(gene_names_path, "rb") as f:
                new_gene_names = list(pickle.load(f))

        # Compute topic-gene via dot product
        if topic_embeddings.ndim == 1:
            topic_embeddings = topic_embeddings[None, :]
        if gene_embeddings.ndim == 1:
            gene_embeddings = gene_embeddings[None, :]
        if topic_embeddings.shape[1] != gene_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: topic D={topic_embeddings.shape[1]} vs gene D={gene_embeddings.shape[1]}"
            )
        new_embeddings = topic_embeddings @ gene_embeddings.T  # (K, G)
        # Optional weights from cell_topic
        new_weights = None
        cell_topic_path = _find_single_file(cell_topic_pat)
        if cell_topic_path is not None:
            with open(cell_topic_path, "rb") as f:
                cell_topic = pickle.load(f)
            cell_topic = np.asarray(cell_topic, dtype=np.float32)
            new_weights = cell_topic.sum(axis=0)

        # Ensure ndarray
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)
        if new_embeddings.ndim == 1:
            new_embeddings = new_embeddings[None, :]
        # Fallback: anonymous gene names based on available shapes if not provided
        if new_gene_names is None:
            if gene_embeddings is not None:
                G = int(gene_embeddings.shape[0])
            else:
                G = int(new_embeddings.shape[1])
            new_gene_names = [f"GENE_{i}" for i in range(G)]

        # Optional: filter background (dense, low-coherence) topics before alignment
        original_K = new_embeddings.shape[0]
        keep_idx = np.arange(original_K, dtype=int)
        filtered_idx: List[int] = []
        if filter_background and original_K > 0:
            X = np.clip(new_embeddings, 0.0, None)
            row_sum = X.sum(axis=1, keepdims=True)
            eps = 1e-12
            # Identify zero rows to avoid division artifacts
            zero_rows = (row_sum <= eps).reshape(-1)
            P = X / np.maximum(row_sum, eps)
            # Hoyer sparsity
            G = P.shape[1]
            L1 = np.sum(P, axis=1)
            L2 = np.sqrt(np.sum(P * P, axis=1) + 1e-20)
            sqrtG = float(np.sqrt(G)) if G > 1 else 1.0
            # For zero rows, force sparsity=0 (drop)
            hoyer = (sqrtG - (L1 / L2)) / (sqrtG - 1.0 + 1e-12) if sqrtG > 1.0 else np.zeros_like(L2)
            hoyer[zero_rows] = 0.0

            # Topic coherence based on gene embeddings: for each topic, compute
            # the average pairwise cosine similarity among its top-N genes.
            # Low coherence means the top genes are not semantically similar.
            top_n = int(max(1, min(coherence_top_n, G)))
            # Pre-normalise gene embeddings once
            ge = np.asarray(gene_embeddings, dtype=np.float32)
            ge_norm = ge / np.maximum(
                np.linalg.norm(ge, axis=1, keepdims=True), 1e-12
            )
            coherence = np.zeros(original_K, dtype=np.float32)
            for k in range(original_K):
                # Use top-N genes by probability in P
                row = P[k]
                if G <= top_n:
                    top_idx = np.arange(G, dtype=int)
                else:
                    top_idx = np.argpartition(row, -top_n)[-top_n:]
                vecs = ge_norm[top_idx]
                if vecs.shape[0] <= 1:
                    coherence[k] = 0.0
                    continue
                sim = vecs @ vecs.T
                iu = np.triu_indices(sim.shape[0], k=1)
                coherence[k] = float(sim[iu].mean())

            low_sparsity = hoyer < float(sparsity_threshold)
            low_coherence = coherence <= float(coherence_threshold)

            # Optional top-k mass gate
            low_mass = np.zeros_like(low_sparsity, dtype=bool)
            if topk_mass_threshold is not None and float(topk_mass_threshold) > 0.0 and G > 0:
                k_eff = int(max(1, min(topk, G)))
                part = np.partition(P, kth=G - k_eff, axis=1)
                topk_mass = np.sum(part[:, -k_eff:], axis=1)
                low_mass = topk_mass < float(topk_mass_threshold)

            # Final drop condition: only filter topics that are simultaneously
            # low-sparsity (dense), low-coherence, and (optionally) low-mass.
            drop = low_sparsity & low_coherence
            if topk_mass_threshold is not None and float(topk_mass_threshold) > 0.0 and G > 0:
                drop &= low_mass

            keep = ~drop
            keep_idx = np.where(keep)[0]
            filtered_idx = [int(i) for i in np.where(drop)[0]]
            if keep_idx.size == 0:
                # Nothing to add or match; early exit
                out = {"matched": [], "added": [], "assigned_ids": [], "store_size": self.size, "filtered": filtered_idx}
                if return_coupling:
                    out["coupling"] = None
                return out
            new_embeddings = new_embeddings[keep_idx]
            if new_weights is not None:
                new_weights = np.asarray(new_weights)[keep_idx]

        # If store empty: add all directly
        if self.size == 0 and self.dim == 0:
            # Initialise store gene names
            self.gene_names = list(new_gene_names)
            base = self._l2_normalize(new_embeddings) if normalize else new_embeddings
            self.store_embeddings = base
            assigned: List[str] = []
            for _ in range(new_embeddings.shape[0]):
                new_id = f"{id_prefix}{len(self.topic_ids)}"
                assigned.append(new_id)
                self.topic_ids.append(new_id)
                self.meta.append(TopicMeta(first_dataset=dataset_name, last_dataset=dataset_name, match_count=1))
            # 'added' should refer to original topic indices
            added_orig = [int(i) for i in keep_idx]
            out = {"matched": [], "added": added_orig, "assigned_ids": assigned, "store_size": self.size}
            if filter_background:
                out["filtered"] = filtered_idx
            if return_coupling:
                out["coupling"] = None
            return out

        # Non-empty store: align or expand gene dimension, then check
        new_embeddings = self._align_genes(new_embeddings, new_gene_names, expand=expand_genes)
        if new_embeddings.shape[1] != self.dim:
            raise ValueError(f"Aligned gene dimension mismatch: store dim={self.dim}, new dim={new_embeddings.shape[1]}")

        # UOT path
        try:
            import ot
        except ImportError as exc:  # pragma: no cover
            raise ImportError("POT is required for UOT alignment. Install via `pip install pot`. ") from exc

        def _norm_w(w: Optional[np.ndarray], n: int) -> np.ndarray:
            if w is None:
                return np.ones(n, dtype=np.float64) / float(n)
            w = np.asarray(w, dtype=np.float64)
            if w.ndim != 1 or w.shape[0] != n:
                raise ValueError("Weights must be 1-D and match number of topics")
            s = w.sum()
            if s <= 0:
                raise ValueError("Weights must sum to positive value")
            return w / s

        a = _norm_w(None, self.size)
        b = _norm_w(new_weights, new_embeddings.shape[0])

        new_norm = self._l2_normalize(new_embeddings) if normalize else new_embeddings

        cost = ot.dist(self.store_embeddings.astype(np.float64), new_norm.astype(np.float64), metric=metric)
        coupling = ot.unbalanced.sinkhorn_unbalanced(a, b, cost, reg=reg, reg_m=reg_m)
        mass_new = coupling.sum(axis=0)

        # Barycentric projections of new topics in the store space
        eps = 1e-8
        bary = np.zeros_like(new_norm)
        for j in range(new_norm.shape[0]):
            col = coupling[:, j]
            m = mass_new[j]
            if m <= eps:
                bary[j] = self.store_embeddings.mean(axis=0)
            else:
                bary[j] = (col[:, None] * self.store_embeddings).sum(axis=0) / m

        matched_pairs: List[Tuple[int, int]] = []
        to_update_idx: List[int] = []
        to_update_vec: List[np.ndarray] = []
        # Maintain both relative and original indices for additions
        to_add_rel: List[int] = []  # indices into new_norm
        to_add_orig: List[int] = []  # original topic indices before filtering

        for j in range(new_norm.shape[0]):
            total = mass_new[j]
            if total <= min_transport_mass:
                to_add_rel.append(j)
                to_add_orig.append(int(keep_idx[j]))
                continue
            col = coupling[:, j]
            i_best = int(np.argmax(col))
            best = float(col[i_best])
            ratio = best / float(total + 1e-12)
            if ratio < min_best_ratio:
                to_add_rel.append(j)
                to_add_orig.append(int(keep_idx[j]))
            else:
                matched_pairs.append((i_best, int(keep_idx[j])))
                to_update_idx.append(i_best)
                to_update_vec.append(bary[j])

        if to_update_idx:
            self.update_topics(to_update_idx, np.stack(to_update_vec), dataset_name=dataset_name, alpha=float(np.clip(smoothing, 0.0, 1.0)))
        assigned_ids: List[str] = []
        if to_add_rel:
            # Append unmatched new topics
            base = new_norm[to_add_rel]
            self.store_embeddings = np.vstack([self.store_embeddings, base])
            for _ in range(len(to_add_rel)):
                new_id = f"{id_prefix}{len(self.topic_ids)}"
                assigned_ids.append(new_id)
                self.topic_ids.append(new_id)
                self.meta.append(TopicMeta(first_dataset=dataset_name, last_dataset=dataset_name, match_count=1))

        out = {"matched": matched_pairs, "added": to_add_orig, "assigned_ids": assigned_ids, "store_size": self.size}
        if filter_background:
            out["filtered"] = filtered_idx
        if return_coupling:
            out["coupling"] = coupling
        return out

    def update_topics(
        self,
        indices: List[int],
        updated_embeddings: np.ndarray,
        *,
        dataset_name: str,
        alpha: float = 0.5,
        normalize: bool = True,
    ) -> None:
        """EMA-style update for existing topics at the given indices."""
        if len(indices) == 0:
            return
        updated_embeddings = np.asarray(updated_embeddings, dtype=np.float32)
        if updated_embeddings.ndim == 1:
            updated_embeddings = updated_embeddings[None, :]
        if updated_embeddings.shape[0] != len(indices):
            raise ValueError("Number of embeddings must match number of indices")
        if updated_embeddings.shape[1] != self.dim:
            raise ValueError("Embedding dimensionality mismatch in update_topics")
        alpha = float(np.clip(alpha, 0.0, 1.0))
        for k, idx in enumerate(indices):
            if not (0 <= idx < self.size):
                raise IndexError("Topic index out of range in update_topics")
            merged = (1.0 - alpha) * self.store_embeddings[idx] + alpha * updated_embeddings[k]
            self.store_embeddings[idx] = merged
            if normalize:
                self.store_embeddings[idx:idx+1] = self._l2_normalize(self.store_embeddings[idx:idx+1])
            # Update meta
            self.meta[idx].match_count += 1
            self.meta[idx].last_dataset = dataset_name

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "store_embeddings": self.store_embeddings,
            "topic_ids": self.topic_ids,
            "meta": [asdict(m) for m in self.meta],
            "gene_names": self.gene_names,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: str) -> "TopicStore":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta_list = [TopicMeta(**m) for m in payload.get("meta", [])]
        return cls(
            store_embeddings=payload.get("store_embeddings"),
            topic_ids=payload.get("topic_ids"),
            meta=meta_list,
            gene_names=payload.get("gene_names"),
        )

    # -----------------
    # Helper utilities
    # -----------------
    def _align_genes(self, new_tg: np.ndarray, new_genes: List[str], *, expand: bool = True) -> np.ndarray:
        """Align a topic-gene matrix to the store's gene order; expand columns if needed.

        Args:
            new_tg: (K, G_new) topic-gene matrix for the incoming dataset
            new_genes: length-G_new list of gene names (column order of new_tg)
            expand: if True, expand store to the union of genes

        Returns:
            new_tg_aligned: (K, G_store_after) aligned matrix
        """
        if not self.gene_names:
            # Store not initialised; set genes
            self.gene_names = list(new_genes)
            return new_tg

        # Expand store if new genes exist and expansion is enabled
        store_gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(self.gene_names)}
        missing_in_store = [g for g in new_genes if g not in store_gene_to_idx]
        if expand and missing_in_store:
            # Append new genes to store gene list
            self.gene_names.extend(missing_in_store)
            # Expand store embedding columns with zeros at the end in the same order
            n_extra = len(missing_in_store)
            extra = np.zeros((self.size, n_extra), dtype=self.store_embeddings.dtype)
            self.store_embeddings = np.hstack([self.store_embeddings, extra])
            # Refresh mapping after expansion
            store_gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        # Build aligned new matrix
        Gs = len(self.gene_names)
        aligned = np.zeros((new_tg.shape[0], Gs), dtype=new_tg.dtype)
        # Map each gene present in both to the proper column
        new_gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(new_genes)}
        shared = set(self.gene_names).intersection(new_genes)
        for g in shared:
            j_new = new_gene_to_idx[g]
            j_store = store_gene_to_idx[g]
            aligned[:, j_store] = new_tg[:, j_new]
        return aligned
