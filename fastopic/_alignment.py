import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


def _row_l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    nrm = torch.norm(x, p=2, dim=1, keepdim=True)
    nrm = torch.clamp(nrm, min=eps)
    return x / nrm


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    # Double-centering: Kc = K - 1/n * 1 K - 1/n * K 1 + 1/n^2 * 1 K 1
    # Implemented via row/col means to avoid forming H explicitly.
    mean_row = K.mean(dim=1, keepdim=True)
    mean_col = K.mean(dim=0, keepdim=True)
    mean_all = K.mean()
    return K - mean_row - mean_col + mean_all


@dataclass
class AlignmentConfig:
    knn_k: int = 48
    rbf_quantiles: Tuple[float, float, float, float] = (0.1, 0.3, 0.6, 0.9)
    cka_sample_n: int = 2048
    max_kernel_genes: int = 4096
    random_state: int = 42


class GeneAlignmentRef:
    """Builds reference structures (kNN Laplacian, multi-kernels) from GenePT for alignment losses.

    This class is reference-only: it loads GenePT embeddings, selects the overlap with the
    model vocabulary, normalizes rows, constructs a cosine kNN graph and multi-kernel set,
    and exposes Laplacian and CKA losses computed on the overlapped rows of current
    word embeddings.
    """

    def __init__(
        self,
        vocab: List[str],
        genept_path: str,
        config: Optional[AlignmentConfig] = None,
    ) -> None:
        self.vocab = list(vocab) if vocab is not None else []
        self.genept_path = genept_path
        self.config = config or AlignmentConfig()

        # Overlap indices and reference matrix
        self._overlap_idx: Optional[torch.Tensor] = None
        self._tilde_E: Optional[torch.Tensor] = None  # (n, d_tilde), float32 CPU tensor

        # No precomputed graph/kernels; both are built on-the-fly per call

        self._build_reference()

    # ------------------------------
    # Public API
    # ------------------------------
    def has_overlap(self) -> bool:
        return self._overlap_idx is not None and self._overlap_idx.numel() > 1

    @property
    def overlap_indices(self) -> Optional[torch.Tensor]:
        return self._overlap_idx

    def laplacian_loss(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute tr(E'^T L E') on-the-fly using a kNN graph built from the reference subset.

        Args:
            word_embeddings: (V, D) tensor on any device
        Returns:
            scalar torch.Tensor loss
        """
        if not self.has_overlap():
            return torch.as_tensor(0.0, device=word_embeddings.device)

        device = word_embeddings.device

        # Subsample overlap for tractability (reuse CKA sampling size)
        idx_all = self._overlap_idx
        n = idx_all.numel()
        sample_n = min(self.config.cka_sample_n, n)
        if sample_n < 2:
            return torch.as_tensor(0.0, device=device)
        if sample_n < n:
            g = torch.Generator(device='cpu')
            g.manual_seed(self.config.random_state)
            perm = torch.randperm(n, generator=g)[:sample_n]
        else:
            perm = torch.arange(n)

        # Current E' subset and reference subset X
        idx = idx_all.index_select(0, perm.to(idx_all.device))
        E = word_embeddings.index_select(0, idx.to(device))
        E = _row_l2_normalize(E)
        X_ref = self._tilde_E.index_select(0, perm.to(self._tilde_E.device))

        if device.type == "cuda":
            # GPU path: build dense kNN graph in PyTorch on the current device
            W = self._build_knn_w_torch(X_ref.to(device), self.config.knn_k)

            # Degree vector d_i = sum_j w_ij (W is symmetric)
            d = W.sum(dim=1)

            # Compute sum_i d_i ||e_i||^2
            part1 = (d * (E * E).sum(dim=1)).sum()

            # Compute sum_{ij} w_ij <e_i, e_j>
            dot_ij_mat = E @ E.t()
            part2 = (W * dot_ij_mat).sum()
        else:
            # CPU path: retain previous sklearn + scipy implementation
            X_np = X_ref.numpy()

            # Build kNN graph W on-the-fly for this subset
            W_csr = self._build_knn_w(X_np, self.config.knn_k)
            W_coo = self._csr_to_coo_torch(W_csr, device=device)
            # Ensure coalesced for indices() access
            W_coo = W_coo.coalesce()
            rows = W_coo.indices()[0]
            cols = W_coo.indices()[1]
            vals = W_coo.values()

            # Degree vector d_i = sum_j w_ij (W is symmetric)
            ns = int(W_csr.shape[0])
            d = torch.zeros(ns, device=device, dtype=E.dtype)
            d.index_add_(0, rows, vals)

            # Compute sum_i d_i ||e_i||^2
            part1 = (d * (E * E).sum(dim=1)).sum()

            # Compute sum_{ij} w_ij <e_i, e_j>
            dot_ij = (E.index_select(0, rows) * E.index_select(0, cols)).sum(dim=1)
            part2 = (vals * dot_ij).sum()

        loss = part1 - part2
        return loss

    def cka_loss(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """CKA structural alignment loss between current E' and reference kernels.

        Uses a subset if overlap is large for tractability.
        """
        if not self.has_overlap():
            return torch.as_tensor(0.0, device=word_embeddings.device)

        device = word_embeddings.device
        idx_all = self._overlap_idx
        n = idx_all.numel()

        # Subsample for tractability
        sample_n = min(self.config.cka_sample_n, n)
        if sample_n < 2:
            return torch.as_tensor(0.0, device=device)
        if sample_n < n:
            g = torch.Generator(device='cpu')
            g.manual_seed(self.config.random_state)
            perm = torch.randperm(n, generator=g)[:sample_n]
        else:
            perm = torch.arange(n)

        # Kernels will be computed on-the-fly for this subset; no template restriction needed

        idx = idx_all.index_select(0, perm.to(idx_all.device))

        # Current E' subset (row-normalized)
        E = word_embeddings.index_select(0, idx.to(device))
        E = _row_l2_normalize(E)
        G = E @ E.t()
        G = _center_gram(G)

        # Reference kernels for the same subset (computed on-the-fly)
        K_list = self._make_kernels_for_indices(perm, device=device)
        if not K_list:
            return torch.as_tensor(0.0, device=device)

        # CKA similarity per kernel -> loss = -mean(sim)
        sims = []
        eps = 1e-8
        G_norm = torch.norm(G, p='fro')
        if float(G_norm.item()) <= eps:
            return torch.as_tensor(0.0, device=device)
        for K in K_list:
            # Move K to current device for computation
            Kd = K.to(device)
            Knorm = torch.norm(Kd, p='fro')
            if float(Knorm.item()) <= eps:
                continue
            sim = torch.sum(G * Kd) / (G_norm * Knorm)
            sims.append(sim)
        if not sims:
            return torch.as_tensor(0.0, device=device)
        loss = -torch.stack(sims).mean()
        return loss

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _build_reference(self) -> None:
        # Load GenePT dict
        with open(self.genept_path, 'rb') as f:
            genept_dict = pickle.load(f)

        # Overlap in original order of vocab
        overlap_idx: List[int] = []
        emb_list: List[np.ndarray] = []
        for i, g in enumerate(self.vocab):
            if g in genept_dict:
                vec = genept_dict[g]
                if isinstance(vec, list):
                    vec = np.asarray(vec, dtype=np.float32)
                else:
                    vec = np.asarray(vec, dtype=np.float32)
                if vec.ndim != 1:
                    continue
                overlap_idx.append(i)
                emb_list.append(vec)

        if not emb_list:
            self._overlap_idx = None
            self._tilde_E = None
            return

        tilde_E = torch.as_tensor(np.vstack(emb_list), dtype=torch.float32)
        tilde_E = _row_l2_normalize(tilde_E)
        self._tilde_E = tilde_E.cpu()
        self._overlap_idx = torch.as_tensor(overlap_idx, dtype=torch.long)

        # No graph/kernel precomputation: both are built on-the-fly in loss calls

    def _build_knn_w(self, X: np.ndarray, k: int) -> sp.csr_matrix:
        # X is row-normalized; cosine distance = 1 - cosine similarity
        n = X.shape[0]
        k_eff = min(k + 1, max(2, n))  # include self, ensure at least 2
        nn = NearestNeighbors(n_neighbors=k_eff, metric='cosine')
        nn.fit(X)
        dists, inds = nn.kneighbors(X, return_distance=True)

        # Convert to similarity; skip self by masking first neighbor if it is self
        rows = []
        cols = []
        vals = []
        for i in range(n):
            for j_idx, j in enumerate(inds[i]):
                if i == j:
                    continue
                sim = 1.0 - float(dists[i, j_idx])
                if sim <= 0.0:
                    continue
                rows.append(i)
                cols.append(int(j))
                vals.append(sim)

        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
        # Symmetrize by max
        W = W.maximum(W.T)
        return W

    # Removed kernel template precomputation; see _make_kernels_for_indices

    def _estimate_rbf_gammas_torch(self, X: torch.Tensor) -> List[float]:
        """Estimate RBF gammas from pairwise squared distances using PyTorch.

        Works on both CPU and GPU tensors.
        """
        n = X.shape[0]
        s = min(2000, n)
        if s < 2:
            return []
        # Sample subset indices deterministically
        g = torch.Generator(device="cpu")
        g.manual_seed(self.config.random_state)
        idx = torch.randperm(n, generator=g)[:s].to(X.device)
        Xs = X.index_select(0, idx)
        G = Xs @ Xs.t()
        sq = (Xs * Xs).sum(dim=1, keepdim=True)
        D2 = sq + sq.t() - 2.0 * G
        iu = torch.triu_indices(s, s, offset=1, device=D2.device)
        vals = D2[iu[0], iu[1]]
        if vals.numel() == 0:
            return []
        qs = torch.tensor(self.config.rbf_quantiles, device=vals.device, dtype=vals.dtype)
        quantiles = torch.quantile(vals, qs)
        gammas = [float(max(q.item(), 1e-6)) for q in quantiles]
        return gammas

    def _get_kernel_templates_for_subset(self, perm_subset: torch.Tensor) -> List[torch.Tensor]:
        """Deprecated API kept for compatibility; kernels are built on-the-fly."""
        return []

    def _make_kernels_for_indices(self, perm_subset: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
        """Compute cosine and multi-scale RBF kernels for a given overlap subset on-the-fly.

        Args:
            perm_subset: indices (0..n_overlap-1) selecting rows from self._tilde_E
            device: device where kernels should live
        Returns:
            List of centered torch kernels on the given device.
        """
        if self._tilde_E is None or perm_subset.numel() < 2:
            return []
        # Gather subset from normalized reference embeddings and move to target device
        X = self._tilde_E.index_select(0, perm_subset.to(self._tilde_E.device)).to(device)
        X = _row_l2_normalize(X)
        # Cosine kernel (X rows are normalized)
        K_cos = _center_gram(X @ X.t())
        # RBF kernels using quantile-based gammas estimated on the same subset
        gammas = self._estimate_rbf_gammas_torch(X)
        K_rbfs: List[torch.Tensor] = []
        if gammas:
            sq = (X * X).sum(dim=1, keepdim=True)
            D2 = sq + sq.t() - 2.0 * (X @ X.t())
            for g in gammas:
                K = torch.exp(-D2 / float(g))
                K = _center_gram(K)
                K_rbfs.append(K)
        return [K_cos] + K_rbfs

    @staticmethod
    def _csr_to_coo_torch(W: sp.csr_matrix, device: torch.device) -> torch.Tensor:
        W_coo = W.tocoo()
        indices = torch.as_tensor(np.vstack([W_coo.row, W_coo.col]), dtype=torch.long, device=device)
        values = torch.as_tensor(W_coo.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, size=W_coo.shape, device=device)

    @staticmethod
    def _build_knn_w_torch(X: torch.Tensor, k: int) -> torch.Tensor:
        """Build cosine kNN graph with PyTorch only (dense similarity matrix).

        Args:
            X: (n, d) row-normalized reference embeddings on target device.
            k: number of neighbors (excluding self).
        Returns:
            Dense (n, n) similarity matrix W on the same device.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D tensor")
        n = X.shape[0]
        if n == 0:
            return torch.zeros((0, 0), dtype=X.dtype, device=X.device)

        # Cosine similarity
        S = X @ X.t()

        # Mask self-similarity so it is not selected as neighbor
        diag_mask = torch.eye(n, dtype=torch.bool, device=X.device)
        S = S.masked_fill(diag_mask, -1e9)

        k_eff = int(min(max(2, n), k + 1))
        if k_eff <= 0:
            return torch.zeros((n, n), dtype=X.dtype, device=X.device)

        top_vals, top_idx = torch.topk(S, k_eff, dim=1)
        top_vals = torch.clamp(top_vals, min=0.0)

        W = torch.zeros_like(S)
        W.scatter_(1, top_idx, top_vals)

        # Symmetrize by max to match CPU behavior
        W = torch.maximum(W, W.t())
        return W
