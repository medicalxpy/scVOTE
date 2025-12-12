from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None  # type: ignore

from .config import VAEConfig
from .data import SingleCellDataset
from .model import VAE
from .utils import get_device, set_seed


def _as_csr(x):
    if sp is None:
        return np.asarray(x)
    if sp.issparse(x):
        return x.tocsr()
    return np.asarray(x)


def _row_sums(counts) -> np.ndarray:
    if sp is not None and sp.issparse(counts):
        s = np.asarray(counts.sum(axis=1)).ravel()
        return s.astype(np.float32)
    return np.asarray(counts, dtype=np.float32).sum(axis=1)


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    n_val = int(math.floor(n * val_fraction))
    n_val = max(1, min(n - 1, n_val)) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def _batch_index_from_obs(adata, batch_key: Optional[str]) -> Tuple[np.ndarray, int]:
    n = adata.n_obs
    if not batch_key:
        return np.zeros(n, dtype=np.int64), 1
    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
    cats = adata.obs[batch_key].astype("category")
    codes = cats.cat.codes.to_numpy().astype(np.int64)
    n_batch = int(cats.cat.categories.size)
    return codes, max(1, n_batch)


def setup_anndata(adata, *, layer: str = "counts", batch_key: Optional[str] = None, labels_key: Optional[str] = None) -> None:
    """Validate that required fields exist for training."""
    if layer not in adata.layers and layer != "X":
        raise KeyError(f"layer '{layer}' not found in adata.layers")
    if batch_key and batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
    if labels_key and labels_key not in adata.obs.columns:
        raise KeyError(f"labels_key '{labels_key}' not found in adata.obs")


@dataclass
class TrainState:
    best_val_loss: float = float("inf")
    patience_counter: int = 0


def _load_genept_embeddings(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict at {path}, got {type(data)}")
    out: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        try:
            vec = np.asarray(v, dtype=np.float32)
        except Exception:
            continue
        if vec.ndim != 1:
            continue
        out[str(k)] = vec
    return out


def _compute_weighted_cell_embeddings(
    counts,
    gene_names: np.ndarray,
    gene_embeddings: Dict[str, np.ndarray],
    *,
    chunk_size: int = 4096,
    eps: float = 1e-8,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    matched_idx = []
    matched_vecs = []
    ref_dim: Optional[int] = None

    for i, g in enumerate(gene_names.tolist()):
        if g not in gene_embeddings:
            continue
        vec = gene_embeddings[g]
        if ref_dim is None:
            ref_dim = int(vec.shape[0])
        if int(vec.shape[0]) != ref_dim:
            continue
        matched_idx.append(i)
        matched_vecs.append(vec)

    if not matched_idx or ref_dim is None:
        raise ValueError("No genes could be matched to the gene embedding dictionary.")

    e_matched = np.stack(matched_vecs, axis=0).astype(np.float32, copy=False)
    matched_idx_arr = np.asarray(matched_idx, dtype=np.int64)

    n_cells = int(counts.shape[0])
    out = np.zeros((n_cells, ref_dim), dtype=np.float32)
    valid = np.zeros((n_cells,), dtype=bool)

    if verbose:
        print(f"[VAE] gene-embedding match: {len(matched_idx_arr)}/{len(gene_names)} genes")

    for start in range(0, n_cells, chunk_size):
        end = min(n_cells, start + chunk_size)
        chunk = counts[start:end]
        if sp is not None and sp.issparse(chunk):
            chunk = chunk.tocsr()[:, matched_idx_arr]
            denom = np.asarray(chunk.sum(axis=1)).ravel().astype(np.float32, copy=False)
            numer = chunk @ e_matched
            numer = np.asarray(numer, dtype=np.float32)
        else:
            chunk = np.asarray(chunk, dtype=np.float32)[:, matched_idx_arr]
            denom = chunk.sum(axis=1).astype(np.float32, copy=False)
            numer = chunk @ e_matched

        ok = denom > 0
        if np.any(ok):
            out[start:end][ok] = numer[ok] / (denom[ok, None] + eps)
            valid[start:end][ok] = True

    return out, valid


class VAEModel:
    """Trainable model wrapper with a familiar API."""

    def __init__(
        self,
        adata,
        *,
        layer: str = "counts",
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        n_latent: int = 30,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "zinb",
        dispersion: str = "gene",
        genept_path: str = "GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle",
        genept_loss_weight: float = 1e-3,
        genept_proj_dim: int = 256,
        genept_proj_hidden: int = 1024,
        seed: int = 0,
        verbose: bool = False,
        device: Optional[str] = None,
    ) -> None:
        setup_anndata(adata, layer=layer, batch_key=batch_key, labels_key=labels_key)

        self.adata = adata
        self.layer = layer
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.seed = seed
        self.verbose = verbose

        counts = adata.layers[layer] if layer != "X" else adata.X
        self.counts = _as_csr(counts)
        self.batch_index, n_batch = _batch_index_from_obs(adata, batch_key)

        self.config = VAEConfig(
            n_input=int(adata.n_vars),
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            dispersion=dispersion,
            n_batch=n_batch,
            seed=seed,
            verbose=verbose,
            device=device,
        )

        self.device = get_device(device)
        set_seed(seed)

        self.module = VAE(self.config).to(self.device)
        self._is_trained = False

        self._init_library_priors()
        self.genept_path = genept_path
        self.genept_loss_weight = float(genept_loss_weight)
        self.genept_proj_dim = int(genept_proj_dim)
        self.genept_proj_hidden = int(genept_proj_hidden)

    @staticmethod
    def setup_anndata(adata, **kwargs) -> None:
        setup_anndata(adata, **kwargs)

    def _init_library_priors(self) -> None:
        sums = _row_sums(self.counts)
        log_lib = np.log(sums + 1.0).astype(np.float32)

        n_batch = self.config.n_batch
        means = np.zeros(n_batch, dtype=np.float32)
        vars_ = np.ones(n_batch, dtype=np.float32)

        for b in range(n_batch):
            mask = self.batch_index == b
            if not np.any(mask):
                continue
            vals = log_lib[mask]
            means[b] = float(np.mean(vals))
            vars_[b] = float(np.var(vals) + 1e-4)

        means_t = torch.tensor(means, dtype=torch.float32, device=self.device)
        vars_t = torch.tensor(vars_, dtype=torch.float32, device=self.device)
        self.module.set_library_priors(means_t, vars_t)

    def train(
        self,
        *,
        max_epochs: int = 1000,
        batch_size: int = 2048,
        plan_kwargs: Optional[Dict[str, Any]] = None,
        accelerator: Optional[str] = None,
        devices: Optional[str] = None,
        check_val_every_n_epoch: int = 1,
        early_stopping: bool = True,
        early_stopping_patience: int = 20,
        datasplitter_kwargs: Optional[Dict[str, Any]] = None,
        validation_fraction: float = 0.1,
        min_delta: float = 1e-4,
    ) -> None:
        lr = 1e-3
        if plan_kwargs and "lr" in plan_kwargs:
            lr = float(plan_kwargs["lr"])

        num_workers = 4
        if datasplitter_kwargs and "num_workers" in datasplitter_kwargs:
            num_workers = int(datasplitter_kwargs["num_workers"])

        self.config.learning_rate = lr
        self.config.max_epochs = max_epochs
        self.config.batch_size = batch_size
        self.config.early_stopping = early_stopping
        self.config.early_stopping_patience = early_stopping_patience
        self.config.check_val_every_n_epoch = check_val_every_n_epoch
        self.config.validation_fraction = validation_fraction
        self.config.min_delta = min_delta
        self.config.num_workers = num_workers

        genept_targets = None
        genept_valid = None
        genept_cell_projector: Optional[nn.Module] = None
        genept_gene_projector: Optional[nn.Module] = None

        if self.genept_loss_weight > 0 and self.genept_path:
            if os.path.exists(self.genept_path):
                try:
                    gene_embeddings = _load_genept_embeddings(self.genept_path)
                    gene_names = np.asarray(self.adata.var_names.tolist(), dtype=str)
                    genept_targets_np, genept_valid_np = _compute_weighted_cell_embeddings(
                        self.counts,
                        gene_names,
                        gene_embeddings,
                        verbose=self.verbose,
                    )
                    genept_targets = torch.from_numpy(genept_targets_np)
                    genept_valid = torch.from_numpy(genept_valid_np)

                    in_dim = int(genept_targets.shape[1])
                    genept_gene_projector = nn.Linear(in_dim, self.genept_proj_dim).to(self.device)
                    genept_cell_projector = nn.Sequential(
                        nn.Linear(self.config.n_latent, self.genept_proj_hidden),
                        nn.ReLU(),
                        nn.Linear(self.genept_proj_hidden, self.genept_proj_dim),
                    ).to(self.device)
                except Exception as e:
                    if self.verbose:
                        print(f"[VAE] gene-embedding alignment disabled: {e}")
                    genept_targets = None
                    genept_valid = None
            else:
                if self.verbose:
                    print(f"[VAE] gene-embedding file not found: {self.genept_path}")

        params = list(self.module.parameters())
        if genept_cell_projector is not None:
            params += list(genept_cell_projector.parameters())
        if genept_gene_projector is not None:
            params += list(genept_gene_projector.parameters())

        optimizer = torch.optim.Adam(params, lr=lr)

        n = int(self.adata.n_obs)
        train_idx, val_idx = _split_indices(n, validation_fraction, self.seed)

        train_counts = self.counts[train_idx]
        val_counts = self.counts[val_idx]
        train_b = self.batch_index[train_idx]
        val_b = self.batch_index[val_idx]

        train_ds = SingleCellDataset(train_counts, train_b, cell_indices=train_idx)
        val_ds = SingleCellDataset(val_counts, val_b, cell_indices=val_idx)

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        state = TrainState()
        best_state: Optional[Dict[str, Any]] = None

        for epoch in range(1, max_epochs + 1):
            self.module.train()
            if genept_cell_projector is not None:
                genept_cell_projector.train()
            if genept_gene_projector is not None:
                genept_gene_projector.train()
            train_loss = 0.0
            n_batches = 0

            for x, b, cell_idx in train_dl:
                x = x.to(self.device)
                b = b.to(self.device)

                optimizer.zero_grad()
                base_loss, recon, kl_z, kl_l, z_mu = self.module.loss(x, b)
                loss = base_loss
                if (
                    genept_targets is not None
                    and genept_valid is not None
                    and genept_cell_projector is not None
                    and genept_gene_projector is not None
                ):
                    with torch.no_grad():
                        tgt = genept_targets[cell_idx].to(self.device)
                        ok = genept_valid[cell_idx].to(self.device)
                    if ok.any().item():
                        z_proj = genept_cell_projector(z_mu[ok])
                        g_proj = genept_gene_projector(tgt[ok])
                        z_proj = F.normalize(z_proj, dim=1, eps=1e-8)
                        g_proj = F.normalize(g_proj, dim=1, eps=1e-8)
                        align = 1.0 - torch.sum(z_proj * g_proj, dim=1)
                        align_loss = align.mean()
                        loss = loss + self.genept_loss_weight * align_loss
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                n_batches += 1

            train_loss = train_loss / max(1, n_batches)

            do_val = (epoch % check_val_every_n_epoch == 0) or (epoch == max_epochs)
            val_loss = None
            if do_val:
                self.module.eval()
                if genept_cell_projector is not None:
                    genept_cell_projector.eval()
                if genept_gene_projector is not None:
                    genept_gene_projector.eval()
                with torch.no_grad():
                    vloss = 0.0
                    vb = 0
                    for x, b, cell_idx in val_dl:
                        x = x.to(self.device)
                        b = b.to(self.device)
                        base_loss, _, _, _, z_mu = self.module.loss(x, b)
                        loss = base_loss
                        if (
                            genept_targets is not None
                            and genept_valid is not None
                            and genept_cell_projector is not None
                            and genept_gene_projector is not None
                        ):
                            tgt = genept_targets[cell_idx].to(self.device)
                            ok = genept_valid[cell_idx].to(self.device)
                            if ok.any().item():
                                z_proj = genept_cell_projector(z_mu[ok])
                                g_proj = genept_gene_projector(tgt[ok])
                                z_proj = F.normalize(z_proj, dim=1, eps=1e-8)
                                g_proj = F.normalize(g_proj, dim=1, eps=1e-8)
                                align = 1.0 - torch.sum(z_proj * g_proj, dim=1)
                                align_loss = align.mean()
                                loss = loss + self.genept_loss_weight * align_loss
                        vloss += float(loss.item())
                        vb += 1
                    val_loss = vloss / max(1, vb)

                if self.verbose:
                    print(f"[VAE] epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")

                if val_loss + min_delta < state.best_val_loss:
                    state.best_val_loss = float(val_loss)
                    state.patience_counter = 0
                    best_state = {
                        "module": self.module.state_dict(),
                        "genept_cell_projector": genept_cell_projector.state_dict() if genept_cell_projector is not None else None,
                        "genept_gene_projector": genept_gene_projector.state_dict() if genept_gene_projector is not None else None,
                    }
                else:
                    state.patience_counter += 1

                if early_stopping and state.patience_counter >= early_stopping_patience:
                    if self.verbose:
                        print(f"[VAE] early stop at epoch {epoch}")
                    break
            else:
                if self.verbose:
                    print(f"[VAE] epoch={epoch:03d} train={train_loss:.4f}")

        if best_state is not None:
            self.module.load_state_dict(best_state["module"])
            if genept_cell_projector is not None and best_state.get("genept_cell_projector") is not None:
                genept_cell_projector.load_state_dict(best_state["genept_cell_projector"])
            if genept_gene_projector is not None and best_state.get("genept_gene_projector") is not None:
                genept_gene_projector.load_state_dict(best_state["genept_gene_projector"])

        self._is_trained = True

    def get_latent_representation(
        self,
        *,
        give_mean: bool = True,
        batch_size: int = 2048,
    ) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        ds = SingleCellDataset(self.counts, self.batch_index)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=self.config.num_workers)

        self.module.eval()
        out = []
        with torch.no_grad():
            for x, b, _ in dl:
                x = x.to(self.device)
                b = b.to(self.device)
                z_mu, z_logvar, _, _, _, _, _ = self.module.forward(x, b, give_mean=False)
                if give_mean:
                    z = z_mu
                else:
                    std = torch.exp(0.5 * z_logvar)
                    eps = torch.randn_like(std)
                    z = z_mu + eps * std
                out.append(z.detach().cpu().numpy())

        return np.concatenate(out, axis=0).astype(np.float32)
