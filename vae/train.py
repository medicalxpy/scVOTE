from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
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

        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

        n = int(self.adata.n_obs)
        train_idx, val_idx = _split_indices(n, validation_fraction, self.seed)

        train_counts = self.counts[train_idx]
        val_counts = self.counts[val_idx]
        train_b = self.batch_index[train_idx]
        val_b = self.batch_index[val_idx]

        train_ds = SingleCellDataset(train_counts, train_b)
        val_ds = SingleCellDataset(val_counts, val_b)

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
            train_loss = 0.0
            n_batches = 0

            for x, b in train_dl:
                x = x.to(self.device)
                b = b.to(self.device)

                optimizer.zero_grad()
                loss, recon, kl_z, kl_l = self.module.loss(x, b)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                n_batches += 1

            train_loss = train_loss / max(1, n_batches)

            do_val = (epoch % check_val_every_n_epoch == 0) or (epoch == max_epochs)
            val_loss = None
            if do_val:
                self.module.eval()
                with torch.no_grad():
                    vloss = 0.0
                    vb = 0
                    for x, b in val_dl:
                        x = x.to(self.device)
                        b = b.to(self.device)
                        loss, _, _, _ = self.module.loss(x, b)
                        vloss += float(loss.item())
                        vb += 1
                    val_loss = vloss / max(1, vb)

                if self.verbose:
                    print(f"[VAE] epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")

                if val_loss + min_delta < state.best_val_loss:
                    state.best_val_loss = float(val_loss)
                    state.patience_counter = 0
                    best_state = {"module": self.module.state_dict()}
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
            for x, b in dl:
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

