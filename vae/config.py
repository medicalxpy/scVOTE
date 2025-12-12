from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VAEConfig:
    """Configuration for the VAE embedding module."""

    n_input: int
    n_latent: int = 30
    n_hidden: int = 128
    n_layers: int = 2
    dropout_rate: float = 0.1
    gene_likelihood: str = "zinb"  # "zinb" | "nb" | "poisson"
    dispersion: str = "gene"  # "gene" | "gene-batch"
    n_batch: int = 1

    learning_rate: float = 1e-3
    max_epochs: int = 1000
    batch_size: int = 2048

    early_stopping: bool = True
    early_stopping_patience: int = 20
    min_delta: float = 1e-4

    kl_weight: float = 1.0

    seed: int = 0
    device: Optional[str] = None  # "cuda", "cpu" or None for auto
    verbose: bool = False
    check_val_every_n_epoch: int = 1
    validation_fraction: float = 0.1
    num_workers: int = 4

    def hidden_dims(self) -> List[int]:
        """Return the list of hidden dimensions for encoder/decoder."""
        return [self.n_hidden] * self.n_layers
