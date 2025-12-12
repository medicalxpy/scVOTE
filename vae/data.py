from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None  # type: ignore


class SingleCellDataset(Dataset):
    """Dataset yielding (counts, batch_index) for each cell."""

    def __init__(
        self,
        counts,
        batch_index: np.ndarray,
    ) -> None:
        if counts is None:
            raise ValueError("counts must not be None")
        if batch_index.ndim != 1:
            raise ValueError("batch_index must be 1D")

        self.counts = counts
        self.batch_index = np.asarray(batch_index, dtype=np.int64)

        n = self.batch_index.shape[0]
        if sp is not None and sp.issparse(self.counts):
            if self.counts.shape[0] != n:
                raise ValueError(
                    f"counts has {self.counts.shape[0]} rows but batch_index has {n}"
                )
        else:
            arr = np.asarray(self.counts)
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(
                    f"counts must be 2D with {n} rows, got shape {arr.shape}"
                )
            self.counts = arr.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return self.batch_index.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if sp is not None and sp.issparse(self.counts):
            row = self.counts[idx]
            x = row.toarray().ravel().astype(np.float32, copy=False)
        else:
            x = self.counts[idx].astype(np.float32, copy=False)
        b = self.batch_index[idx]
        return torch.from_numpy(x), torch.tensor(b, dtype=torch.long)

