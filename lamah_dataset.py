from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class LamaHCEDataset(Dataset):
    """Sliding-window PyTorch Dataset for preprocessed LamaH-CE tensors."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        input_len: int,
        horizons: Iterable[int],
        target_mode: str = "raw",
        return_static: bool = True,
        return_adj: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.input_len = int(input_len)
        self.horizons = np.asarray(list(horizons), dtype=np.int64)
        self.target_mode = target_mode
        self.return_static = return_static
        self.return_adj = return_adj

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        if self.input_len <= 0:
            raise ValueError(f"input_len must be positive, got {input_len}")
        if self.horizons.size == 0 or np.any(self.horizons <= 0):
            raise ValueError(f"horizons must contain positive integers, got {self.horizons.tolist()}")
        if self.target_mode not in {"raw", "normalized"}:
            raise ValueError(f"target_mode must be 'raw' or 'normalized', got {target_mode!r}")

        self.X = self._load_required("X.npy").astype(np.float32)
        self.Q = self._load_required("Q.npy").astype(np.float32)
        self.mask = self._load_required("mask.npy").astype(bool)
        self.feature_mask = self._load_required("feature_mask.npy").astype(bool)
        self.dates = self._load_required("dates.npy", allow_pickle=True).astype(str)
        self.features = self._load_required("features.npy", allow_pickle=True).astype(str)
        self.S = self._load_required("S.npy").astype(np.float32) if return_static else None
        self.A = self._load_required("A.npy").astype(np.float32) if return_adj else None

        split_file = self.processed_dir / "split_indices.npz"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        splits = np.load(split_file)
        key = f"{self.split}_indices"
        if key not in splits:
            raise KeyError(f"Missing {key} in {split_file}")
        self.split_indices = splits[key].astype(np.int64)

        self._validate_shapes()
        self.samples = self._make_samples()
        if not self.samples:
            raise ValueError(
                f"No samples for split={split}, input_len={input_len}, horizons={self.horizons.tolist()}. "
                "Reduce input_len/horizons or check split size."
            )

        if target_mode == "normalized":
            q_matches = np.flatnonzero(self.features == "qobs")
            if q_matches.size == 0:
                raise ValueError("target_mode='normalized' requires qobs in features.npy")
            self.q_feature_idx = int(q_matches[0])
        else:
            self.q_feature_idx = None

    def _load_required(self, filename: str, allow_pickle: bool = False) -> np.ndarray:
        path = self.processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required processed file: {path}")
        return np.load(path, allow_pickle=allow_pickle)

    def _validate_shapes(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X.npy must have shape [T, N, F], got {self.X.shape}")
        t, n, _ = self.X.shape
        if self.Q.shape != (t, n):
            raise ValueError(f"Q.npy must have shape {(t, n)}, got {self.Q.shape}")
        if self.mask.shape != (t, n):
            raise ValueError(f"mask.npy must have shape {(t, n)}, got {self.mask.shape}")
        if self.feature_mask.shape != self.X.shape:
            raise ValueError(f"feature_mask.npy must have shape {self.X.shape}, got {self.feature_mask.shape}")
        if self.dates.shape[0] != t:
            raise ValueError(f"dates.npy length must be {t}, got {self.dates.shape[0]}")
        if self.S is not None and self.S.shape[0] != n:
            raise ValueError(f"S.npy first dimension must be N={n}, got {self.S.shape}")
        if self.A is not None and self.A.shape != (n, n):
            raise ValueError(f"A.npy must have shape {(n, n)}, got {self.A.shape}")

    def _make_samples(self) -> list[int]:
        split_set = set(int(i) for i in self.split_indices.tolist())
        max_horizon = int(self.horizons.max())
        samples: list[int] = []
        for end_idx in self.split_indices:
            end_idx = int(end_idx)
            input_start = end_idx - self.input_len + 1
            if input_start < 0 or end_idx + max_horizon >= self.X.shape[0]:
                continue
            input_indices = range(input_start, end_idx + 1)
            target_indices = [end_idx + int(h) for h in self.horizons]
            if all(i in split_set for i in input_indices) and all(i in split_set for i in target_indices):
                samples.append(end_idx)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str]]:
        end_idx = self.samples[idx]
        input_slice = slice(end_idx - self.input_len + 1, end_idx + 1)
        target_indices = np.asarray([end_idx + int(h) for h in self.horizons], dtype=np.int64)

        x = torch.from_numpy(self.X[input_slice]).float()
        x_mask = torch.from_numpy(self.feature_mask[input_slice].astype(np.float32))
        if self.target_mode == "normalized":
            y_np = self.X[target_indices, :, self.q_feature_idx]
        else:
            y_np = self.Q[target_indices]
        y = torch.from_numpy(y_np.T.astype(np.float32))
        y_mask = torch.from_numpy(self.mask[target_indices].T.astype(bool))

        item: dict[str, torch.Tensor | list[str]] = {
            "x": x,
            "x_mask": x_mask,
            "y": y,
            "y_mask": y_mask,
            "dates": [str(self.dates[i]) for i in target_indices],
        }
        if self.return_static:
            item["static"] = torch.from_numpy(self.S).float()
        if self.return_adj:
            item["A"] = torch.from_numpy(self.A).float()
        return item
