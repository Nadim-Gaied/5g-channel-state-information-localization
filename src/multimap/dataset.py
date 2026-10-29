"""PyTorch dataset for one clean multi-BTS DeepMIMO scenario export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.multimap.normalization import load_normalization


class MultiMapScenarioDataset(Dataset):
    """Serve padded BTS sets and one shared map without duplicating storage."""

    def __init__(
        self,
        path: str | Path,
        max_bts: int | None = None,
        random_bts_subsets: bool = False,
        minimum_bts: int = 1,
        feature_mean: torch.Tensor | np.ndarray | None = None,
        feature_std: torch.Tensor | np.ndarray | None = None,
        normalization_path: str | Path | None = None,
        map_size: tuple[int, int] | None = None,
    ) -> None:
        with np.load(path) as data:
            channels = np.asarray(data["channels"], dtype=np.complex64)
            self.channels = torch.from_numpy(
                np.stack((channels.real, channels.imag), axis=2).astype(np.float32)
            )
            self.bs_positions = torch.from_numpy(data["bs_positions"].astype(np.float32))
            self.bs_available = torch.from_numpy(data["bs_available"].astype(bool))
            self.positions = torch.from_numpy(data["positions"].astype(np.float32))
            self.map_raster = torch.from_numpy(data["map_raster"].astype(np.float32))
            self.map_bounds = torch.from_numpy(data["map_bounds"].astype(np.float32))
            self.scenario = str(data["scenario"]) if "scenario" in data else Path(path).stem
            stored_mean = torch.from_numpy(data["feature_mean"].astype(np.float32))
            stored_std = torch.from_numpy(data["feature_std"].astype(np.float32))

        if map_size is not None:
            self.map_raster = F.interpolate(
                self.map_raster.unsqueeze(0),
                size=tuple(int(value) for value in map_size),
                mode="bilinear",
                align_corners=False,
            )[0]

        if self.channels.ndim != 5 or self.channels.shape[2] != 2:
            raise ValueError("Expected CSI shape [N, B, 2, antennas, subcarriers]")
        self.num_bts = int(self.channels.shape[1])
        self.max_bts = int(max_bts or self.num_bts)
        if self.max_bts < self.num_bts:
            raise ValueError("max_bts cannot be smaller than the stored BTS count")
        if not 1 <= minimum_bts <= self.num_bts:
            raise ValueError("minimum_bts must fit inside the stored BTS count")
        self.random_bts_subsets = random_bts_subsets
        self.minimum_bts = minimum_bts

        if normalization_path is not None and (
            feature_mean is not None or feature_std is not None
        ):
            raise ValueError(
                "Use normalization_path or explicit feature_mean/feature_std, not both"
            )
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("feature_mean and feature_std must be provided together")
        if normalization_path is not None:
            selected_mean, selected_std = load_normalization(normalization_path)
            self.normalization_source = str(Path(normalization_path).resolve())
        elif feature_mean is not None and feature_std is not None:
            selected_mean, selected_std = feature_mean, feature_std
            self.normalization_source = "explicit"
        else:
            selected_mean, selected_std = stored_mean, stored_std
            self.normalization_source = "scenario-local"
        self.feature_mean = torch.as_tensor(selected_mean, dtype=torch.float32)
        self.feature_std = torch.as_tensor(selected_std, dtype=torch.float32)
        if self.feature_mean.shape != (2,) or self.feature_std.shape != (2,):
            raise ValueError("feature_mean and feature_std must have shape [2]")
        if not torch.isfinite(self.feature_mean).all() or not torch.isfinite(
            self.feature_std
        ).all() or torch.any(self.feature_std <= 0):
            raise ValueError("Normalization statistics must be finite with positive std")
        mean = self.feature_mean.view(1, 1, 2, 1, 1)
        std = self.feature_std.view(1, 1, 2, 1, 1)
        self.channels = (self.channels - mean) / std
        xmin, ymin, xmax, ymax = self.map_bounds
        span = torch.stack((xmax - xmin, ymax - ymin))
        lower = torch.stack((xmin, ymin))
        self.targets = 2.0 * (self.positions[:, :2] - lower) / span - 1.0
        bs_xy = 2.0 * (self.bs_positions[:, :2] - lower) / span - 1.0
        bs_z = self.bs_positions[:, 2:3] / 100.0
        self.normalized_bs_positions = torch.cat((bs_xy, bs_z), dim=1)

    def __len__(self) -> int:
        return len(self.channels)

    def _subset_mask(self, available: torch.Tensor) -> torch.Tensor:
        if not self.random_bts_subsets:
            return available.clone()
        candidates = torch.nonzero(available, as_tuple=False).flatten()
        lower = min(self.minimum_bts, len(candidates))
        keep = int(torch.randint(lower, len(candidates) + 1, ()).item())
        selected = candidates[torch.randperm(len(candidates))[:keep]]
        mask = torch.zeros_like(available)
        mask[selected] = True
        return mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        mask = self._subset_mask(self.bs_available[index])
        csi = torch.zeros(
            self.max_bts, *self.channels.shape[2:], dtype=self.channels.dtype
        )
        bs_positions = torch.zeros(self.max_bts, 3, dtype=torch.float32)
        bs_mask = torch.zeros(self.max_bts, dtype=torch.bool)
        csi[: self.num_bts] = self.channels[index]
        bs_positions[: self.num_bts] = self.normalized_bs_positions
        bs_mask[: self.num_bts] = mask
        csi[~bs_mask] = 0.0
        return {
            "csi": csi,
            "bs_positions": bs_positions,
            "bs_mask": bs_mask,
            "map": self.map_raster,
            "target": self.targets[index],
            "position": self.positions[index, :2],
            "map_bounds": self.map_bounds,
            "scenario": self.scenario,
        }
