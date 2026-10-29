"""Training-only global normalization for multi-map complex CSI datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class RunningComplexStats:
    """Numerically stable real/imaginary statistics merged block by block."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if not np.iscomplexobj(values):
            raise ValueError("Expected complex CSI values")
        block_count = int(values.size)
        if block_count == 0:
            return
        block_mean = np.asarray(
            [values.real.mean(dtype=np.float64), values.imag.mean(dtype=np.float64)],
            dtype=np.float64,
        )
        block_m2 = np.asarray(
            [
                np.square(values.real - block_mean[0], dtype=np.float64).sum(),
                np.square(values.imag - block_mean[1], dtype=np.float64).sum(),
            ],
            dtype=np.float64,
        )
        if self.count == 0:
            self.count = block_count
            self.mean = block_mean
            self.m2 = block_m2
            return

        assert self.mean is not None and self.m2 is not None
        total = self.count + block_count
        delta = block_mean - self.mean
        self.m2 += block_m2 + delta**2 * self.count * block_count / total
        self.mean += delta * block_count / total
        self.count = total

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("Cannot calculate normalization from zero CSI values")
        variance = self.m2 / self.count
        std = np.sqrt(np.maximum(variance, 1e-24))
        return self.mean.astype(np.float32), std.astype(np.float32)


def scenario_feature_stats(path: str | Path) -> tuple[RunningComplexStats, int]:
    """Accumulate one scenario while excluding unavailable BTS links."""
    stats = RunningComplexStats()
    with np.load(path) as data:
        channels = np.asarray(data["channels"], dtype=np.complex64)
        availability = np.asarray(data["bs_available"], dtype=bool)
        if channels.ndim != 4:
            raise ValueError(f"Expected channels [N,B,A,K], got {channels.shape}")
        if availability.shape != channels.shape[:2]:
            raise ValueError(
                f"Availability shape {availability.shape} does not match {channels.shape[:2]}"
            )
        for station in range(channels.shape[1]):
            stats.update(channels[availability[:, station], station])
    return stats, int(availability.sum())


def _merge(target: RunningComplexStats, source: RunningComplexStats) -> None:
    if source.count == 0:
        return
    assert source.mean is not None and source.m2 is not None
    if target.count == 0:
        target.count = source.count
        target.mean = source.mean.copy()
        target.m2 = source.m2.copy()
        return
    assert target.mean is not None and target.m2 is not None
    total = target.count + source.count
    delta = source.mean - target.mean
    target.m2 += source.m2 + delta**2 * target.count * source.count / total
    target.mean += delta * source.count / total
    target.count = total


def compute_global_feature_stats(paths: Iterable[str | Path]) -> dict:
    """Compute global CSI statistics and per-scenario audit information."""
    global_stats = RunningComplexStats()
    scenarios = []
    for value in paths:
        path = Path(value)
        local_stats, active_links = scenario_feature_stats(path)
        local_mean, local_std = local_stats.result()
        _merge(global_stats, local_stats)
        scenarios.append(
            {
                "path": str(path.resolve()),
                "active_links": active_links,
                "scalar_values_per_component": local_stats.count,
                "feature_mean": local_mean.tolist(),
                "feature_std": local_std.tolist(),
            }
        )
    feature_mean, feature_std = global_stats.result()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "active BTS links from training maps only",
        "representation": "real_imag",
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "scalar_values_per_component": global_stats.count,
        "active_links": sum(item["active_links"] for item in scenarios),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def save_normalization(report: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        feature_mean=np.asarray(report["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(report["feature_std"], dtype=np.float32),
        scenario_count=np.asarray(report["scenario_count"], dtype=np.int32),
        active_links=np.asarray(report["active_links"], dtype=np.int64),
        scalar_values_per_component=np.asarray(
            report["scalar_values_per_component"], dtype=np.int64
        ),
        representation=np.asarray(report["representation"]),
        policy=np.asarray(report["policy"]),
    )
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        mean = np.asarray(data["feature_mean"], dtype=np.float32)
        std = np.asarray(data["feature_std"], dtype=np.float32)
        representation = str(data["representation"])
    if representation != "real_imag":
        raise ValueError(f"Unsupported normalization representation: {representation}")
    if mean.shape != (2,) or std.shape != (2,):
        raise ValueError("Normalization mean/std must have shape [2]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalization statistics must be finite with positive std")
    return mean, std
