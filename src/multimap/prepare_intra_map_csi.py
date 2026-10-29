"""Prepare per-map CSI-only datasets on the six zero-shot test scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import spatial_tile_split
from src.experiment_config import resolve_project_path


def select_strongest_active_link(
    channels: np.ndarray, availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select one CSI matrix per UE using raw mean link power."""
    if channels.ndim != 4:
        raise ValueError("Expected channels [users, BTS, antennas, subcarriers]")
    if availability.shape != channels.shape[:2]:
        raise ValueError("Availability must have shape [users, BTS]")
    if not availability.any(axis=1).all():
        raise ValueError("Every user must have at least one active BTS")
    power = np.mean(np.abs(channels) ** 2, axis=(2, 3))
    strongest = np.argmax(np.where(availability, power, -np.inf), axis=1)
    selected = channels[np.arange(len(channels)), strongest]
    return selected.astype(np.complex64), strongest.astype(np.int16)


def prepare_scenario(
    source_path: Path,
    output_path: Path,
    *,
    seed: int,
    tile_size: tuple[float, float],
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    with np.load(source_path) as data:
        channels = np.asarray(data["channels"], dtype=np.complex64)
        availability = np.asarray(data["bs_available"], dtype=bool)
        positions = np.asarray(data["positions"], dtype=np.float32)
        scenario = str(data["scenario"]) if "scenario" in data else source_path.stem

    selected, selected_bts = select_strongest_active_link(channels, availability)
    features = np.stack((selected.real, selected.imag), axis=1).astype(np.float32)
    train_idx, val_idx, test_idx, tile_xy = spatial_tile_split(
        positions, tile_size=tile_size, seed=seed
    )

    feature_mean = features[train_idx].mean(axis=(0, 2, 3), keepdims=True)
    feature_std = np.maximum(
        features[train_idx].std(axis=(0, 2, 3), keepdims=True), 1e-12
    )
    normalized_features = (features - feature_mean) / feature_std
    xy = positions[:, :2]
    position_mean = xy[train_idx].mean(axis=0, keepdims=True)
    position_std = np.maximum(xy[train_idx].std(axis=0, keepdims=True), 1e-6)
    targets = (xy - position_mean) / position_std

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=normalized_features.astype(np.float32),
        targets=targets.astype(np.float32),
        positions=positions.astype(np.float32),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        tile_xy=tile_xy.astype(np.int32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        position_mean=position_mean.astype(np.float32),
        position_std=position_std.astype(np.float32),
        source_indices=np.arange(len(features), dtype=np.int64),
        selected_bts=selected_bts,
        channel_power=np.mean(np.abs(selected) ** 2, axis=(1, 2)).astype(np.float32),
        representation=np.asarray("real_imag"),
        tile_size=np.asarray(tile_size, dtype=np.float32),
        scenario=np.asarray(scenario),
    )
    unique_selected, selected_counts = np.unique(selected_bts, return_counts=True)
    span = xy.max(axis=0) - xy.min(axis=0)
    report = {
        "scenario": scenario,
        "source": str(source_path),
        "output": str(output_path),
        "samples": int(len(features)),
        "input_shape": list(features.shape[1:]),
        "bts_selection": "strongest_active_raw_csi_power",
        "selected_bts_counts": {
            str(int(index)): int(count)
            for index, count in zip(unique_selected, selected_counts, strict=True)
        },
        "tile_size_m": list(tile_size),
        "coverage_size_m": span.tolist(),
        "coverage_diagonal_m": float(np.linalg.norm(span)),
        "split_samples": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_tiles": {
            "train": int(len(np.unique(tile_xy[train_idx], axis=0))),
            "validation": int(len(np.unique(tile_xy[val_idx], axis=0))),
            "test": int(len(np.unique(tile_xy[test_idx], axis=0))),
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/multimap/full_city_batch.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/multimap"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/multimap/csi_only_intra_map"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=float, nargs=2, default=(10.0, 10.0))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(resolve_project_path(args.data_config).read_text(encoding="utf-8"))
    data_root = resolve_project_path(args.data_root)
    output_root = resolve_project_path(args.output_root)
    scenarios = [item["name"] for item in config["scenarios"] if item["split"] == "test"]
    reports = []
    for scenario in scenarios:
        report = prepare_scenario(
            data_root / "test" / f"{scenario}.npz",
            output_root / f"{scenario}.npz",
            seed=args.seed,
            tile_size=tuple(args.tile_size),
            overwrite=args.overwrite,
        )
        reports.append(report)
        print(
            f"{scenario}: {report['split_samples']} "
            f"coverage={report['coverage_size_m']}",
            flush=True,
        )
    (output_root / "manifest.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(reports)} scenarios in {output_root}", flush=True)


if __name__ == "__main__":
    main()
