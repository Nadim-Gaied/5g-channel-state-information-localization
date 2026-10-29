"""Build one clean, aligned multi-BTS scenario export from DeepMIMO v4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import deepmimo as dm
import numpy as np

from src.experiment_config import resolve_project_path
from src.multimap.map_raster import centered_bounds, rasterize_scene


CHANNEL_MATRICES = [
    "rx_pos",
    "tx_pos",
    "power",
    "phase",
    "delay",
    "aoa_az",
    "aoa_el",
    "aod_az",
    "aod_el",
]


def _transmitter_sets(config: dict[str, Any], scenario_root: Path) -> list[int]:
    configured = config["tx_sets"]
    if configured != "all":
        return [int(value) for value in configured]
    params = json.loads(
        (scenario_root / str(config["scenario"]) / "params.json").read_text(
            encoding="utf-8"
        )
    )
    tx_sets = [
        int(settings["id"])
        for settings in params["txrx_sets"].values()
        if bool(settings["is_tx"])
    ]
    if not tx_sets:
        raise ValueError("The scenario does not contain a transmitter set")
    return sorted(tx_sets)


def _receiver_set(config: dict[str, Any], scenario_root: Path) -> int:
    configured = config.get("rx_set", "auto")
    if configured != "auto":
        return int(configured)
    params = json.loads(
        (scenario_root / str(config["scenario"]) / "params.json").read_text(
            encoding="utf-8"
        )
    )
    rx_sets = [
        int(settings["id"])
        for settings in params["txrx_sets"].values()
        if bool(settings["is_rx"]) and not bool(settings["is_tx"])
    ]
    if len(rx_sets) != 1:
        raise ValueError(
            "Automatic receiver-set selection requires exactly one RX-only set; "
            f"found {rx_sets}"
        )
    return rx_sets[0]


def _channel_parameters(settings: dict[str, Any]) -> dm.ChannelParameters:
    selected = np.arange(int(settings["selected_subcarriers"]), dtype=np.int64)
    return dm.ChannelParameters(
        num_paths=int(settings["num_paths"]),
        bs_antenna={
            "shape": [int(value) for value in settings["antenna_shape"]],
            "spacing": float(settings["antenna_spacing"]),
        },
        ue_antenna={"shape": [1, 1], "spacing": 0.5},
        ofdm={
            "subcarriers": int(settings["num_ofdm"]),
            "selected_subcarriers": selected,
            "bandwidth": float(settings["bandwidth_hz"]),
            "rx_filter": 0,
        },
    )


def build_scenario(config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    output_path = resolve_project_path(config["output_path"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    scenario_root = resolve_project_path(config["scenario_root"])
    scenario_name = str(config["scenario"])
    if not (scenario_root / scenario_name).is_dir():
        raise FileNotFoundError(f"Scenario not found: {scenario_root / scenario_name}")
    dm.config.set("scenarios_folder", str(scenario_root))

    tx_sets = _transmitter_sets(config, scenario_root)
    tx_index = int(config.get("tx_index", 0))
    rx_set = _receiver_set(config, scenario_root)
    max_paths = int(config["radio"]["num_paths"])
    path_counts = []
    receiver_positions = None
    bs_positions = []
    scene = None
    for tx_set in tx_sets:
        dataset = dm.load(
            scenario_name,
            tx_sets={tx_set: [tx_index]},
            rx_sets={rx_set: "all"},
            matrices=["rx_pos", "tx_pos", "aoa_az"],
            max_paths=max_paths,
        )
        positions = np.asarray(dataset.rx_pos, dtype=np.float32)
        if receiver_positions is None:
            receiver_positions = positions
            scene = dataset.scene
        elif not np.allclose(receiver_positions, positions):
            raise ValueError("Receiver grids are not aligned across BTS pairs")
        path_counts.append(np.asarray(dataset.num_paths, dtype=np.int16))
        tx_positions = np.asarray(dataset.tx_pos, dtype=np.float32).reshape(-1, 3)
        if tx_positions.shape != (1, 3):
            raise ValueError(f"Unexpected transmitter-position shape: {tx_positions.shape}")
        bs_positions.append(tx_positions[0])

    assert receiver_positions is not None and scene is not None
    counts = np.stack(path_counts, axis=1)
    minimum_active = int(config.get("minimum_active_bts", 1))
    eligible = np.flatnonzero((counts > 0).sum(axis=1) >= minimum_active)
    requested_users = int(config.get("pilot_users", len(eligible)))
    if requested_users > len(eligible):
        raise ValueError(
            f"Requested {requested_users} users but only {len(eligible)} are eligible"
        )
    rng = np.random.default_rng(int(config.get("sampling_seed", 42)))
    source_indices = np.sort(rng.choice(eligible, requested_users, replace=False))

    params = _channel_parameters(config["radio"])
    antennas = int(np.prod(config["radio"]["antenna_shape"]))
    subcarriers = int(config["radio"]["selected_subcarriers"])
    channels = np.zeros(
        (len(source_indices), len(tx_sets), antennas, subcarriers),
        dtype=np.complex64,
    )
    for station_index, tx_set in enumerate(tx_sets):
        dataset = dm.load(
            scenario_name,
            tx_sets={tx_set: [tx_index]},
            rx_sets={rx_set: source_indices.tolist()},
            matrices=CHANNEL_MATRICES,
            max_paths=max_paths,
        )
        generated = dataset.compute_channels(params)
        if generated.shape != (len(source_indices), 1, antennas, subcarriers):
            raise ValueError(f"Unexpected channel shape: {generated.shape}")
        channels[:, station_index] = generated[:, 0]

    bs_positions_array = np.stack(bs_positions).astype(np.float32)
    positions = receiver_positions[source_indices].astype(np.float32)
    availability = counts[source_indices] > 0
    map_settings = config["map"]
    raster_size = tuple(int(value) for value in map_settings["raster_size"])
    bounds = centered_bounds(
        np.concatenate((receiver_positions[eligible, :2], bs_positions_array[:, :2])),
        raster_size,
        float(map_settings["resolution_m"]),
    )
    map_raster = rasterize_scene(
        scene,
        bounds,
        raster_size=raster_size,
        height_scale_m=float(map_settings["height_scale_m"]),
    )

    features = np.stack((channels.real, channels.imag), axis=2)
    active_features = features[availability]
    feature_mean = active_features.mean(axis=(0, 2, 3)).astype(np.float32)
    feature_std = np.maximum(
        active_features.std(axis=(0, 2, 3)), 1e-12
    ).astype(np.float32)

    metadata = {
        "name": config["name"],
        "scenario": scenario_name,
        "samples": int(len(source_indices)),
        "source_receivers": int(len(receiver_positions)),
        "eligible_receivers": int(len(eligible)),
        "minimum_active_bts": minimum_active,
        "tx_sets": tx_sets,
        "rx_set": rx_set,
        "channel_shape": list(channels.shape),
        "map_shape": list(map_raster.shape),
        "map_bounds": bounds.as_array().tolist(),
        "map_resolution_m": float(map_settings["resolution_m"]),
        "active_links_per_bts": availability.sum(axis=0).tolist(),
        "radio": config["radio"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            temporary_path,
            channels=channels,
            bs_positions=bs_positions_array,
            bs_available=availability,
            positions=positions,
            source_indices=source_indices.astype(np.int64),
            map_raster=map_raster,
            map_bounds=bounds.as_array(),
            feature_mean=feature_mean,
            feature_std=feature_std,
            scenario=np.asarray(scenario_name),
            tx_sets=np.asarray(tx_sets, dtype=np.int16),
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        resolve_project_path(args.config).read_text(encoding="utf-8")
    )
    print(json.dumps(build_scenario(config, args.overwrite), indent=2), flush=True)


if __name__ == "__main__":
    main()
