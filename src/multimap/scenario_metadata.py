"""Read and validate the common DeepMIMO metadata used by multi-map runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_SCENE_FILES = ("params.json", "objects.json")
SUPPORTED_VERTEX_FILES = ("vertices.mat", "vertices.npz")


def read_params(scenario_dir: Path) -> dict[str, Any]:
    return json.loads((scenario_dir / "params.json").read_text(encoding="utf-8"))


def validate_scenario(
    scenario_dir: Path,
    *,
    expected_frequency_hz: float,
    expected_raytracer_contains: str | None = None,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_SCENE_FILES if not (scenario_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required files: {', '.join(missing)}")
    vertex_file = next(
        (name for name in SUPPORTED_VERTEX_FILES if (scenario_dir / name).is_file()),
        None,
    )
    if vertex_file is None:
        raise ValueError(
            "missing scene vertices; expected one of: "
            + ", ".join(SUPPORTED_VERTEX_FILES)
        )

    params = read_params(scenario_dir)
    version = str(params.get("version", ""))
    if not version.startswith("4."):
        raise ValueError(f"DeepMIMO v4 is required, found version {version!r}")

    scene_count = int(params.get("scene", {}).get("num_scenes", 0))
    if scene_count != 1:
        raise ValueError(f"one static scene is required, found {scene_count}")

    rt_params = params.get("rt_params", {})
    frequency_hz = float(rt_params.get("frequency", 0.0))
    tolerance_hz = max(1.0, expected_frequency_hz * 1e-6)
    if abs(frequency_hz - expected_frequency_hz) > tolerance_hz:
        raise ValueError(
            f"expected {expected_frequency_hz:g} Hz, found {frequency_hz:g} Hz"
        )

    raytracer = str(rt_params.get("raytracer_name", ""))
    if expected_raytracer_contains and expected_raytracer_contains.lower() not in raytracer.lower():
        raise ValueError(
            f"expected ray tracer containing {expected_raytracer_contains!r}, "
            f"found {raytracer!r}"
        )

    sets = list(params.get("txrx_sets", {}).values())
    tx_sets = sorted(int(item["id"]) for item in sets if bool(item.get("is_tx")))
    rx_only_sets = sorted(
        int(item["id"])
        for item in sets
        if bool(item.get("is_rx")) and not bool(item.get("is_tx"))
    )
    if len(tx_sets) < 2:
        raise ValueError(f"at least two transmitter sets are required, found {tx_sets}")
    if len(rx_only_sets) != 1:
        raise ValueError(
            "exactly one RX-only user set is required, "
            f"found {rx_only_sets}"
        )

    return {
        "version": version,
        "frequency_hz": frequency_hz,
        "raytracer": raytracer,
        "tx_sets": tx_sets,
        "rx_set": rx_only_sets[0],
        "num_bts": len(tx_sets),
        "num_receivers": int(
            next(item["num_points"] for item in sets if int(item["id"]) == rx_only_sets[0])
        ),
        "max_reflections": rt_params.get("max_reflections"),
        "max_diffractions": rt_params.get("max_diffractions"),
        "max_scattering": rt_params.get("max_scattering"),
        "vertex_file": vertex_file,
    }
