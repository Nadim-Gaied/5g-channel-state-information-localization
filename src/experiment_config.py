"""Load and validate reproducible CSI experiment configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_sections = {"name", "dataset", "preprocessing", "training", "robustness"}
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")

    training = config["training"]
    if not training.get("models") or not training.get("seeds"):
        raise ValueError("training.models and training.seeds cannot be empty")
    if len(set(training["seeds"])) != len(training["seeds"]):
        raise ValueError("training.seeds must contain unique values")

    config["_config_path"] = str(config_path)
    return config


def experiment_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config["dataset"]
    return {
        "raw": resolve_project_path(dataset["raw_path"]),
        "processed": resolve_project_path(dataset["processed_path"]),
        "output": resolve_project_path(config["output_root"]),
    }
