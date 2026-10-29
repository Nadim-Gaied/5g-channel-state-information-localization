"""Compare intra-map CSI-only baselines with the zero-shot general model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.experiment_config import resolve_project_path


def summarize(errors: np.ndarray, diagonal_m: float) -> dict[str, float]:
    normalized = np.asarray(errors, dtype=np.float64) / diagonal_m
    return {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "mean_normalized_pct": float(100.0 * np.mean(normalized)),
        "median_normalized_pct": float(100.0 * np.median(normalized)),
        "p90_normalized_pct": float(100.0 * np.percentile(normalized, 90)),
        "p95_normalized_pct": float(100.0 * np.percentile(normalized, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/multimap/full_city_batch.json"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/multimap/csi_only_intra_map"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/multimap/baselines/intra_map_csi_only_v0"),
    )
    parser.add_argument(
        "--general-errors",
        type=Path,
        default=Path("outputs/multimap/training/clean_v0/test_errors_m.npy"),
    )
    parser.add_argument(
        "--general-metrics",
        type=Path,
        default=Path("outputs/multimap/training/clean_v0/metrics.json"),
    )
    parser.add_argument("--models", nargs="+", default=("cnn_residual", "vit_deep"))
    args = parser.parse_args()

    data_config = json.loads(resolve_project_path(args.data_config).read_text(encoding="utf-8"))
    scenarios = [item["name"] for item in data_config["scenarios"] if item["split"] == "test"]
    prepared_root = resolve_project_path(args.prepared_root)
    baseline_root = resolve_project_path(args.baseline_root)
    general_errors = np.load(resolve_project_path(args.general_errors))
    general_metrics = json.loads(resolve_project_path(args.general_metrics).read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[np.ndarray]] = {"general_zero_shot": []}
    aggregate_diagonals: dict[str, list[np.ndarray]] = {"general_zero_shot": []}
    for model in args.models:
        aggregate[model] = []
        aggregate_diagonals[model] = []

    offset = 0
    for scenario in scenarios:
        prepared_path = prepared_root / f"{scenario}.npz"
        with np.load(prepared_path) as data:
            positions = np.asarray(data["positions"][:, :2], dtype=np.float32)
            test_idx = np.asarray(data["test_idx"], dtype=np.int64)
        span = positions.max(axis=0) - positions.min(axis=0)
        diagonal_m = float(np.linalg.norm(span))
        scenario_count = len(positions)
        scenario_general = general_errors[offset : offset + scenario_count][test_idx]
        offset += scenario_count

        general_summary = summarize(scenario_general, diagonal_m)
        rows.append(
            {
                "scenario": scenario,
                "model": "general_zero_shot",
                "training_context": "29 other maps; this map unseen",
                "parameters": int(general_metrics["parameters"]),
                "test_samples": int(len(test_idx)),
                **general_summary,
            }
        )
        aggregate["general_zero_shot"].append(scenario_general)
        aggregate_diagonals["general_zero_shot"].append(
            np.full(len(test_idx), diagonal_m, dtype=np.float32)
        )

        expected_targets = positions[test_idx]
        for model in args.models:
            model_root = baseline_root / scenario / model
            predictions_path = model_root / "predictions.npz"
            results_path = model_root / "results.json"
            if not predictions_path.is_file() or not results_path.is_file():
                raise FileNotFoundError(f"Missing completed baseline: {model_root}")
            with np.load(predictions_path) as predictions:
                predicted = np.asarray(predictions["test_predictions"], dtype=np.float32)
                targets = np.asarray(predictions["test_targets"], dtype=np.float32)
            if not np.allclose(targets, expected_targets):
                raise ValueError(f"Test targets do not match prepared split for {scenario}/{model}")
            errors = np.linalg.norm(predicted - targets, axis=1)
            result = json.loads(results_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "training_context": "same map, spatially disjoint train/test tiles",
                    "parameters": int(result["parameters"]),
                    "test_samples": int(len(test_idx)),
                    **summarize(errors, diagonal_m),
                }
            )
            aggregate[model].append(errors)
            aggregate_diagonals[model].append(
                np.full(len(test_idx), diagonal_m, dtype=np.float32)
            )

    if offset != len(general_errors):
        raise ValueError("General-model error vector does not match test scenario sizes")

    aggregate_rows = []
    for model, chunks in aggregate.items():
        errors = np.concatenate(chunks)
        diagonals = np.concatenate(aggregate_diagonals[model])
        normalized = errors / diagonals
        aggregate_rows.append(
            {
                "model": model,
                "test_samples": int(len(errors)),
                "mean_m": float(errors.mean()),
                "median_m": float(np.median(errors)),
                "p90_m": float(np.percentile(errors, 90)),
                "p95_m": float(np.percentile(errors, 95)),
                "mean_normalized_pct": float(100.0 * normalized.mean()),
                "median_normalized_pct": float(100.0 * np.median(normalized)),
                "p90_normalized_pct": float(100.0 * np.percentile(normalized, 90)),
                "p95_normalized_pct": float(100.0 * np.percentile(normalized, 95)),
            }
        )

    baseline_root.mkdir(parents=True, exist_ok=True)
    report = {
        "important_protocol_difference": {
            "general_zero_shot": "trained on 29 other maps; benchmark map never seen",
            "csi_only": "trained on 70% of the same map; evaluated on disjoint spatial tiles",
        },
        "per_scenario": rows,
        "aggregate": aggregate_rows,
    }
    (baseline_root / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (baseline_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"Comparison: {baseline_root / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
