"""Aggregate multi-seed CSI localization experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_config import experiment_paths


METRICS = ("mean_m", "median_m", "p90_m", "p95_m", "max_m")


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_experiment(config: dict[str, Any]) -> dict[str, Any]:
    output_root = experiment_paths(config)["output"]
    aggregate_dir = output_root / "aggregate"
    figure_dir = aggregate_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    models = list(config["training"]["models"])

    run_results: dict[str, dict[int, dict[str, Any]]] = {
        model: {} for model in models
    }
    run_predictions: dict[str, dict[int, Any]] = {model: {} for model in models}
    rows = []
    for seed in seeds:
        for model in models:
            model_dir = output_root / "runs" / f"seed_{seed}" / model
            result_path = model_dir / "results.json"
            prediction_path = model_dir / "predictions.npz"
            if not result_path.exists() or not prediction_path.exists():
                raise FileNotFoundError(f"Incomplete run: {model_dir}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run_results[model][seed] = result
            run_predictions[model][seed] = np.load(prediction_path)
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "parameters": result["parameters"],
                    "training_seconds": result["training_seconds"],
                    **result["test"],
                }
            )

    summary: dict[str, Any] = {
        "experiment": config["name"],
        "seeds": seeds,
        "models": {},
    }
    for model in models:
        results = [run_results[model][seed] for seed in seeds]
        summary["models"][model] = {
            "parameters": sorted({result["parameters"] for result in results}),
            "training_seconds": _statistics(
                [result["training_seconds"] for result in results]
            ),
            "test": {
                metric: _statistics([result["test"][metric] for result in results])
                for metric in METRICS
            },
            "best_epoch": _statistics(
                [
                    min(
                        result["history"],
                        key=lambda item: item["validation_mean_m"],
                    )["epoch"]
                    for result in results
                ]
            ),
        }

    if len(models) == 2:
        first, second = models
        summary["paired_difference_first_minus_second"] = {
            "first": first,
            "second": second,
            "test": {
                metric: _statistics(
                    [
                        run_results[first][seed]["test"][metric]
                        - run_results[second][seed]["test"][metric]
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            },
        }

    with (aggregate_dir / "seed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    all_errors = []
    errors_by_model: dict[str, list[np.ndarray]] = {model: [] for model in models}
    for model in models:
        for seed in seeds:
            predictions = run_predictions[model][seed]
            errors = np.linalg.norm(
                predictions["test_predictions"] - predictions["test_targets"], axis=1
            )
            errors_by_model[model].append(errors)
            all_errors.append(errors)
    max_error = max(float(errors.max()) for errors in all_errors)
    grid = np.linspace(0.0, max_error, 400)

    plt.figure(figsize=(7.0, 4.8))
    for model in models:
        cdfs = np.stack(
            [
                np.searchsorted(np.sort(errors), grid, side="right") / len(errors)
                for errors in errors_by_model[model]
            ]
        )
        mean_cdf = cdfs.mean(axis=0)
        std_cdf = cdfs.std(axis=0)
        plt.plot(grid, mean_cdf, label=model.upper())
        plt.fill_between(
            grid,
            np.clip(mean_cdf - std_cdf, 0.0, 1.0),
            np.clip(mean_cdf + std_cdf, 0.0, 1.0),
            alpha=0.15,
        )
    plt.xlabel("Localization error (m)")
    plt.ylabel("Empirical CDF")
    plt.title(f"Test error across {len(seeds)} seeds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cdf_mean_seeds.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    for model in models:
        histories = [run_results[model][seed]["history"] for seed in seeds]
        epochs = np.asarray([item["epoch"] for item in histories[0]])
        validation_errors = np.asarray(
            [
                [item["validation_mean_m"] for item in history]
                for history in histories
            ]
        )
        mean_error = validation_errors.mean(axis=0)
        std_error = validation_errors.std(axis=0)
        plt.plot(epochs, mean_error, label=model.upper())
        plt.fill_between(
            epochs,
            np.maximum(0.0, mean_error - std_error),
            mean_error + std_error,
            alpha=0.15,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean validation error (m)")
    plt.title("Validation convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "validation_convergence.png", dpi=180)
    plt.close()

    for predictions_by_seed in run_predictions.values():
        for predictions in predictions_by_seed.values():
            predictions.close()

    (aggregate_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
