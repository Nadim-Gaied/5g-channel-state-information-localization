"""Create comparison figures and a compact result table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", default=["cnn", "vit"])
    args = parser.parse_args()

    output_root = args.output_root
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_data = {}
    comparison = {}
    for model_name in args.models:
        model_dir = output_root / model_name
        result = json.loads((model_dir / "results.json").read_text(encoding="utf-8"))
        predictions = np.load(model_dir / "predictions.npz")
        model_data[model_name] = predictions
        comparison[model_name] = {
            "parameters": result["parameters"],
            "training_seconds": result["training_seconds"],
            **result["test"],
        }
    comparison["mean_position_baseline"] = json.loads(
        (output_root / args.models[0] / "results.json").read_text(encoding="utf-8")
    )["mean_position_baseline_test"]

    plt.figure(figsize=(6.5, 4.5))
    for model_name, predictions in model_data.items():
        errors = np.linalg.norm(
            predictions["test_predictions"] - predictions["test_targets"], axis=1
        )
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(errors) + 1) / len(errors)
        plt.plot(sorted_errors, cumulative, label=model_name.upper())
    plt.xlabel("Localization error (m)")
    plt.ylabel("Empirical CDF")
    plt.title("Test localization error")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cdf_comparison.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, (model_name, predictions) in zip(axes, model_data.items()):
        targets = predictions["test_targets"]
        predicted = predictions["test_predictions"]
        axis.scatter(targets[:, 0], targets[:, 1], s=8, alpha=0.35, label="Target")
        axis.scatter(
            predicted[:, 0], predicted[:, 1], s=8, alpha=0.35, label="Prediction"
        )
        axis.set_title(model_name.upper())
        axis.set_xlabel("x (m)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("y (m)")
    axes[1].legend(loc="best")
    fig.suptitle("Target and predicted test positions")
    fig.tight_layout()
    fig.savefig(figure_dir / "prediction_maps.png", dpi=180)
    plt.close(fig)

    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
