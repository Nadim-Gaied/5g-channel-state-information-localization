"""Compare clean-trained and noise-augmented CSI localization experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment_config import experiment_paths, load_experiment_config


def _load_robustness(config: dict[str, Any]) -> dict[str, Any]:
    path = experiment_paths(config)["output"] / "robustness" / "robustness.json"
    return json.loads(path.read_text(encoding="utf-8"))["aggregate"]


def compare_noise_augmentation(
    baseline_config: dict[str, Any],
    augmented_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    baseline = _load_robustness(baseline_config)
    augmented = _load_robustness(augmented_config)
    models = list(baseline_config["training"]["models"])
    if models != list(augmented_config["training"]["models"]):
        raise ValueError("Experiments must use the same models in the same order")

    snr_levels = [float(value) for value in baseline_config["robustness"]["snr_db"]]
    if snr_levels != [
        float(value) for value in augmented_config["robustness"]["snr_db"]
    ]:
        raise ValueError("Experiments must use identical robustness SNR levels")
    labels = ["clean", *[f"{value:g}dB" for value in snr_levels]]

    rows: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {
        "baseline_experiment": baseline_config["name"],
        "augmented_experiment": augmented_config["name"],
        "models": {},
    }
    for model in models:
        comparison["models"][model] = {}
        for label in labels:
            before = baseline[model][label]["mean_m"]
            after = augmented[model][label]["mean_m"]
            improvement = 100.0 * (before["mean"] - after["mean"]) / before["mean"]
            values = {
                "baseline_mean_m": before["mean"],
                "baseline_std_m": before["std"],
                "augmented_mean_m": after["mean"],
                "augmented_std_m": after["std"],
                "relative_improvement_percent": improvement,
            }
            comparison["models"][model][label] = values
            rows.append({"model": model, "condition": label, **values})

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8.0, 5.2))
    for model in models:
        for source, values, linestyle in (
            ("clean training", baseline, "--"),
            ("noise-augmented training", augmented, "-"),
        ):
            means = [values[model][f"{snr:g}dB"]["mean_m"]["mean"] for snr in snr_levels]
            stds = [values[model][f"{snr:g}dB"]["mean_m"]["std"] for snr in snr_levels]
            plt.errorbar(
                snr_levels,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                linestyle=linestyle,
                label=f"{model} — {source}",
            )
    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean localization error (m)")
    plt.title("Effect of AWGN training augmentation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "noise_augmentation_comparison.png", dpi=180)
    plt.close()

    table_rows = []
    for model in models:
        for label in labels:
            values = comparison["models"][model][label]
            table_rows.append(
                f"| {model} | {label} | {values['baseline_mean_m']:.3f} | "
                f"{values['augmented_mean_m']:.3f} | "
                f"{values['relative_improvement_percent']:+.1f} % |"
            )
    report = [
        "# Noise-augmentation comparison",
        "",
        "Mean localization errors are aggregated across three seeds. Positive improvement "
        "means that noise-augmented training reduced error.",
        "",
        "| Model | Test condition | Clean training (m) | Noise-augmented training (m) | Relative improvement |",
        "|---|---|---:|---:|---:|",
        *table_rows,
        "",
        "Training uses complex AWGN with SNR sampled uniformly from 0 to 30 dB; "
        "20% of examples remain clean.",
        "",
        "- [Comparison figure](noise_augmentation_comparison.png)",
        "- [JSON metrics](comparison.json)",
        "- [CSV metrics](metrics.csv)",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--augmented-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_noise_augmentation(
        load_experiment_config(args.baseline_config),
        load_experiment_config(args.augmented_config),
        args.output_dir,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
