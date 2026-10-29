"""Generate a compact Markdown report from experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.experiment_config import experiment_paths


def _metric(value: dict[str, float]) -> str:
    return f"{value['mean']:.3f} ± {value['std']:.3f}"


def _protocol_lines(config: dict[str, Any], dataset: dict[str, Any]) -> list[str]:
    deepmimo = config["dataset"]["deepmimo"]
    if "tx_set" in deepmimo:
        station = f"TX set `{deepmimo['tx_set']}`, index `{deepmimo.get('tx_index', 0)}`"
        users = f"{dataset['samples']:,} active receivers"
    else:
        station = f"base station `{deepmimo['active_bs']}`"
        rows = deepmimo.get("user_rows")
        users = (
            f"{dataset['samples']:,} users on rows "
            f"`{rows[0]}–{rows[1]}`"
            if rows
            else f"{dataset['samples']:,} users"
        )
    lines = [
        f"- Scenario: `{deepmimo['scenario']}`, {station}.",
        f"- Samples: **{users}**.",
        f"- CSI: `{dataset['feature_shape'][2]} antennas × "
        f"{dataset['feature_shape'][3]} subcarriers`, "
        f"up to {deepmimo['num_paths']} stored paths.",
        f"- Spatial split: {dataset['split_samples']['train']:,} train, "
        f"{dataset['split_samples']['validation']:,} validation, "
        f"{dataset['split_samples']['test']:,} test; no tile overlap.",
        f"- Training: {config['training']['epochs']} epochs, "
        f"seeds `{config['training']['seeds']}`, CUDA AMP.",
    ]
    augmentation = config["training"].get("noise_augmentation", {})
    if augmentation.get("enabled", False):
        lines.append(
            "- Augmentation: complex AWGN with SNR sampled uniformly from "
            f"{augmentation['minimum_snr_db']:g} et "
            f"{augmentation['maximum_snr_db']:g} dB ; "
            f"{100 * augmentation.get('clean_probability', 0.0):g} % "
            "of examples left clean."
        )
    else:
        lines.append("- Training noise augmentation: none.")
    return lines


def generate_report(config: dict[str, Any]) -> Path:
    paths = experiment_paths(config)
    output_root = paths["output"]
    aggregate = json.loads(
        (output_root / "aggregate" / "summary.json").read_text(encoding="utf-8")
    )
    robustness = json.loads(
        (output_root / "robustness" / "robustness.json").read_text(encoding="utf-8")
    )["aggregate"]
    dataset = json.loads(
        paths["processed"].with_suffix(".json").read_text(encoding="utf-8")
    )
    models = list(config["training"]["models"])
    first_name, second_name = models
    first = aggregate["models"][first_name]
    second = aggregate["models"][second_name]

    first_error = first["test"]["mean_m"]["mean"]
    second_error = second["test"]["mean_m"]["mean"]
    clean_winner = first_name if first_error < second_error else second_name
    clean_best = min(first_error, second_error)
    clean_worst = max(first_error, second_error)
    clean_gain = 100.0 * (clean_worst - clean_best) / clean_worst

    noise_rows = []
    for snr in config["robustness"]["snr_db"]:
        label = f"{float(snr):g}dB"
        first_noise = robustness[first_name][label]["mean_m"]
        second_noise = robustness[second_name][label]["mean_m"]
        winner = first_name if first_noise["mean"] < second_noise["mean"] else second_name
        noise_rows.append(
            f"| {float(snr):g} | {_metric(first_noise)} | "
            f"{_metric(second_noise)} | {winner} |"
        )

    lines = [
        f"# Experiment report — {config['name']}",
        "",
        "## Protocol",
        "",
        *_protocol_lines(config, dataset),
        "",
        "## Clean-CSI results",
        "",
        "| Model | Parameters | Train time (s) | Mean (m) | Median (m) | P90 (m) | P95 (m) | Maximum (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {first_name} | {first['parameters'][0]:,} | "
        f"{_metric(first['training_seconds'])} | {_metric(first['test']['mean_m'])} | "
        f"{_metric(first['test']['median_m'])} | {_metric(first['test']['p90_m'])} | "
        f"{_metric(first['test']['p95_m'])} | {_metric(first['test']['max_m'])} |",
        f"| {second_name} | {second['parameters'][0]:,} | "
        f"{_metric(second['training_seconds'])} | {_metric(second['test']['mean_m'])} | "
        f"{_metric(second['test']['median_m'])} | {_metric(second['test']['p90_m'])} | "
        f"{_metric(second['test']['p95_m'])} | {_metric(second['test']['max_m'])} |",
        "",
        f"On clean CSI, `{clean_winner}` has the lower mean error by **{clean_gain:.1f}%**.",
        "",
        "## Noise sensitivity",
        "",
        "Mean localization error in metres, reported as mean ± standard deviation across seeds:",
        "",
        f"| SNR (dB) | {first_name} | {second_name} | Lower error |",
        "|---:|---:|---:|---|",
        *noise_rows,
        "",
        "The lower-error model is selected independently at each SNR. The JSON artifact "
        "contains the complete metrics.",
        "",
        "## Artifacts",
        "",
        "- [Test-error CDF](aggregate/figures/cdf_mean_seeds.png)",
        "- [Validation convergence](aggregate/figures/validation_convergence.png)",
        "- [Noise sensitivity](robustness/noise_robustness.png)",
        "- [Aggregate metrics](aggregate/summary.json)",
        "- [Per-seed metrics](aggregate/seed_metrics.csv)",
        "- [Noise metrics](robustness/robustness.json)",
        "- [Frozen configuration](experiment_config.json)",
        "",
    ]
    report_path = output_root / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
