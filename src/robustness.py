"""Evaluate multi-seed CSI models under complex additive Gaussian noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiment_config import experiment_paths, load_experiment_config
from src.metrics import localization_metrics
from src.models import build_model


def add_complex_noise(
    normalized_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add AWGN at a target SNR in the original real/imaginary CSI space."""
    raw = normalized_features * feature_std + feature_mean
    signal_power = np.mean(raw[:, 0] ** 2 + raw[:, 1] ** 2, axis=(1, 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    component_std = np.sqrt(noise_power / 2.0)[:, None, None, None]
    noise = rng.standard_normal(raw.shape, dtype=np.float32) * component_std
    noisy = raw + noise
    return ((noisy - feature_mean) / feature_std).astype(np.float32)


def predict(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    use_amp: bool = False,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(
                device, non_blocking=True
            )
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(batch)
            outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def evaluate_robustness(config: dict[str, Any]) -> dict[str, Any]:
    paths = experiment_paths(config)
    output_root = paths["output"]
    robustness_dir = output_root / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    settings = config["robustness"]
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    models = list(config["training"]["models"])
    device = torch.device(config["training"]["device"])
    use_amp = bool(config["training"].get("amp", False) and device.type == "cuda")

    with np.load(paths["processed"]) as data:
        test_idx = data["test_idx"]
        clean_features = data["features"][test_idx]
        normalized_targets = data["targets"][test_idx]
        feature_mean = data["feature_mean"]
        feature_std = data["feature_std"]
        position_mean = data["position_mean"].reshape(2)
        position_std = data["position_std"].reshape(2)
        representation = (
            str(data["representation"])
            if "representation" in data
            else "real_imag"
        )
    if representation != "real_imag":
        raise ValueError("AWGN robustness currently requires real_imag features")

    target_positions = normalized_targets * position_std + position_mean
    snr_levels = [float(value) for value in settings["snr_db"]]
    rng = np.random.default_rng(int(settings["noise_seed"]))
    features_by_snr = {"clean": clean_features}
    for snr_db in snr_levels:
        features_by_snr[f"{snr_db:g}dB"] = add_complex_noise(
            clean_features, feature_mean, feature_std, snr_db, rng
        )

    run_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for seed in seeds:
        seed_key = str(seed)
        run_results[seed_key] = {}
        for model_name in models:
            model_dir = output_root / "runs" / f"seed_{seed}" / model_name
            metadata = json.loads(
                (model_dir / "results.json").read_text(encoding="utf-8")
            )
            model = build_model(
                model_name,
                input_size=tuple(metadata["input_size"]),
                patch_size=(
                    tuple(metadata["patch_size"])
                    if metadata.get("patch_size")
                    else None
                ),
            ).to(device)
            model.load_state_dict(
                torch.load(
                    model_dir / "best.pt", map_location=device, weights_only=True
                )
            )
            model_results = {}
            for label, features in features_by_snr.items():
                normalized_predictions = predict(
                    model,
                    features,
                    device,
                    int(settings["batch_size"]),
                    use_amp,
                )
                prediction_positions = (
                    normalized_predictions * position_std + position_mean
                )
                model_results[label] = localization_metrics(
                    prediction_positions, target_positions
                )
            run_results[seed_key][model_name] = model_results
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    labels = ["clean", *[f"{snr:g}dB" for snr in snr_levels]]
    aggregate = {}
    for model_name in models:
        aggregate[model_name] = {}
        for label in labels:
            aggregate[model_name][label] = {
                metric: _mean_std(
                    [run_results[str(seed)][model_name][label][metric] for seed in seeds]
                )
                for metric in ("mean_m", "median_m", "p90_m", "p95_m", "max_m")
            }

    result = {
        "experiment": config["name"],
        "noise_seed": int(settings["noise_seed"]),
        "runs": run_results,
        "aggregate": aggregate,
    }
    (robustness_dir / "robustness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(7.0, 4.8))
    for model_name in models:
        means = [
            aggregate[model_name][f"{snr:g}dB"]["mean_m"]["mean"]
            for snr in snr_levels
        ]
        stds = [
            aggregate[model_name][f"{snr:g}dB"]["mean_m"]["std"]
            for snr in snr_levels
        ]
        plt.errorbar(
            snr_levels,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=model_name.upper(),
        )
    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean localization error (m)")
    plt.title(f"Noise sensitivity across {len(seeds)} seeds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(robustness_dir / "noise_robustness.png", dpi=180)
    plt.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/city_0_newyork_3p5_max_32x64.json"),
    )
    args = parser.parse_args()
    result = evaluate_robustness(load_experiment_config(args.config))
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
