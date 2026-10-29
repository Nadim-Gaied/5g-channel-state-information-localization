"""Create the two figures used in the repository README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
COLORS = {"cnn": "#2563EB", "vit": "#E76F51", "general": "#0F766E"}


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save_clean_png(figure: plt.Figure, name: str) -> None:
    path = ASSETS / name
    figure.savefig(path, dpi=180, bbox_inches="tight")
    with Image.open(path) as image:
        image.save(path, dpi=(180, 180))


def noise_figure() -> None:
    clean = load_json(
        "outputs/experiments/city_0_newyork_3p5_max_32x64/robustness/robustness.json"
    )["aggregate"]
    augmented = load_json(
        "outputs/experiments/city_0_newyork_3p5_noise_aug_32x64/robustness/robustness.json"
    )["aggregate"]
    snr = np.array([30, 20, 10, 5, 0])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for axis, data, title in zip(
        axes,
        (clean, augmented),
        ("Trained on clean CSI", "Trained with random SNR augmentation"),
    ):
        for key, label, color in (
            ("cnn_residual", "Residual CNN", COLORS["cnn"]),
            ("vit_deep", "Vision Transformer", COLORS["vit"]),
        ):
            means = [data[key][f"{level}dB"]["mean_m"]["mean"] for level in snr]
            stds = [data[key][f"{level}dB"]["mean_m"]["std"] for level in snr]
            axis.errorbar(
                snr, means, yerr=stds, color=color, marker="o", linewidth=2,
                capsize=3, label=label,
            )
        axis.invert_xaxis()
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlabel("SNR (dB)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean localization error (m)")
    axes[0].legend(frameon=False)
    figure.suptitle("Noise augmentation changes the CNN–ViT comparison", weight="bold")
    figure.tight_layout()
    save_clean_png(figure, "noise_augmentation.png")
    plt.close(figure)


def unseen_map_figure() -> None:
    comparison = load_json(
        "outputs/multimap/baselines/intra_map_csi_only_v0/comparison.json"
    )
    labels = {
        "general_zero_shot": "Map-conditioned\nmodel",
        "cnn_residual": "Scenario-specific\nCNN",
        "vit_deep": "Scenario-specific\nViT",
    }
    colors = {
        "general_zero_shot": COLORS["general"],
        "cnn_residual": COLORS["cnn"],
        "vit_deep": COLORS["vit"],
    }
    rows = comparison["aggregate"]
    x = np.arange(len(rows))
    values = [row["mean_m"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar(
        x, values, width=0.62,
        color=[colors[row["model"]] for row in rows],
    )
    axis.bar_label(bars, labels=[f"{value:.1f} m" for value in values], padding=4)
    axis.set_xticks(x, [labels[row["model"]] for row in rows])
    axis.set_ylabel("Mean localization error (m)")
    axis.set_title("Localization on six held-out maps", weight="bold")
    axis.set_ylim(0, max(values) * 1.2)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.5, -0.22,
        "Same 1,894 test positions; the map-conditioned model receives richer inputs",
        transform=axis.transAxes, ha="center", fontsize=9, color="#4B5563",
    )
    figure.tight_layout()
    save_clean_png(figure, "unseen_map_comparison.png")
    plt.close(figure)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    noise_figure()
    unseen_map_figure()
    print(f"Wrote README figures to {ASSETS}")


if __name__ == "__main__":
    main()
