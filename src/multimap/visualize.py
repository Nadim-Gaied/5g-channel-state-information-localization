"""Visual checks for a generated multi-map scenario artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_scenario(path: Path, output: Path) -> None:
    with np.load(path) as data:
        raster = data["map_raster"]
        bounds = data["map_bounds"]
        users = data["positions"]
        stations = data["bs_positions"]
        scenario = str(data["scenario"])
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    axes[0].imshow(raster[0], extent=extent, origin="lower", cmap="gray_r")
    axes[0].scatter(users[:, 0], users[:, 1], s=5, c="tab:blue", alpha=0.45, label="UE")
    axes[0].scatter(
        stations[:, 0], stations[:, 1], s=90, marker="^", c="tab:red", label="BTS"
    )
    axes[0].set_title("Building occupancy and positions")
    axes[0].legend()
    image = axes[1].imshow(
        raster[1], extent=extent, origin="lower", cmap="viridis", vmin=0, vmax=1
    )
    axes[1].set_title("Normalized building height")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    figure.suptitle(scenario)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_scenario(args.input, args.output)
    print(f"FIGURE={args.output}")


if __name__ == "__main__":
    main()
