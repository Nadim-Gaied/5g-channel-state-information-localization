"""Configuration-driven orchestration for reproducible CSI experiments."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from src.aggregate import aggregate_experiment
from src.data.preprocess import preprocess
from src.experiment_config import experiment_paths, load_experiment_config
from src.robustness import evaluate_robustness
from src.report import generate_report
from src.train import run_training


def save_manifest(config: dict, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    (output_root / "experiment_config.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def run_preprocessing(config: dict) -> dict:
    paths = experiment_paths(config)
    if not paths["raw"].exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {paths['raw']}\n"
            "Run the generation stage first."
        )
    settings = config["preprocessing"]
    return preprocess(
        paths["raw"],
        paths["processed"],
        seed=int(settings["split_seed"]),
        tile_size=tuple(settings["spatial_tile_size_m"]),
        representation=settings["representation"],
        minimum_channel_power=float(settings.get("minimum_channel_power", 0.0)),
    )


def run_models(
    config: dict,
    overwrite: bool = False,
    seed_override: list[int] | None = None,
    epoch_override: int | None = None,
) -> None:
    paths = experiment_paths(config)
    if not paths["processed"].exists():
        raise FileNotFoundError(f"Processed dataset not found: {paths['processed']}")
    settings = config["training"]
    device_name = settings["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The experiment requests CUDA but PyTorch cannot access it")
    seeds = seed_override or [int(seed) for seed in settings["seeds"]]
    epochs = int(epoch_override or settings["epochs"])
    configured_warmup = int(settings.get("warmup_epochs", 0))
    warmup_epochs = min(configured_warmup, max(0, epochs - 1))

    for seed in seeds:
        for model_name in settings["models"]:
            run_root = paths["output"] / "runs" / f"seed_{seed}"
            result_path = run_root / model_name / "results.json"
            if result_path.exists() and not overwrite:
                print(f"SKIP existing run: seed={seed} model={model_name}", flush=True)
                continue
            arguments = argparse.Namespace(
                model=model_name,
                data=paths["processed"],
                output_dir=run_root,
                epochs=epochs,
                batch_size=int(settings["batch_size"]),
                learning_rate=float(settings["learning_rate"]),
                weight_decay=float(settings["weight_decay"]),
                seed=seed,
                threads=int(settings.get("threads", 4)),
                device=device_name,
                patch_size=(
                    settings.get("vit_patch_size")
                    if model_name.startswith("vit")
                    else None
                ),
                scheduler=settings.get("scheduler", "none"),
                warmup_epochs=warmup_epochs,
                minimum_lr_ratio=float(settings.get("minimum_lr_ratio", 0.05)),
                amp=bool(settings.get("amp", False)),
                noise_augmentation=settings.get("noise_augmentation", {}),
            )
            print(f"RUN seed={seed} model={model_name}", flush=True)
            run_training(arguments)
            gc.collect()
            if device_name == "cuda":
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/city_0_newyork_3p5_max_32x64.json"),
    )
    parser.add_argument(
        "--stage",
        choices=(
            "generate",
            "preprocess",
            "train",
            "aggregate",
            "robustness",
            "report",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    paths = experiment_paths(config)
    save_manifest(config, paths["output"])

    if args.stage in ("generate", "all"):
        if config["dataset"].get("generator") == "deepmimo_v4":
            from src.data.generate_deepmimo_v4 import generate_deepmimo_v4

            summary = generate_deepmimo_v4(config, overwrite=args.overwrite)
            print(json.dumps(summary, indent=2), flush=True)
        elif args.stage == "generate":
            raise ValueError(
                "This experiment uses the external MATLAB DeepMIMO v1 generator"
            )
    if args.stage in ("preprocess", "all"):
        summary = run_preprocessing(config)
        print(json.dumps(summary, indent=2), flush=True)
    if args.stage in ("train", "all"):
        run_models(config, args.overwrite, args.seeds, args.epochs)
    if args.stage in ("aggregate", "all"):
        summary = aggregate_experiment(config)
        print(json.dumps(summary, indent=2), flush=True)
    if args.stage in ("robustness", "all"):
        robustness = evaluate_robustness(config)
        print(json.dumps(robustness["aggregate"], indent=2), flush=True)
    if args.stage in ("report", "all"):
        report_path = generate_report(config)
        print(f"REPORT={report_path}", flush=True)


if __name__ == "__main__":
    main()
