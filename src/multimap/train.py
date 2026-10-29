"""Train and evaluate the clean map-conditioned multi-BTS localizer."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt

from src.experiment_config import resolve_project_path
from src.models.multimap import MultiMapLocalizer
from src.multimap.dataset import MultiMapScenarioDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def scenario_paths(data_config: dict[str, Any], data_root: Path, split: str) -> list[Path]:
    return [
        data_root / split / f"{item['name']}.npz"
        for item in data_config["scenarios"]
        if item["split"] == split
    ]


def build_split_dataset(
    paths: list[Path],
    *,
    max_bts: int,
    normalization_path: Path,
    map_size: tuple[int, int],
    random_bts_subsets: bool,
    minimum_bts: int,
) -> ConcatDataset:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing processed maps:\n" + "\n".join(map(str, missing)))
    return ConcatDataset(
        [
            MultiMapScenarioDataset(
                path,
                max_bts=max_bts,
                random_bts_subsets=random_bts_subsets,
                minimum_bts=minimum_bts,
                normalization_path=normalization_path,
                map_size=map_size,
            )
            for path in paths
        ]
    )


def max_bts_in_files(paths: list[Path]) -> int:
    maximum = 0
    for path in paths:
        with np.load(path) as data:
            maximum = max(maximum, int(data["bs_positions"].shape[0]))
    if maximum == 0:
        raise ValueError("No BTS found")
    return maximum


def make_loader(
    dataset: ConcatDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def predict_positions_m(prediction: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    lower = bounds[:, :2]
    span = bounds[:, 2:] - lower
    return lower + (prediction + 1.0) * span / 2.0


def forward_batch(
    model: nn.Module, batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(
        batch["csi"].to(device, non_blocking=True),
        batch["bs_positions"].to(device, non_blocking=True),
        batch["bs_mask"].to(device, non_blocking=True),
        batch["map"].to(device, non_blocking=True),
    )
    target = batch["target"].to(device, non_blocking=True)
    return prediction, target


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    samples = 0
    loss_fn = nn.MSELoss()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prediction, target = forward_batch(model, batch, device)
            loss = loss_fn(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = len(target)
        total_loss += float(loss.detach()) * count
        samples += count
        if log_every > 0 and (batch_index + 1) % log_every == 0:
            print(
                f"  batch={batch_index + 1}/{len(loader)} "
                f"running_mse={total_loss / samples:.6f}",
                flush=True,
            )
    if samples == 0:
        raise ValueError("Training loader produced no samples")
    return total_loss / samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    errors: list[np.ndarray] = []
    by_scenario: dict[str, list[np.ndarray]] = defaultdict(list)
    normalized_mse = 0.0
    values = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        with torch.autocast(device_type=device.type, enabled=use_amp):
            prediction, target = forward_batch(model, batch, device)
        normalized_mse += float((prediction.float() - target).square().sum())
        values += int(target.numel())
        bounds = batch["map_bounds"].to(device)
        predicted_m = predict_positions_m(prediction.float(), bounds)
        true_m = batch["position"].to(device)
        batch_errors = torch.linalg.vector_norm(predicted_m - true_m, dim=1).cpu().numpy()
        errors.append(batch_errors)
        for scenario, error in zip(batch["scenario"], batch_errors, strict=True):
            by_scenario[str(scenario)].append(np.asarray([error], dtype=np.float32))
    if not errors:
        raise ValueError("Evaluation loader produced no samples")
    all_errors = np.concatenate(errors)

    def summarize(values_array: np.ndarray) -> dict[str, float]:
        return {
            "mean_m": float(values_array.mean()),
            "median_m": float(np.median(values_array)),
            "p90_m": float(np.percentile(values_array, 90)),
            "p95_m": float(np.percentile(values_array, 95)),
        }

    metrics: dict[str, Any] = summarize(all_errors)
    metrics["normalized_mse"] = normalized_mse / values
    metrics["samples"] = int(len(all_errors))
    metrics["per_scenario"] = {
        scenario: summarize(np.concatenate(scenario_errors))
        for scenario, scenario_errors in sorted(by_scenario.items())
    }
    return metrics, all_errors


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_cdf(errors: np.ndarray, path: Path) -> None:
    ordered = np.sort(np.asarray(errors, dtype=np.float32))
    probability = np.arange(1, len(ordered) + 1, dtype=np.float32) / len(ordered)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(ordered, probability)
    axis.set_xlabel("Localization error (m)")
    axis.set_ylabel("CDF")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimap/train_clean_v0.json"),
    )
    parser.add_argument("--data-config", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--map-size", type=int)
    parser.add_argument("--resume", type=Path, help="Resume from a last.pt checkpoint")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-maps-per-split", type=int, help="Debug only: limit loaded maps")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not touch the held-out test maps (recommended for sanity/tuning runs)",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    data_config_path = resolve_project_path(args.data_config or config["data_config"])
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    data_root = resolve_project_path(config["data_root"])
    normalization_path = resolve_project_path(
        args.normalization or config["normalization_path"]
    )
    output_root = resolve_project_path(args.output or config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    use_amp = bool(training["amp"] and device.type == "cuda")
    seed_everything(int(config["seed"]))

    epochs = int(args.epochs or training["epochs"])
    batch_size = int(args.batch_size or training["batch_size"])
    workers = int(training["workers"] if args.workers is None else args.workers)
    map_side = int(args.map_size or training["map_size"][0])
    map_size = (map_side, map_side)
    paths = {
        split: scenario_paths(data_config, data_root, split)
        for split in ("train", "validation", "test")
    }
    # Validation-only runs never open a held-out archive, even for BTS metadata.
    loaded_paths = {
        split: values[:args.max_maps_per_split] if args.max_maps_per_split else values
        for split, values in paths.items()
    }
    max_bts = max_bts_in_files(
        loaded_paths["train"] + loaded_paths["validation"]
        if args.skip_test else sum(loaded_paths.values(), [])
    )
    runtime = {
        "epochs": epochs, "batch_size": batch_size, "workers": workers,
        "map_size": list(map_size), "skip_test": args.skip_test,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "max_maps_per_split": args.max_maps_per_split,
        "data_config": str(data_config_path),
        "normalization_path": str(normalization_path),
        "device": str(device), "amp": use_amp,
    }

    print(
        f"device={device} maps={len(paths['train'])}/{len(paths['validation'])}/"
        f"{len(paths['test'])} max_bts={max_bts} map_size={map_size}",
        flush=True,
    )
    train_data = build_split_dataset(
        loaded_paths["train"],
        max_bts=max_bts,
        normalization_path=normalization_path,
        map_size=map_size,
        random_bts_subsets=bool(training["random_bts_subsets"]),
        minimum_bts=int(training["minimum_bts"]),
    )
    validation_data = build_split_dataset(
        loaded_paths["validation"],
        max_bts=max_bts,
        normalization_path=normalization_path,
        map_size=map_size,
        random_bts_subsets=False,
        minimum_bts=int(training["minimum_bts"]),
    )
    train_loader = make_loader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        device=device,
    )
    validation_loader = make_loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        device=device,
    )

    model = MultiMapLocalizer(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history = []
    best_error = math.inf
    epochs_without_improvement = 0
    start_epoch = 1
    elapsed_before = 0.0
    checkpoint_path = output_root / "best.pt"
    last_checkpoint_path = output_root / "last.pt"
    if args.resume is not None:
        resume_path = resolve_project_path(args.resume)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if "config" in resume and resume["config"] != config:
            raise ValueError("Resume configuration mismatch")
        if "runtime" in resume and resume["runtime"] != runtime:
            raise ValueError("Resume runtime overrides mismatch")
        if not checkpoint_path.is_file():
            raise ValueError("Resume requires the existing best.pt")
        best_saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if best_saved["config"] != config:
            raise ValueError("Existing best.pt configuration mismatch")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = list(resume["history"])
        best_error = float(resume["best_error"])
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        start_epoch = int(resume["epoch"]) + 1
        elapsed_before = float(resume.get("training_seconds", 0.0))
        if "rng_state" in resume:
            restore_rng_state(resume["rng_state"])
        else:
            print("Legacy checkpoint has no RNG state; exact replay unavailable", flush=True)
        print(f"resuming from epoch {start_epoch}", flush=True)
    start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        if epochs_without_improvement >= int(training["patience"]):
            break
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            args.max_train_batches,
            int(training.get("log_every_batches", 100)),
        )
        validation_metrics, _ = evaluate(
            model,
            validation_loader,
            device,
            use_amp,
            args.max_eval_batches,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"val_mean={validation_metrics['mean_m']:.3f}m "
            f"val_median={validation_metrics['median_m']:.3f}m "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )
        if validation_metrics["mean_m"] < best_error:
            best_error = validation_metrics["mean_m"]
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "data_config": str(data_config_path),
                    "normalization_path": str(normalization_path),
                    "max_bts": max_bts,
                    "map_size": map_size,
                    "epoch": epoch,
                    "validation": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        atomic_json_save(history, output_root / "history.json")
        atomic_torch_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_error": best_error,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch": epoch,
                "config": config,
                "runtime": runtime,
                "rng_state": capture_rng_state(),
                "training_seconds": elapsed_before + time.perf_counter() - start,
            },
            last_checkpoint_path,
        )
        if epochs_without_improvement >= int(training["patience"]):
            print("early stopping", flush=True)
            break

    # Finalization may be resumed after the last epoch without entering the loop.
    # Repair interrupted JSON publication from the authoritative checkpoint history.
    atomic_json_save(history, output_root / "history.json")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    sample_counts = {
        "train": len(train_data),
        "validation": len(validation_data),
    }
    del train_loader, validation_loader, train_data, validation_data
    gc.collect()

    test_metrics = None
    test_errors = None
    if not args.skip_test:
        test_data = build_split_dataset(
            loaded_paths["test"],
            max_bts=max_bts,
            normalization_path=normalization_path,
            map_size=map_size,
            random_bts_subsets=False,
            minimum_bts=int(training["minimum_bts"]),
        )
        sample_counts["test"] = len(test_data)
        test_loader = make_loader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            workers=workers,
            device=device,
        )
        test_metrics, test_errors = evaluate(
            model,
            test_loader,
            device,
            use_amp,
            args.max_eval_batches,
        )
    result = {
        "name": config["name"],
        "device": str(device),
        "parameters": parameter_count(model),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation": checkpoint["validation"],
        "test": test_metrics,
        "training_seconds": elapsed_before + time.perf_counter() - start,
        "map_counts": {split: len(split_paths) for split, split_paths in paths.items()},
        "loaded_map_counts": {
            split: len(values) for split, values in loaded_paths.items()
            if split != "test" or not args.skip_test
        },
        "sample_counts": sample_counts,
        "max_bts": max_bts,
        "map_size": list(map_size),
        "truncated_run": args.max_train_batches is not None
        or args.max_eval_batches is not None
        or args.max_maps_per_split is not None,
        "test_evaluated": not args.skip_test,
        "runtime": runtime,
        "environment": {
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    atomic_json_save(result, output_root / "metrics.json")
    if test_errors is not None:
        np.save(output_root / "test_errors_m.npy", test_errors)
        save_cdf(test_errors, output_root / "test_cdf.png")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
