"""Train and evaluate the CNN or ViT CSI localization model."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import CSILocalizationDataset, load_csi_splits
from src.metrics import localization_metrics
from src.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_random_complex_awgn(
    normalized_features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    minimum_snr_db: float,
    maximum_snr_db: float,
    clean_probability: float = 0.0,
) -> torch.Tensor:
    """Add per-sample complex AWGN at a uniformly sampled SNR."""
    if normalized_features.shape[1] != 2:
        raise ValueError("Complex AWGN requires real/imaginary feature channels")
    if minimum_snr_db > maximum_snr_db:
        raise ValueError("minimum_snr_db cannot exceed maximum_snr_db")
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must be between 0 and 1")

    mean = feature_mean.to(
        device=normalized_features.device, dtype=normalized_features.dtype
    )
    std = feature_std.to(
        device=normalized_features.device, dtype=normalized_features.dtype
    )
    raw = normalized_features * std + mean
    signal_power = (raw[:, 0].square() + raw[:, 1].square()).mean(
        dim=(1, 2)
    )
    snr_db = torch.empty(
        len(raw), device=raw.device, dtype=raw.dtype
    ).uniform_(minimum_snr_db, maximum_snr_db)
    noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
    component_std = torch.sqrt(noise_power / 2.0).view(-1, 1, 1, 1)
    if clean_probability:
        clean = torch.rand(len(raw), device=raw.device) < clean_probability
        component_std[clean] = 0.0
    noisy_raw = raw + torch.randn_like(raw) * component_std
    return (noisy_raw - mean) / std


def make_loader(
    dataset, batch_size: int, shuffle: bool, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def evaluate(
    model,
    loader,
    dataset,
    device,
    use_amp: bool = False,
) -> tuple[float, dict, np.ndarray]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                prediction = model(features)
            total_loss += float(loss_fn(prediction, target))
            predictions.append(
                dataset.denormalize_positions(prediction.float()).cpu().numpy()
            )
            targets.append(dataset.denormalize_positions(target).cpu().numpy())

    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    mean_mse = total_loss / (len(dataset) * 2)
    metrics = localization_metrics(prediction_array, target_array)
    return mean_mse, metrics, prediction_array


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    minimum_lr_ratio: float,
):
    """Linear warm-up followed by cosine decay, or ``None`` when disabled."""
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if not 0.0 <= minimum_lr_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio must be between 0 and 1")

    def multiplier(epoch_index: int) -> float:
        completed_epoch = epoch_index + 1
        if warmup_epochs and completed_epoch <= warmup_epochs:
            return completed_epoch / warmup_epochs
        progress = (completed_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def run_training(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    torch.set_num_threads(min(args.threads, torch.get_num_threads()))
    device = torch.device(args.device)

    bundle, datasets = load_csi_splits(args.data)
    train_data = datasets["train"]
    val_data = datasets["validation"]
    test_data = datasets["test"]
    train_loader = make_loader(train_data, args.batch_size, True, device)
    val_loader = make_loader(val_data, args.batch_size, False, device)
    test_loader = make_loader(test_data, args.batch_size, False, device)

    patch_size = (
        tuple(args.patch_size)
        if args.patch_size and args.model.startswith("vit")
        else None
    )
    model = build_model(
        args.model,
        input_size=bundle.input_size,
        patch_size=patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()
    scheduler = (
        build_scheduler(
            optimizer,
            args.epochs,
            args.warmup_epochs,
            args.minimum_lr_ratio,
        )
        if args.scheduler == "warmup_cosine"
        else None
    )
    use_amp = bool(args.amp and device.type == "cuda")
    noise_augmentation = getattr(args, "noise_augmentation", None) or {}
    use_noise_augmentation = bool(noise_augmentation.get("enabled", False))
    if use_noise_augmentation and bundle.representation != "real_imag":
        raise ValueError("Training noise augmentation requires real_imag features")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history = []
    best_val_error = float("inf")
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if use_noise_augmentation:
                features = add_random_complex_awgn(
                    features,
                    bundle.feature_mean,
                    bundle.feature_std,
                    float(noise_augmentation["minimum_snr_db"]),
                    float(noise_augmentation["maximum_snr_db"]),
                    float(noise_augmentation.get("clean_probability", 0.0)),
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                predictions = model(features)
                loss = loss_fn(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item() * len(features)

        train_mse = total_loss / len(train_data)
        val_mse, val_metrics, _ = evaluate(
            model, val_loader, val_data, device, use_amp
        )
        epoch_result = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_mse": train_mse,
            "validation_mse": val_mse,
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)

        if val_metrics["mean_m"] < best_val_error:
            best_val_error = val_metrics["mean_m"]
            torch.save(model.state_dict(), checkpoint_path)
        if scheduler is not None:
            scheduler.step()

    elapsed = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, val_metrics, val_predictions = evaluate(
        model, val_loader, val_data, device, use_amp
    )
    _, test_metrics, test_predictions = evaluate(
        model, test_loader, test_data, device, use_amp
    )

    mean_prediction = np.repeat(
        train_data.positions.numpy().mean(axis=0, keepdims=True),
        len(test_data),
        axis=0,
    )
    baseline_metrics = localization_metrics(mean_prediction, test_data.positions.numpy())

    np.savez_compressed(
        output_dir / "predictions.npz",
        validation_predictions=val_predictions,
        validation_targets=val_data.positions.numpy(),
        test_predictions=test_predictions,
        test_targets=test_data.positions.numpy(),
    )
    result = {
        "model": args.model,
        "device": str(device),
        "seed": args.seed,
        "data": str(args.data),
        "input_size": list(bundle.input_size),
        "patch_size": list(patch_size) if patch_size else None,
        "amp": use_amp,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "minimum_lr_ratio": args.minimum_lr_ratio,
        "noise_augmentation": noise_augmentation,
        "training_seconds": elapsed,
        "best_validation": val_metrics,
        "test": test_metrics,
        "mean_position_baseline_test": baseline_metrics,
        "history": history,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("cnn", "vit", "cnn_residual", "vit_deep"),
        required=True,
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/city_0_newyork_3p5_max_32x64.npz")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        metavar=("ANTENNAS", "SUBCARRIERS"),
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "warmup_cosine"),
        default="none",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.05)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.set_defaults(noise_augmentation={})
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
