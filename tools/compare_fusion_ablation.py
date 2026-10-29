"""Predefined validation selection, then held-out evaluation of add and winner."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools import run_fusion_ablation as runner
from src.models.multimap import MultiMapLocalizer
from src.multimap.train import build_split_dataset, evaluate, make_loader, scenario_paths

SEEDS = (42, 43, 44)


def ensure_shared_controls(configs):
    """Reject any extra experimental variable before accessing held-out data."""
    reference = None
    for config in configs:
        controls = {key: value for key, value in config.items()
                    if key not in ("name", "seed", "output_root", "model")}
        controls["model"] = {key: value for key, value in config["model"].items()
                             if key != "fusion"}
        if reference is None:
            reference = controls
        elif controls != reference:
            raise ValueError("Training/data/architecture controls differ across runs")


def select_candidate(records):
    expected = {(mode, seed) for mode in runner.MODES for seed in SEEDS}
    if len(records) != 9 or {(r["fusion"], r["seed"]) for r in records} != expected:
        raise ValueError("Selection requires all nine distinct completed runs")
    for record in records:
        metrics = record["metrics"]
        if (metrics["test"] is not None or metrics["truncated_run"]
                or metrics["sample_counts"] != {"train": 59392, "validation": 12288}):
            raise ValueError("Invalid full validation-only training record")
        if not np.isfinite(metrics["best_validation"]["mean_m"]):
            raise ValueError("Nonfinite validation metric")
    means = {mode: float(np.mean([
        r["metrics"]["best_validation"]["mean_m"] for r in records if r["fusion"] == mode
    ])) for mode in runner.MODES}
    # Fixed tie-break order; no test result participates in this decision.
    chosen = min(("concat_mlp", "gated"), key=lambda mode: means[mode])
    return {"selected_fusion": chosen, "selection_metric": "mean_validation_mean_m",
            "validation_means_m": means, "seeds": list(SEEDS)}


def collect_records(root):
    records = []
    configs = []
    for mode in runner.MODES:
        for seed in SEEDS:
            _, config, output, identity = runner.configuration(root, mode, seed, False)
            configs.append(config)
            if runner.read_json(output / "identity.json") != identity:
                raise ValueError(f"Changed experimental controls: {mode}/{seed}")
            if not runner.completed(output, config):
                raise ValueError(f"Incomplete controlled run: {mode}/{seed}")
            records.append({"fusion": mode, "seed": seed,
                            "metrics": runner.read_json(output / "metrics.json"),
                            "checkpoint_sha256": runner.digest(output / "best.pt")})
    ensure_shared_controls(configs)
    return records


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def error_metrics(errors):
    errors = np.asarray(errors)
    if errors.shape != (12288,) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Incomplete or invalid final test error vector")
    return {"mean_m": float(errors.mean()), "median_m": float(np.median(errors)),
            "p90_m": float(np.percentile(errors, 90)),
            "p95_m": float(np.percentile(errors, 95))}


def final_evaluation(root, decision, records):
    """Called only after selection.json is durably published."""
    base = root / runner.BASE
    if runner.read_json(base / "selection.json") != decision:
        raise ValueError("Validation selection has not been committed")
    if not torch.cuda.is_available():
        raise RuntimeError("Final controlled evaluation requires CUDA")
    selected = decision["selected_fusion"]
    config = runner.read_json(root / "configs/multimap/fusion_ablation/add_42.json")
    data_config = runner.read_json(root / config["data_config"])
    checkpoint = runner.load_checkpoint(base / "add/seed_42/best.pt")
    device = torch.device("cuda")
    dataset = build_split_dataset(
        scenario_paths(data_config, root / config["data_root"], "test"),
        max_bts=checkpoint["max_bts"], normalization_path=root / config["normalization_path"],
        map_size=tuple(checkpoint["map_size"]), random_bts_subsets=False,
        minimum_bts=config["training"]["minimum_bts"],
    )
    loader = make_loader(dataset, batch_size=config["training"]["batch_size"],
                         shuffle=False, workers=0, device=device)
    rows, vectors = [], {}
    for mode in ("add", selected):
        for seed in SEEDS:
            record = next(r for r in records if r["fusion"] == mode and r["seed"] == seed)
            output = base / "final_test" / mode / f"seed_{seed}"
            identity = {"checkpoint_sha256": record["checkpoint_sha256"],
                        "selection": decision, "evaluation": "full_test_no_adaptation"}
            if (output / "identity.json").exists():
                if runner.read_json(output / "identity.json") != identity:
                    raise ValueError("Final evaluation identity changed")
            else:
                output.mkdir(parents=True, exist_ok=True)
                if any(output.iterdir()):
                    raise ValueError("Unidentified existing final evaluation artifacts")
                runner.write_json(output / "identity.json", identity)
            if (output / "metrics.json").exists() and (output / "errors_m.npy").exists():
                errors = np.load(output / "errors_m.npy")
                metrics = runner.read_json(output / "metrics.json")
                for key, value in error_metrics(errors).items():
                    if not np.isclose(metrics[key], value, rtol=1e-6, atol=1e-6):
                        raise ValueError("Stored final metrics disagree with errors")
            else:
                best = runner.load_checkpoint(base / mode / f"seed_{seed}" / "best.pt")
                model = MultiMapLocalizer(**best["config"]["model"]).to(device)
                model.load_state_dict(best["model"], strict=True)
                metrics, errors = evaluate(model, loader, device, config["training"]["amp"], None)
                error_metrics(errors)
                with (output / "errors_m.npy.tmp").open("wb") as stream:
                    np.save(stream, errors)
                (output / "errors_m.npy.tmp").replace(output / "errors_m.npy")
                runner.write_json(output / "metrics.json", metrics)
                del model, best
                gc.collect()
                torch.cuda.empty_cache()
            rows.append({"fusion": mode, "seed": seed, **error_metrics(errors)})
            vectors[(mode, seed)] = errors
            print(f"Final test {mode}/{seed}: mean={metrics['mean_m']:.4f} m", flush=True)
    return rows, vectors


def create_report(root, decision, records, test_rows, vectors):
    base = root / runner.BASE
    validation_rows = [{"fusion": r["fusion"], "seed": r["seed"],
                        "best_epoch": r["metrics"]["best_epoch"],
                        "parameters": r["metrics"]["parameters"],
                        **{k: r["metrics"]["best_validation"][k]
                           for k in ("mean_m", "median_m", "p90_m", "p95_m")}}
                       for r in records]
    write_csv(base / "validation_summary.csv", validation_rows)
    write_csv(base / "test_summary.csv", test_rows)
    aggregates = {}
    for mode in ("add", decision["selected_fusion"]):
        rows = [r for r in test_rows if r["fusion"] == mode]
        aggregates[mode] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                                   "seed_std": float(np.std([r[key] for r in rows], ddof=1))}
                            for key in ("mean_m", "median_m", "p90_m", "p95_m")}
    runner.write_json(base / "comparison.json", {**decision, "test": aggregates})
    figure, axis = plt.subplots(figsize=(7, 5))
    for mode in aggregates:
        values = np.sort(np.concatenate([vectors[(mode, s)] for s in SEEDS]))
        axis.plot(values, np.arange(1, len(values)+1)/len(values), label=mode)
    axis.set(xlabel="Localization error (m)", ylabel="Empirical CDF", ylim=(0, 1))
    axis.grid(alpha=.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(base / "test_cdf.png", dpi=180)
    plt.close(figure)
    selected = decision["selected_fusion"]
    lines = [
        "# Fusion ablation results", "",
        "Addition, concatenation and gated fusion were trained with seeds 42, 43 "
        "and 44 under the same data split and training controls. Test data was not "
        "loaded during training or candidate selection.", "",
        "## Validation", "",
        "| Fusion | Mean validation error across seeds |", "|---|---:|",
    ]
    for mode, value in decision["validation_means_m"].items():
        lines.append(f"| {mode} | {value:.2f} m |")
    lines += [
        "",
        f"**{selected}** was selected as the better learned alternative before test "
        "evaluation. Addition retained the best validation result overall.", "",
        "## Test", "", "| Fusion | Mean | Median | P90 | P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, values in aggregates.items():
        cells = [
            f"{values[key]['mean']:.2f} ± {values[key]['seed_std']:.2f} m"
            for key in ("mean_m", "median_m", "p90_m", "p95_m")
        ]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    delta = aggregates[selected]["mean_m"]["mean"] - aggregates["add"]["mean_m"]["mean"]
    lines += [
        "",
        f"The selected learned fusion changed mean error by "
        f"{100 * delta / aggregates['add']['mean_m']['mean']:+.2f}% relative to addition. "
        "Validation ranking, parameter count and tail errors must be considered alongside "
        "this mean.", "",
        "Three seeds on one physical-map split support a descriptive comparison, not a "
        "statistical significance claim. The same 12,288 users are evaluated by every seed.",
    ]
    (root / "docs/FUSION_ABLATION_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with runner.gpu_lock(ROOT / runner.BASE):
        records = collect_records(ROOT)  # No test dataset access before all nine pass.
        decision = {**select_candidate(records), "checkpoint_sha256": {
            f"{r['fusion']}/{r['seed']}": r["checkpoint_sha256"] for r in records}}
        selection_path = ROOT / runner.BASE / "selection.json"
        if selection_path.exists() and runner.read_json(selection_path) != decision:
            raise ValueError("Refusing to change an existing test-selection decision")
        runner.write_json(selection_path, decision)
        rows, vectors = final_evaluation(ROOT, decision, records)
        create_report(ROOT, decision, records, rows, vectors)


if __name__ == "__main__":
    main()
