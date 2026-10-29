"""Create one global normalization artifact from training-map CSI archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import resolve_project_path
from src.multimap.normalization import compute_global_feature_stats, save_normalization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimap/full_city_batch.json"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/multimap")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/multimap/normalization/full_train_stats.npz"),
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = resolve_project_path(args.data_root)
    training = [item for item in config["scenarios"] if item["split"] == "train"]
    paths = [data_root / "train" / f"{item['name']}.npz" for item in training]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Refusing to compute partial global statistics; missing training maps:\n"
            + formatted
        )

    print(f"Computing statistics from {len(paths)} training maps...", flush=True)
    report = compute_global_feature_stats(paths)
    report["config"] = str(config_path)
    report["splits_excluded"] = ["validation", "test"]
    output = resolve_project_path(args.output)
    save_normalization(report, output)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
