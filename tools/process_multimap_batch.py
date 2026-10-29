"""Convert a downloaded scenario batch to uniform clean CSI archives."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import resolve_project_path
from src.multimap.build_scenario import build_scenario
from src.multimap.scenario_metadata import validate_scenario


def write_status(path: Path, config_path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def scenario_config(batch: dict[str, Any], item: dict[str, str], index: int) -> dict[str, Any]:
    name = item["name"]
    return {
        "name": f"{batch['name']}_{name}",
        "output_path": f"data/multimap/{item['split']}/{name}.npz",
        "scenario_root": batch["scenario_root"],
        "scenario": name,
        "rx_set": "auto",
        "tx_sets": "all",
        "tx_index": 0,
        "pilot_users": int(batch["users_per_scenario"]),
        "sampling_seed": int(batch["sampling_seed"]) + index,
        "minimum_active_bts": int(batch["minimum_active_bts"]),
        "radio": batch["radio"],
        "map": batch["map"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimap/initial_batch.json"),
    )
    parser.add_argument("--scenario", help="Process only one named scenario")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("outputs/multimap/process_status.json"),
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    batch = json.loads(config_path.read_text(encoding="utf-8"))
    scenario_root = resolve_project_path(batch["scenario_root"])
    selected = [
        (index, item)
        for index, item in enumerate(batch["scenarios"])
        if args.scenario is None or item["name"] == args.scenario
    ]
    if not selected:
        raise SystemExit(f"Scenario not present in batch: {args.scenario}")

    status_path = resolve_project_path(args.status)
    results: list[dict[str, Any]] = []
    for index, item in selected:
        name = item["name"]
        generated_config = scenario_config(batch, item, index)
        output_path = resolve_project_path(generated_config["output_path"])
        result: dict[str, Any] = {
            "scenario": name,
            "split": item["split"],
            "output": str(output_path),
        }
        print(f"\n[{name}]", flush=True)
        try:
            result["source_metadata"] = validate_scenario(
                scenario_root / name,
                expected_frequency_hz=float(batch["expected_frequency_hz"]),
                expected_raytracer_contains=batch.get("expected_raytracer_contains"),
            )
            if output_path.is_file() and not args.overwrite:
                result["status"] = "already_processed"
                print(f"already processed: {output_path}", flush=True)
            else:
                result["generated_metadata"] = build_scenario(
                    generated_config, overwrite=args.overwrite
                )
                result["status"] = "processed"
                print(f"processed: {output_path}", flush=True)
        except Exception as exc:  # retain progress if another scenario fails
            result.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            print(f"FAILED: {result['error']}", flush=True)
        results.append(result)
        write_status(status_path, config_path, results)

    failed = sum(item["status"] == "failed" for item in results)
    print(f"\nCompleted: {len(results) - failed}/{len(results)}", flush=True)
    print(f"Status: {status_path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
