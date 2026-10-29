"""Download and validate a small, restartable DeepMIMO multi-map batch."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deepmimo as dm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import resolve_project_path
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multimap/initial_batch.json"),
    )
    parser.add_argument("--scenario", help="Download only one named scenario")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per missing scenario (default: 3)",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("outputs/multimap/download_status.json"),
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scenario_root = resolve_project_path(config["scenario_root"])
    scenario_root.mkdir(parents=True, exist_ok=True)
    dm.config.set("scenarios_folder", str(scenario_root))
    selected = [
        item
        for item in config["scenarios"]
        if args.scenario is None or item["name"] == args.scenario
    ]
    if not selected:
        raise SystemExit(f"Scenario not present in batch: {args.scenario}")

    status_path = resolve_project_path(args.status)
    results: list[dict[str, Any]] = []
    for item in selected:
        name = item["name"]
        download_name = item.get("download_name", name)
        scenario_dir = scenario_root / name
        result: dict[str, Any] = {"scenario": name, "split": item["split"]}
        print(f"\n[{name}]", flush=True)
        try:
            if not (scenario_dir / "params.json").is_file():
                remote_dir = scenario_root / download_name
                downloaded = None
                for attempt in range(1, args.retries + 1):
                    downloaded = dm.download(download_name, output_dir=str(scenario_root))
                    if (remote_dir / "params.json").is_file():
                        break
                    if attempt < args.retries:
                        delay = 5 * attempt
                        print(
                            f"retry {attempt + 1}/{args.retries} in {delay}s",
                            flush=True,
                        )
                        time.sleep(delay)
                if download_name != name and remote_dir.is_dir() and not scenario_dir.exists():
                    remote_dir.rename(scenario_dir)
                if downloaded is None and not (scenario_dir / "params.json").is_file():
                    raise RuntimeError("DeepMIMO download did not produce a scenario folder")
                result["download"] = "downloaded"
                if download_name != name:
                    result["download_name"] = download_name
            else:
                result["download"] = "already_present"
            result["metadata"] = validate_scenario(
                scenario_dir,
                expected_frequency_hz=float(config["expected_frequency_hz"]),
                expected_raytracer_contains=config.get("expected_raytracer_contains"),
            )
            result["status"] = "ready"
            print(
                f"ready: {result['metadata']['num_bts']} BTS, "
                f"{result['metadata']['num_receivers']} receivers",
                flush=True,
            )
        except Exception as exc:  # continue so a later scenario can still succeed
            result.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            print(f"FAILED: {result['error']}", flush=True)
        results.append(result)
        write_status(status_path, config_path, results)

    ready = sum(item["status"] == "ready" for item in results)
    print(f"\nReady: {ready}/{len(results)}", flush=True)
    print(f"Status: {status_path}", flush=True)
    if ready != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
