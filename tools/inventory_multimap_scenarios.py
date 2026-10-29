"""Inventory compatible DeepMIMO scenarios without downloading them."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import deepmimo as dm


def family_name(name: str) -> str:
    for suffix in ("_lwm", "_s"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def canonical_name(name: str) -> str:
    """Repair two non-ASCII city slugs currently misdecoded by the catalog API."""
    if name.startswith("city_100_bogot"):
        return "city_100_bogota_3p5"
    if name.startswith("city_92_s") and name.endswith("opaulo_3p5"):
        return "city_92_sopaulo_3p5"
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multimap/scenario_inventory.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    query = {
        "bands": ["sub6"],
        "environment": "outdoor",
        "dynamic": False,
        "numTx": {"min": 2, "max": 1000},
    }
    names = dm.search(query) or []
    names = sorted(
        canonical_name(name)
        for name in names
        if name.startswith("city_")
        and ("3p5" in name.lower() or name == "city_01_villas_florida")
    )
    families: dict[str, list[str]] = {}
    for name in names:
        families.setdefault(family_name(name), []).append(name)

    # Prefer one base ray-tracer export per physical map to prevent leakage
    # between near-duplicate scenario variants.
    selected = []
    for family, variants in sorted(families.items()):
        preferred = family if family in variants else sorted(variants)[0]
        selected.append((family, preferred, sorted(variants)))

    rng = random.Random(args.seed)
    shuffled = [family for family, _, _ in selected if family != "city_0_newyork_3p5"]
    rng.shuffle(shuffled)
    n_val = max(1, round(0.15 * len(selected)))
    n_test = max(1, round(0.15 * len(selected)))
    validation = set(shuffled[:n_val])
    test = set(shuffled[n_val : n_val + n_test])

    scenario_root = Path("deepmimo_scenarios")
    entries = []
    for family, preferred, variants in selected:
        split = "validation" if family in validation else "test" if family in test else "train"
        entries.append(
            {
                "family": family,
                "scenario": preferred,
                "variants": variants,
                "split": split,
                "status": (
                    "available_locally"
                    if (scenario_root / preferred).is_dir()
                    else "not_downloaded"
                ),
            }
        )

    inventory = {
        "purpose": "Candidate inventory only; no scenario was downloaded.",
        "query": query,
        "selection": {
            "frequency_name_filter": "3p5",
            "one_variant_per_physical_city": True,
            "forced_training_scenario": "city_0_newyork_3p5",
            "split_seed": args.seed,
        },
        "raw_variant_count": len(names),
        "deduplicated_map_count": len(entries),
        "split_counts": {
            split: sum(entry["split"] == split for entry in entries)
            for split in ("train", "validation", "test")
        },
        "scenarios": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
