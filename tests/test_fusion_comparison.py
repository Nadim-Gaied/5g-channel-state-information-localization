import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.compare_fusion_ablation import select_candidate, error_metrics, ensure_shared_controls


def test_all_nine_frozen_configs_share_experimental_controls():
    configs = [json.loads(p.read_text()) for p in
               sorted(Path("configs/multimap/fusion_ablation").glob("*.json"))]
    assert len(configs) == 9
    ensure_shared_controls(configs)
    changed = copy.deepcopy(configs)
    changed[-1]["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="controls differ"):
        ensure_shared_controls(changed)
    changed = copy.deepcopy(configs)
    changed[-1]["model"]["set_depth"] += 1
    with pytest.raises(ValueError, match="controls differ"):
        ensure_shared_controls(changed)


def records():
    return [{"fusion": mode, "seed": seed, "metrics": {
        "test": None, "truncated_run": False,
        "sample_counts": {"train": 59392, "validation": 12288},
        "best_validation": {"mean_m": value + (seed-42)},
    }} for mode, value in (("add", 10), ("concat_mlp", 30), ("gated", 20))
        for seed in (42, 43, 44)]


def test_validation_selection_can_choose_candidate_worse_than_add():
    result = select_candidate(records())
    assert result["selected_fusion"] == "gated"
    assert result["validation_means_m"] == {"add": 11, "concat_mlp": 31, "gated": 21}
    assert result["selection_metric"] == "mean_validation_mean_m"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "test", "truncated", "counts", "nan"])
def test_selection_refuses_incomplete_or_leaking_experiment(mutation):
    values = copy.deepcopy(records())
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[0] = values[1]
    elif mutation == "test":
        values[0]["metrics"]["test"] = {"mean_m": 1}
    elif mutation == "truncated":
        values[0]["metrics"]["truncated_run"] = True
    elif mutation == "counts":
        values[0]["metrics"]["sample_counts"]["validation"] = 4
    else:
        values[0]["metrics"]["best_validation"]["mean_m"] = float("nan")
    with pytest.raises(ValueError):
        select_candidate(values)


def test_final_error_vector_requires_full_finite_nonnegative_test_set():
    values = np.arange(12288, dtype=np.float32)
    assert error_metrics(values)["mean_m"] == float(values.mean())
    for invalid in (values[:-1], np.full(12288, np.nan), -np.ones(12288)):
        with pytest.raises(ValueError):
            error_metrics(invalid)
