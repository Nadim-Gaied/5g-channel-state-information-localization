"""Runner lifecycle guards use fakes; resume replay uses a tiny CPU trainer."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools import run_fusion_ablation as runner
from src.multimap import train


@pytest.fixture
def experiment(tmp_path):
    source = Path("configs/multimap/fusion_ablation/add_42.json")
    config = json.loads(source.read_text())
    target = tmp_path / source
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(config))
    for name in (config["data_config"], config["normalization_path"]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return tmp_path, config, tmp_path / config["output_root"]


def fake_loader(path):
    return json.loads(path.read_text())


def artifacts(output, config, *, finished=True, smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    count = 1 if smoke or not finished else 7
    validation = {"mean_m": 10.0, "samples": 4 if smoke else 12288}
    history = [{"epoch": i, "train_mse": 0.2, "validation": validation}
               for i in range(1, count + 1)]
    best = {"config": config, "epoch": 1, "validation": validation}
    last = {"config": config, "epoch": count, "history": history,
            "optimizer": {}, "scheduler": {}, "scaler": {}, "rng_state": {}, "runtime": {}}
    runtime = {
        "epochs": 1 if smoke else config["training"]["epochs"],
        "batch_size": 2 if smoke else config["training"]["batch_size"],
        "workers": config["training"]["workers"], "map_size": config["training"]["map_size"],
        "skip_test": True, "device": "cuda", "amp": config["training"]["amp"],
        "max_train_batches": 2 if smoke else None, "max_eval_batches": 2 if smoke else None,
        "max_maps_per_split": 1 if smoke else None,
    }
    last["runtime"] = runtime
    metrics = {
        "test": None, "test_evaluated": False, "device": "cuda",
        "truncated_run": smoke, "best_epoch": 1, "best_validation": validation,
        "sample_counts": {"train": 59392, "validation": 12288},
        "map_counts": {"train": 29, "validation": 6, "test": 6},
        "runtime": runtime,
    }
    for name, value in (("best.pt", best), ("last.pt", last),
                        ("metrics.json", metrics), ("history.json", history)):
        runner.write_json(output / name, value)


def prepare_identity(root):
    _, _, output, identity = runner.configuration(root, "add", 42, False)
    runner.write_json(output / "identity.json", identity)


def test_fresh_run_and_completed_skip(experiment):
    root, config, output = experiment
    calls = []
    def launch(command, cwd, log, timeout, on_start):
        calls.append(command)
        assert "--skip-test" in command
        assert "--max-train-batches" not in command
        assert "--max-eval-batches" not in command
        assert "--epochs" not in command
        assert 0 < timeout <= 5400
        artifacts(output, config)
        return 0
    runner.run_one(root, "add", 42, launcher=launch, loader=fake_loader)
    runner.run_one(root, "add", 42, launcher=launch, loader=fake_loader)
    assert len(calls) == 1
    assert (output / "command_001.json").is_file()


def test_resume_preserves_best_and_passes_last(experiment):
    root, config, output = experiment
    prepare_identity(root)
    artifacts(output, config, finished=False)
    original_best = (output / "best.pt").read_bytes()
    def launch(command, cwd, log, timeout, on_start):
        assert command[-2:] == ["--resume", str(output / "last.pt")]
        assert (output / "best.pt").read_bytes() == original_best
        artifacts(output, config)
        return 0
    runner.run_one(root, "add", 42, launcher=launch, loader=fake_loader)
    assert (output / "best.pt").read_bytes() == original_best


@pytest.mark.parametrize("guard", ["config", "normalization", "data_config", "checkpoint", "runtime"])
def test_mismatch_refused_without_launch(experiment, guard):
    root, config, output = experiment
    prepare_identity(root)
    artifacts(output, config, finished=False)
    if guard == "config":
        path = root / "configs/multimap/fusion_ablation/add_42.json"
        changed = copy.deepcopy(config)
        changed["training"]["learning_rate"] = 1.0
        runner.write_json(path, changed)
    elif guard == "checkpoint":
        value = fake_loader(output / "last.pt")
        value["config"]["seed"] = 43
        runner.write_json(output / "last.pt", value)
    elif guard == "runtime":
        value = fake_loader(output / "last.pt")
        value["runtime"]["batch_size"] = 1
        runner.write_json(output / "last.pt", value)
    else:
        (root / config["normalization_path" if guard == "normalization" else guard]).write_bytes(b"changed")
    def unexpected(*args):
        pytest.fail("mismatch must not launch training")
    with pytest.raises(ValueError, match="mismatch"):
        runner.run_one(root, "add", 42, launcher=unexpected, loader=fake_loader)


def test_cumulative_budget_survives_failures(experiment):
    root, config, output = experiment
    times = iter([0, 0, 3000, 3000])
    def fail(*args):
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_one(root, "add", 42, launcher=fail, loader=fake_loader, clock=lambda: next(times))
    assert runner.read_json(output / "runner_state.json")["elapsed_seconds"] == 3000
    def retry(command, cwd, log, timeout, on_start):
        assert timeout == 2400
        raise TimeoutError("time limit")
    times = iter([3000, 3000, 5400, 5400])
    with pytest.raises(TimeoutError):
        runner.run_one(root, "add", 42, launcher=retry, loader=fake_loader, clock=lambda: next(times))
    with pytest.raises(TimeoutError, match="exhausted"):
        runner.run_one(root, "add", 42, launcher=fail, loader=fake_loader, clock=lambda: 5500)


def test_recovery_does_not_charge_inactive_time(experiment):
    root, config, output = experiment
    prepare_identity(root)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "train_001.log"
    log.write_text("epoch 6 complete")
    import os
    os.utime(log, (900.0, 900.0))
    runner.write_json(output / "runner_state.json", {"attempts": [{
        "started_at": 100.0, "status": "running", "pid": 999,
    }]})
    seen = {}
    def resume(command, cwd, log_path, timeout, on_start):
        seen["timeout"] = timeout
        artifacts(output, config)
        return 0
    runner.run_one(root, "add", 42, launcher=resume, loader=fake_loader,
                   clock=lambda: 20_000.0)
    state = runner.read_json(output / "runner_state.json")
    assert state["attempts"][0]["ended_at"] == 900.0
    assert state["attempts"][0]["accounting"] == "last_durable_activity"
    assert seen["timeout"] == runner.BUDGET_SECONDS - 800.0


def test_smoke_isolation(experiment):
    root, config, full_output = experiment
    def launch(command, cwd, log, timeout, on_start):
        assert "--skip-test" in command
        assert command[command.index("--max-train-batches") + 1] == "2"
        assert command[command.index("--max-eval-batches") + 1] == "2"
        smoke_output = Path(command[command.index("--output") + 1])
        assert "smoke" in smoke_output.parts
        artifacts(smoke_output, config, smoke=True)
        return 0
    runner.run_one(root, "add", 42, True, launcher=launch, loader=fake_loader)
    assert not full_output.exists()


def test_gpu_lock_rejects_second_owner(tmp_path):
    with runner.gpu_lock(tmp_path):
        with pytest.raises(RuntimeError, match="GPU lock"):
            with runner.gpu_lock(tmp_path):
                pytest.fail("second owner acquired GPU")


def test_live_orphan_trainer_blocks_new_run(tmp_path, monkeypatch):
    runner.write_json(tmp_path / "active_process.json", {"pid": 123})
    monkeypatch.setattr(runner, "process_alive", lambda pid: pid == 123)
    with pytest.raises(RuntimeError, match="still running"):
        with runner.gpu_lock(tmp_path):
            pytest.fail("orphan trainer was ignored")


def test_foreground_timeout_kills_and_reaps_child(tmp_path, monkeypatch):
    class Child:
        pid = 456
        killed = False
        waits = 0
        def wait(self, timeout=None):
            self.waits += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake trainer", timeout)
            return -1
        def poll(self):
            return None
        def kill(self):
            self.killed = True
    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    pids = []
    with pytest.raises(subprocess.TimeoutExpired):
        runner.launch([sys.executable], tmp_path, tmp_path / "timeout.log", 1, pids.append)
    assert pids == [456] and child.killed and child.waits == 2


def test_incomplete_pair_and_test_leakage_refused(experiment):
    _, config, output = experiment
    artifacts(output, config)
    metrics = runner.read_json(output / "metrics.json")
    metrics["test"] = {"mean_m": 3}
    runner.write_json(output / "metrics.json", metrics)
    with pytest.raises(ValueError, match="Test evaluation"):
        runner.completed(output, config, loader=fake_loader)
    (output / "last.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        runner.completed(output, config, loader=fake_loader)


def test_rng_roundtrip():
    train.seed_everything(43)
    state = train.capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(4))
    train.restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], atol=0, rtol=0)


def test_epoch_resume_exact_replay_and_no_test_open(tmp_path, monkeypatch):
    """Interrupt after epoch 1's durable save, then replay epoch 2 exactly."""
    from torch.utils.data import Dataset
    class TinyDataset(Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, index):
            noise = torch.rand(()) + random.random() + np.random.random()
            return {"csi": torch.ones(2, 2, 16, 16) * noise,
                    "bs_positions": torch.ones(2, 3), "bs_mask": torch.ones(2, dtype=torch.bool),
                    "map": torch.zeros(2, 32, 32), "target": torch.tensor([0.1, 0.2]),
                    "map_bounds": torch.tensor([0., 0., 10., 10.]),
                    "position": torch.tensor([5.5, 6.]), "scenario": "tiny"}
    data_config = tmp_path / "data.json"
    runner.write_json(data_config, {"scenarios": [
        {"name": split, "split": split} for split in ("train", "validation", "test")]})
    config = {"name": "tiny", "data_config": str(data_config), "data_root": str(tmp_path),
              "normalization_path": str(tmp_path / "unused.npz"), "seed": 44,
              "model": {"embedding_dim": 8, "num_heads": 2, "set_depth": 1, "fusion": "gated"},
              "training": {"epochs": 2, "batch_size": 2, "workers": 0, "map_size": [32, 32],
                           "amp": False, "random_bts_subsets": True, "minimum_bts": 1,
                           "learning_rate": .001, "weight_decay": .0001,
                           "minimum_learning_rate": .00001, "patience": 6}}
    config_path = tmp_path / "train.json"
    runner.write_json(config_path, config)
    def build(paths, **kwargs):
        assert all("test" not in p.parts for p in paths)
        return TinyDataset()
    def maximum(paths):
        assert all("test" not in p.parts for p in paths)
        return 2
    monkeypatch.setattr(train, "build_split_dataset", build)
    monkeypatch.setattr(train, "max_bts_in_files", maximum)
    def invoke(output, resume=False):
        argv = ["train", "--config", str(config_path), "--output", str(output),
                "--skip-test", "--device", "cpu"]
        if resume:
            argv += ["--resume", str(output / "last.pt")]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
    uninterrupted, resumed = tmp_path / "whole", tmp_path / "resumed"
    invoke(uninterrupted)
    save = train.atomic_torch_save
    def interrupt(value, path):
        save(value, path)
        if path.name == "last.pt":
            raise RuntimeError("simulated power loss")
    monkeypatch.setattr(train, "atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="simulated power loss"):
        invoke(resumed)
    monkeypatch.setattr(train, "atomic_torch_save", save)
    invoke(resumed, resume=True)
    whole, replay = runner.load_checkpoint(uninterrupted / "last.pt"), runner.load_checkpoint(resumed / "last.pt")
    for name in whole["model"]:
        torch.testing.assert_close(whole["model"][name], replay["model"][name], atol=0, rtol=0)
    for a, b in zip(whole["history"], replay["history"]):
        assert a["train_mse"] == b["train_mse"]
        assert a["validation"] == b["validation"]
    metrics = runner.read_json(resumed / "metrics.json")
    assert metrics["test"] is None
    assert metrics["truncated_run"] is False
    assert metrics["training_seconds"] >= sum(h["seconds"] for h in replay["history"])
    best_bytes = (resumed / "best.pt").read_bytes()
    (resumed / "history.json").write_text("incomplete JSON publication")
    invoke(resumed, resume=True)  # A complete epoch budget only regenerates metadata.
    assert (resumed / "best.pt").read_bytes() == best_bytes
    assert len(runner.read_json(resumed / "history.json")) == 2
