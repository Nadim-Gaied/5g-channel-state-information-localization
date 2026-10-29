"""Foreground, validation-only runner for the frozen fusion ablation.

Examples (use .venv/Scripts/python.exe):
  tools/run_fusion_ablation.py --fusion add --seed 42
  tools/run_fusion_ablation.py --smoke
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("outputs/multimap/fusion_ablation")
MODES = ("add", "concat_mlp", "gated")
BUDGET_SECONDS = 90 * 60


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def configuration(root, fusion, seed, smoke):
    path = root / "configs/multimap/fusion_ablation" / f"{fusion}_{seed}.json"
    config = read_json(path)
    expected_output = BASE / fusion / f"seed_{seed}"
    if (config["model"]["fusion"] != fusion or config["seed"] != seed
            or Path(config["output_root"]) != expected_output):
        raise ValueError("Frozen configuration mode/seed/output mismatch")
    output = root / (BASE / "smoke" / fusion / f"seed_{seed}" if smoke else expected_output)
    identity = {
        "config": config, "config_sha256": digest(path), "smoke": smoke,
        "data_config_sha256": digest(root / config["data_config"]),
        "normalization_sha256": digest(root / config["normalization_path"]),
    }
    return path, config, output, identity


def check_checkpoint_config(output, config, loader=load_checkpoint):
    checkpoints = {}
    for name in ("best.pt", "last.pt"):
        if (output / name).exists():
            checkpoint = loader(output / name)
            if checkpoint.get("config") != config:
                raise ValueError(f"{name} configuration mismatch; refusing to overwrite")
            checkpoints[name] = checkpoint
    if checkpoints and set(checkpoints) != {"best.pt", "last.pt"}:
        raise ValueError("Incomplete checkpoint pair; preserve and inspect before recovery")
    return checkpoints


def completed(output, config, smoke=False, loader=load_checkpoint):
    checkpoints = check_checkpoint_config(output, config, loader)
    if not checkpoints or not (output / "metrics.json").exists():
        return False
    if not (output / "history.json").exists():
        return False
    try:
        metrics, history = read_json(output / "metrics.json"), read_json(output / "history.json")
    except json.JSONDecodeError:
        return False  # Regenerate final metadata from the durable last.pt.
    best, last = checkpoints["best.pt"], checkpoints["last.pt"]
    if metrics.get("test") is not None or metrics.get("test_evaluated") is not False:
        raise ValueError("Test evaluation detected in validation-only run")
    if metrics.get("device") != "cuda":
        raise ValueError("Ablation requires CUDA")
    if bool(metrics.get("truncated_run")) != smoke:
        raise ValueError("Full/smoke output mismatch")
    expected_runtime = {
        "epochs": 1 if smoke else config["training"]["epochs"],
        "batch_size": 2 if smoke else config["training"]["batch_size"],
        "workers": config["training"]["workers"],
        "map_size": config["training"]["map_size"],
        "skip_test": True, "device": "cuda", "amp": config["training"]["amp"],
        "max_train_batches": 2 if smoke else None,
        "max_eval_batches": 2 if smoke else None,
        "max_maps_per_split": 1 if smoke else None,
    }
    for runtime in (metrics.get("runtime", {}), last.get("runtime", {})):
        if any(runtime.get(key) != value for key, value in expected_runtime.items()):
            raise ValueError("Saved runtime overrides mismatch")
    if history != last.get("history"):
        # last.pt is authoritative after an interrupted multi-file save.
        return False
    epochs = 1 if smoke else config["training"]["epochs"]
    if not history or len(history) > epochs:
        raise ValueError("Invalid history length")
    if [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
        raise ValueError("Noncontiguous training history")
    for record in history:
        if not math.isfinite(record["train_mse"]) or not math.isfinite(record["validation"]["mean_m"]):
            raise ValueError("Nonfinite training/validation metric")
        if record["validation"]["samples"] != (4 if smoke else 12288):
            raise ValueError("Validation batch truncation or incorrect sample count")
    if not smoke:
        if metrics["sample_counts"] != {"train": 59392, "validation": 12288}:
            raise ValueError("Full data sample counts changed")
        if metrics["map_counts"] != {"train": 29, "validation": 6, "test": 6}:
            raise ValueError("Physical-map splits changed")
    best_record = min(history, key=lambda h: h["validation"]["mean_m"])
    if best.get("epoch") != best_record["epoch"]:
        return False
    if (metrics.get("best_epoch") != best["epoch"]
            or metrics.get("best_validation") != best.get("validation")
            or best.get("validation") != best_record["validation"]):
        return False
    if last.get("epoch") != len(history):
        return False
    if not all(k in last for k in ("optimizer", "scheduler", "scaler", "rng_state", "runtime")):
        raise ValueError("Checkpoint cannot reliably resume")
    return len(history) == epochs or (
        len(history) - best["epoch"] >= config["training"]["patience"]
    )


def process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied is fail-closed; an invalid PID is not alive.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError("Cannot inspect previous trainer")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def gpu_lock(base):
    base.mkdir(parents=True, exist_ok=True)
    path = base / "gpu.lock"
    with path.open("a+b") as lock:
        try:
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another fusion runner owns the GPU lock") from error
        try:
            lease = base / "active_process.json"
            if lease.exists() and process_alive(read_json(lease).get("pid")):
                raise RuntimeError("Previous foreground trainer is still running")
            yield lease
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def elapsed_seconds(attempts, now):
    return sum(max(0.0, a.get("ended_at", now) - a["started_at"]) for a in attempts)


def last_training_activity(output, attempt_number, started_at, now):
    """Return the last durable evidence that an interrupted trainer was alive.

    Charging an open attempt until the *recovery* timestamp incorrectly turns
    machine downtime or a paused session into GPU time. The foreground
    trainer writes its log and checkpoints throughout training, so their mtimes
    provide a conservative, durable upper bound without resetting real work.
    """
    candidates = [output / f"train_{attempt_number:03d}.log", output / "last.pt",
                  output / "best.pt", output / "history.json"]
    activity = float(started_at)
    for path in candidates:
        if path.exists():
            activity = max(activity, min(float(now), path.stat().st_mtime))
    return activity


def close_interrupted_attempts(state, output, now):
    """Close or repair orphan attempts using durable activity, not downtime."""
    for number, attempt in enumerate(state["attempts"], 1):
        repair_legacy = (attempt.get("status") == "interrupted_before_recovery"
                         and attempt.get("accounting") != "last_durable_activity")
        if "ended_at" not in attempt or repair_legacy:
            attempt["ended_at"] = last_training_activity(
                output, number, attempt["started_at"], now)
            attempt["status"] = "interrupted_before_recovery"
            attempt["accounting"] = "last_durable_activity"


def launch(command, root, log_path, timeout, on_start):
    """Wait in the foreground; terminate and reap the child on any interruption."""
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            on_start(child.pid)
            return child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()


def run_one(root, fusion, seed, smoke=False, *, launcher=launch,
            loader=load_checkpoint, clock=time.time):
    config_path, config, output, identity = configuration(root, fusion, seed, smoke)
    with gpu_lock(root / BASE) as lease:
        output.mkdir(parents=True, exist_ok=True)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("Config/data/normalization identity mismatch")
        elif any(output.iterdir()):
            raise ValueError("Existing output has no identity; refusing to clobber it")
        else:
            write_json(identity_path, identity)
        if completed(output, config, smoke, loader):
            print(f"Already complete: {output}", flush=True)
            return output
        state_path = output / "runner_state.json"
        state = read_json(state_path) if state_path.exists() else {"attempts": []}
        now = clock()
        close_interrupted_attempts(state, output, now)
        spent = elapsed_seconds(state["attempts"], now)
        if spent >= BUDGET_SECONDS:
            write_json(state_path, state)
            raise TimeoutError("90-minute cumulative run budget exhausted")
        command = [sys.executable, "-u", "-m", "src.multimap.train", "--config",
                   str(config_path), "--skip-test", "--device", "cuda"]
        if smoke:
            command += ["--output", str(output), "--epochs", "1", "--batch-size", "2",
                        "--max-train-batches", "2", "--max-eval-batches", "2",
                        "--max-maps-per-split", "1"]
        if (output / "last.pt").exists():
            command += ["--resume", str(output / "last.pt")]
        attempt = {"started_at": clock(), "command": command, "status": "running"}
        state["attempts"].append(attempt)
        write_json(state_path, state)
        number = len(state["attempts"])
        write_json(output / f"command_{number:03d}.json", {
            "argv": command, "cwd": str(root), "identity": identity,
            "remaining_seconds": BUDGET_SECONDS - spent,
            "source_sha256": {
                str(path): digest(root / path)
                for path in (Path("src/multimap/train.py"), Path("src/models/multimap.py"),
                             Path("src/multimap/dataset.py")) if (root / path).exists()
            },
        })
        def on_start(pid):
            attempt["pid"] = pid
            write_json(state_path, state)
            write_json(lease, {"pid": pid, "output": str(output)})
        print(f"Training {fusion} seed={seed} smoke={smoke}; log={output / f'train_{number:03d}.log'}", flush=True)
        try:
            code = launcher(command, root, output / f"train_{number:03d}.log",
                            BUDGET_SECONDS - spent, on_start)
            attempt["returncode"] = code
            if code:
                raise RuntimeError(f"Trainer failed with exit code {code}; inspect log")
            if not completed(output, config, smoke, loader):
                raise RuntimeError("Trainer exited without a valid completed run")
            attempt["status"] = "completed"
        except BaseException as error:
            attempt["status"] = "failed"
            attempt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            attempt["ended_at"] = clock()
            state["elapsed_seconds"] = elapsed_seconds(state["attempts"], clock())
            write_json(state_path, state)
            write_json(lease, {"pid": None})
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", choices=MODES)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Separate CUDA smoke outputs, two train/validation batches; all variants if fusion omitted")
    args = parser.parse_args()
    if not args.smoke and args.fusion is None:
        parser.error("--fusion is required for a full controlled run")
    variants = MODES if args.fusion is None else (args.fusion,)
    summary_path = ROOT / BASE / "smoke_summary.json"
    if args.smoke:
        write_json(summary_path, {"passed": False, "variants": {}})
    evidence = {}
    for fusion in variants:
        output = run_one(ROOT, fusion, args.seed, args.smoke)
        evidence[fusion] = {"output": str(output), "metrics": read_json(output / "metrics.json")}
    if args.smoke:
        write_json(summary_path, {"passed": set(evidence) == set(MODES), "variants": evidence})


if __name__ == "__main__":
    main()
