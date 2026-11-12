"""Server-only deterministic engineering fixture; never a scientific result.

Run in the existing server environment with --work under isolated checkpoint-work.
Exercises the actual trainer loop, loss, augmentation/collation and persistent eval
workers, replacing CUDA precision/device and the large decoder with a tiny CPU head.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.25),
                                       torch.nn.Linear(5, 1), torch.nn.Softplus())

    def forward(self, batch):
        x = batch["feats"][1].float().mean(2)
        return {"density": self.net(x).squeeze(-1) * batch["mask"]}


class CpuTorch:
    """Fixture-only device/precision substitution; real trainer RNG/optimizers remain."""
    class cuda:
        is_available = staticmethod(lambda: False)
        max_memory_allocated = staticmethod(lambda: 0)

    def __getattr__(self, name):
        return getattr(torch, name)

    device = staticmethod(lambda _: torch.device("cpu"))
    autocast = staticmethod(lambda *a, **k: contextlib.nullcontext())


def child(args):
    from temporac_cvpr import train

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train.torch = CpuTorch()
    train.build_model = lambda cfg: TinyHead()
    original = train.atomic_save

    def stop_after_checkpoint(obj, path):
        original(obj, path)
        if path.name == "latest.pt" and obj["epoch"] == args.stop:
            os._exit(73)  # a separate interpreter truly dies after committing the epoch

    if args.stop is not None:
        train.atomic_save = stop_after_checkpoint
    cfg = {"rates": [1], "pool_to": 1, "batch": 2, "epochs": 4, "warmup": 1,
           "workers": 2, "ema": 0.9, "eval_every": 2, "crop_prob": 0.8,
           "crop_min": 0.6, "shift_prob": 0.5, "shift_min_period": 1.0,
           "warp_prob": 0.5, "shift_eval": True, "w_count": 0.0, "w_period": 0.0}
    if args.drift == "config":
        cfg["lr"] = 0.01
    manifest = args.work / ("changed_manifest.json" if args.drift == "manifest" else "manifest.json")
    if args.drift == "source":
        identify = train.protocol_identity
        def changed_identity(a, c):
            identity = identify(a, c)
            identity["source"]["fixture_changed.py"] = "different"
            return identity
        train.protocol_identity = changed_identity
    sys.argv = ["train", "--manifest", str(manifest), "--features", str(args.work / "features"),
                "--out", str(args.work / args.run), "--seed", "19", "--cfg", json.dumps(cfg),
                "--val-split", "valid_a,valid_b", "--test-split", "test_a,test_b"]
    if args.resume:
        sys.argv.append("--resume")
    train.main()


def prepare(work):
    assert not work.exists(), "Use a new isolated fixture directory, never historical runs"
    work.mkdir(parents=True)
    features = work / "features"
    for part in ["meta", "r1", "r2"]:
        (features / part).mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(72)
    for i in range(10):
        split = "train" if i < 6 else ("valid_a" if i < 8 else "valid_b")
        key = f"{split}__v{i}"
        frames = 128 + 16 * (i % 3)
        rows.append({"split": split, "video_id": f"v{i}", "count": 4, "bounds": [8, 24, 40, 56, 72, 88, 104, 120],
                     "dataset": "Fixture", "num_frames_csv": frames})
        (features / "meta" / f"{key}.json").write_text(json.dumps({"frames_decoded": frames, "fps": 30}))
        for rate in [1, 2]:
            np.save(features / f"r{rate}" / f"{key}.npy",
                    rng.normal(size=((frames + 2 * rate - 1) // (2 * rate), 1, 3)).astype(np.float16))
    (work / "manifest.json").write_text(json.dumps(rows))
    (work / "changed_manifest.json").write_text(json.dumps(rows) + "\n")


def parent(args):
    from temporac_cvpr import train

    prepare(args.work)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    checks = []

    def run(name, *extra, expected=0, message=None):
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--work", str(args.work),
               "--child", "--run", name, *extra]
        proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (args.work / f"{name}_{len(checks)}.log").write_text(proc.stdout)
        assert proc.returncode == expected, proc.stdout
        if message:
            assert message in proc.stdout, proc.stdout
        checks.append({"run": name, "arguments": list(extra), "exit_code": proc.returncode})

    run("continuous")
    for stop in [0, 1, 3]:  # before eval workers exist, after initialization, after final epoch
        name = f"interrupted_{stop}"
        run(name, "--stop", str(stop), expected=73)
        ck = load(args.work / name / "latest.pt")
        assert ck["epoch"] == stop and ck["scaler"] is None
        assert bool(list((args.work / name).glob("best*.pt"))) == (stop >= 1)
        if stop == 0:
            train.atomic_save({"state": {"uncommitted_fixture": torch.tensor(1)}},
                              args.work / name / "best_valid_a.pt")
        # A crash after writing another log record must not duplicate a durable epoch on resume.
        with (args.work / name / "log.jsonl").open("a") as f:
            f.write('{"epoch":999,"uncommitted_fixture":true}\n')
        for drift in ["config", "manifest", "source"]:
            run(name, "--resume", "--drift", drift, expected=1, message="Resume protocol changed")
        run(name, "--resume")
        actual, expected = load(args.work / name / "latest.pt"), load(args.work / "continuous" / "latest.pt")
        for key in ["model", "optimizer", "scheduler", "scaler", "ema", "bests", "rng", "epoch", "protocol"]:
            equal(actual[key], expected[key])
        for filename in ["best_valid_a.pt", "best_valid_b.pt"]:
            equal(load(args.work / name / filename), load(args.work / "continuous" / filename))
        for filename in ["result.json", "result_valid_a.json", "result_valid_b.json",
                         "predictions_valid_a.json", "predictions_valid_b.json"]:
            equal(json.loads((args.work / name / filename).read_text()),
                  json.loads((args.work / "continuous" / filename).read_text()))
        logs = lambda n: [{k: v for k, v in json.loads(s).items() if k not in {"time", "gpu_mem_gb"}}
                          for s in (args.work / n / "log.jsonl").read_text().splitlines()]
        equal(logs(name), logs("continuous"))
        run(name, "--resume", expected=1, message="Completed run is immutable")
    historical = args.work / "historical_partial"
    historical.mkdir()
    (historical / "log.jsonl").write_text('{"epoch":12}\n')
    run("historical_partial", expected=1, message="refusing overwrite")
    run("historical_partial", "--resume", expected=1, message="old logs cannot recover lost weights")
    # Fail before rename and verify the previous durable checkpoint remains loadable.
    path = args.work / "atomic.pt"
    train.atomic_save({"value": torch.tensor([7])}, path)
    original = train.torch.save
    def fail_save(obj, f):
        f.write(b"partial_fixture")
        raise OSError("injected fixture write failure")
    train.torch.save = fail_save
    try:
        try:
            train.atomic_save({"value": torch.tensor([99])}, path)
        except OSError:
            pass
        else:
            raise AssertionError("injected write must fail")
    finally:
        train.torch.save = original
    equal(load(path), {"value": torch.tensor([7])})
    assert not path.with_name("atomic.pt.tmp").exists()
    # Direct CPU RNG roundtrip also covers global Python/NumPy draws unused by the fixture head.
    streams = {"augmentation": random.Random(82)}
    state = train.capture_rng(streams)
    draws = lambda: [random.random(), np.random.rand(), torch.rand(3), streams["augmentation"].random()]
    first = draws()
    train.restore_rng(state, streams)
    equal(first, draws())
    report = {"scope": "synthetic CPU engineering fixture only; no scientific result", "passed": True,
              "checks": checks, "exact_equal": ["model", "optimizer", "scheduler", "EMA", "best states",
              "CPU/Python/NumPy/augmentation RNG", "losses", "validation selections", "predictions"],
              "atomic_write_failure_preserved_previous": True,
              "cuda_kernel_determinism": "not tested; CUDA RNG guarded implementation inspected"}
    (args.work / "ENGINEERING_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--run", default="fixture")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop", type=int)
    ap.add_argument("--drift", choices=["config", "manifest", "source"])
    args = ap.parse_args()
    child(args) if args.child else parent(args)
