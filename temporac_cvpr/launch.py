"""Launch training runs as detached background processes on the server.

Usage: python -m temporac_cvpr.launch SPEC.json [--max-per-gpu N]
SPEC: {"root": ..., "manifest": [...], "features": ..., "group": "dev1",
       "runs": [{"name": ..., "gpu": 0, "seed": 0, "cfg": {...}, "with_test": false,
                 "features": optional override}, ...]}
Each run writes to ROOT/runs/GROUP/NAME/ (config.json, log.jsonl, result.json, stdout.log).
Completed runs (result.json), live runs (pid alive) and runs with a partial log are
skipped, never overwritten.  With --max-per-gpu the launcher stays alive as a queue:
it keeps at most N of this spec's runs per GPU and fills free slots as runs finish.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def alive(pid_file: Path) -> bool:
    return pid_file.exists() and Path(f"/proc/{pid_file.read_text().strip()}").exists()


def start(spec: dict, r: dict, root: Path) -> int:
    out = root / "runs" / spec["group"] / r["name"]
    out.mkdir(parents=True, exist_ok=True)
    py = str(root / "env" / "bin" / "python")
    cmd = [py, "-m", "temporac_cvpr.train", "--manifest", *spec["manifest"],
           "--features", r.get("features", spec["features"]),
           "--out", str(out), "--seed", str(r.get("seed", 0)), "--cfg", json.dumps(r.get("cfg", {}))]
    for k in ("train_split", "val_split", "test_split"):  # per-run override (joint training targets)
        if k in r or k in spec:
            cmd += [f"--{k.replace('_', '-')}", r.get(k, spec.get(k))]
    if r.get("with_test"):
        cmd.append("--with-test")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r["gpu"]), PYTHONPATH=str(root / "code"),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(out / "stdout.log", "a") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(root / "code"),
                             start_new_session=True)
    (out / "pid").write_text(str(p.pid))
    print(time.strftime("%H:%M:%S"), "launched", r["name"], "gpu", r["gpu"], "pid", p.pid, flush=True)
    return p.pid


def status(spec: dict, r: dict, root: Path) -> str:
    out = root / "runs" / spec["group"] / r["name"]
    skip = root / "runs" / spec["group"] / f"SKIP_{r['name']}"  # tombstone for archived/failed runs
    if skip.exists():
        return "partial"
    if (out / "result.json").exists():
        return "done"
    if alive(out / "pid"):
        return "running"
    if (out / "log.jsonl").exists():
        return "partial"
    return "todo"


def main(spec_path: str, max_per_gpu: int | None) -> None:
    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec["root"])
    if max_per_gpu is None:
        for r in spec["runs"]:
            st = status(spec, r, root)
            if st == "todo":
                start(spec, r, root)
            else:
                print("skip", st, r["name"])
        return
    while True:
        sts = {r["name"]: status(spec, r, root) for r in spec["runs"]}
        busy = {}
        for r in spec["runs"]:
            if sts[r["name"]] == "running":
                busy[r["gpu"]] = busy.get(r["gpu"], 0) + 1
        for r in spec["runs"]:
            if sts[r["name"]] == "todo" and busy.get(r["gpu"], 0) < max_per_gpu:
                start(spec, r, root)
                sts[r["name"]] = "running"
                busy[r["gpu"]] = busy.get(r["gpu"], 0) + 1
                time.sleep(20)  # stagger data loading
        if all(s in ("done", "partial") for s in sts.values()):
            print(time.strftime("%H:%M:%S"), "QUEUE_DONE",
                  {k: v for k, v in sts.items() if v == "partial"}, flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    mpg = None
    if "--max-per-gpu" in sys.argv:
        mpg = int(sys.argv[sys.argv.index("--max-per-gpu") + 1])
    main(sys.argv[1], mpg)
