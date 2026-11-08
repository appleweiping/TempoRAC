"""Launch a set of training runs as detached background processes on the server.

Usage: python -m temporac_cvpr.launch SPEC.json
SPEC: {"root": ..., "manifest": [...], "features": ..., "group": "dev1",
       "runs": [{"name": ..., "gpu": 0, "seed": 0, "cfg": {...}, "with_test": false}, ...]}
Each run writes to ROOT/runs/GROUP/NAME/ (config.json, log.jsonl, result.json, stdout.log).
Existing completed runs (result.json present) are skipped, never overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec["root"])
    py = str(root / "env" / "bin" / "python")
    for r in spec["runs"]:
        out = root / "runs" / spec["group"] / r["name"]
        if (out / "result.json").exists():
            print("skip (done)", r["name"])
            continue
        pidf = out / "pid"
        if pidf.exists() and Path(f"/proc/{pidf.read_text().strip()}").exists():
            print("skip (running)", r["name"])
            continue
        if (out / "log.jsonl").exists():
            print("skip (partial log present; inspect or archive before relaunch)", r["name"])
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [py, "-m", "temporac_cvpr.train", "--manifest", *spec["manifest"], "--features", spec["features"],
               "--out", str(out), "--seed", str(r.get("seed", 0)), "--cfg", json.dumps(r.get("cfg", {}))]
        for k in ("train_split", "val_split", "test_split"):
            if k in spec:
                cmd += [f"--{k.replace('_', '-')}", spec[k]]
        if r.get("with_test"):
            cmd.append("--with-test")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r["gpu"]), PYTHONPATH=str(root / "code"),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with open(out / "stdout.log", "a") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(root / "code"),
                                 start_new_session=True)
        (out / "pid").write_text(str(p.pid))
        print("launched", r["name"], "gpu", r["gpu"], "pid", p.pid)


if __name__ == "__main__":
    main(sys.argv[1])
