"""Official RepNet (updated 2025 ckpt-70) on the Countix test split, with the existing official-core runner.

Reuses external-baselines-20261111/repnet-runner/evaluate_repnet.py unchanged (official model code, weight restore
with exact-equality checks, notebook inference parameters, per-video atomic records, frozen scorer). Only three
things differ from the RepCount/UCFRep/QUVA run:
  * population: Countix cx_test (2403 videos) of the union manifest, crops from the frozen decode metadata;
  * the GPU gate: the old gate required an exclusive GPU1 after a specific predecessor queue; this shared-GPU run
    checks only that the chosen GPU has enough free memory, and TensorFlow is capped (logical memory limit);
  * inference batch_size 8 instead of 20 (number of 64-frame windows per forward pass; windows are independent,
    so this changes memory only, not predictions).
Usage (server): envs/repnet/bin/python repnet_countix.py --gpu 0 --mem 3072 --out DIR [--mode preflight|evaluate] [--resume]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/TempoRAC")
RUNNER = ROOT / "external-baselines-20261111/repnet-runner"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mem", type=int, default=3072, help="TensorFlow GPU memory limit (MiB)")
    ap.add_argument("--min-free", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="evaluate", choices=["preflight", "evaluate"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("evaluate_repnet", RUNNER / "evaluate_repnet.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    def gpu_check(cfg):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"],
                             check=True, capture_output=True, text=True).stdout
        free = {int(r.split(",")[0]): int(r.split(",")[1]) for r in out.strip().splitlines()}
        g = cfg["runtime"]["gpu"]
        if free[g] < cfg["runtime"]["min_gpu_free_mb"]:
            raise RuntimeError(f"Insufficient free GPU memory on GPU{g}: {free[g]} MiB")
        return dict(gpu_snapshot=out, shared_gpu=True)

    ev.gpu_check = gpu_check
    cfg = json.loads((RUNNER / "evaluation_config.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Countix manifest rows carry sha256 but no byte size; the runner's size pre-check needs it. Sizes are read from
    # disk; the runner still verifies every file's SHA-256 against the original manifest value.
    derived = out / "countix_test_manifest.json"
    if not derived.exists():
        import os
        rows = [dict(r, bytes=os.path.getsize(r["file"])) for r in json.loads((ROOT / "data/UNION/manifest_dev.json").read_text())
                if r.get("dataset") == "Countix" and r["split"] == "cx_test"]
        derived.write_text(json.dumps(rows))
    cfg["populations"] = [{"name": "Countix", "manifest": str(derived), "dataset": "Countix",
                           "split": "cx_test", "n": 2403,
                           "decode_metadata_dir": str(ROOT / "features/UNION_stmae_b8_p4/meta")}]
    cfg["inference"]["batch_size"] = 8
    cfg["runtime"].update(gpu=a.gpu, gpu_memory_limit_mb=a.mem, min_gpu_free_mb=a.min_free)
    cfg.pop("predecessor", None)
    cpath = out / "countix_config.json"
    if not (a.resume and cpath.exists()):
        cpath.write_text(json.dumps(cfg, indent=2))
    sys.argv = ["evaluate_repnet.py", "--config", str(cpath), "--output", str(out / "run"), "--mode", a.mode] + \
               (["--resume"] if a.resume else [])
    ev.main()


if __name__ == "__main__":
    main()
