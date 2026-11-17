"""Finish the corrected (known-source) ablation arms on shared GPUs, without competing with other projects.

Exact configurations and evaluations come from the frozen contract via ``ablation_worker.checked_plans`` (same
training snapshot, manifest, features, splits, checkpoints and environment as the arms already completed). The
only differences are scheduling and one addition:
  * an arm starts on a GPU only if that GPU has at least MIN_FREE_MIB free memory, and at most one of our arms
    runs per GPU, so other projects keep their memory headroom;
  * the RepCount-only arms (wo_joint_rc_s*) are additionally scored on UCFRep, Countix and QUVA with their single
    RepCount-selected checkpoint, for the RepCount-trained cross-dataset comparison with external baselines.
A partially written arm directory is archived (never deleted) and the arm restarts from scratch.
Usage (server): python ks_queue.py [--min-free 6000] [--dry]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/TempoRAC")
CDIR = ROOT / "resume20261112/ablation-continuation"
SRC = ROOT / "source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb"
OUT = ROOT / "runs/frozen_known_source_v1_ablation20261112"
LOG = ROOT / "resume20261117/ks_queue.events.jsonl"
PY = str(ROOT / "env/bin/python")
sys.path.insert(0, str(CDIR))
from ablation_worker import checked_plans  # noqa: E402

TRANSFER = [  # extra evaluations of RepCount-only models (cross-dataset protocol of the external baselines)
    {"tag": "xfer_ucf_test", "manifest": None, "split": "ucf_test", "features": None, "aux": None},
    {"tag": "xfer_cx_test", "manifest": None, "split": "cx_test", "features": None, "aux": None},
    {"tag": "xfer_quva", "manifest": str(ROOT / "data/QUVA/manifest.json"), "split": "test",
     "features": str(ROOT / "features/QUVA_stmae_b8_p4"), "aux": [str(ROOT / "features/QUVA_stmae_p4")]},
]


def event(**kw):
    kw["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def free_mib(gpu: int) -> int:
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    return int(out.strip())


def arm_done(name: str, plan: dict) -> bool:
    d = OUT / name
    need = [d / f"eval_{j['tag']}.json" for j in plan["evaluations"]]
    if name.startswith("wo_joint_rc_"):
        need += [d / f"eval_{t['tag']}.json" for t in TRANSFER]
    return all(p.exists() for p in need)


def arm_script(name: str, plan: dict, gpu: int, manifest: str) -> list[list[str]]:
    """Commands for one arm: training (if not finished) then every missing evaluation."""
    r, d = plan["run"], OUT / name
    cmds = []
    if not (d / "result.json").exists():
        cmds.append([PY, "-u", "-B", "-m", "temporac_cvpr.train", "--manifest", manifest, "--features", r["features"],
                     "--out", str(d), "--seed", str(r["seed"]), "--cfg", json.dumps(r["cfg"]),
                     "--train-split", r["train_split"], "--val-split", r["val_split"], "--test-split", r["test_split"]])
    for j in plan["evaluations"]:
        if not (d / f"eval_{j['tag']}.json").exists():
            cmds.append([PY, "-u", "-B", "-m", "temporac_cvpr.eval_saved", "--run", str(d), "--manifest", manifest,
                         "--features", j["features"], "--split", j["split"], "--tag", j["tag"], "--ckpt", j["checkpoint"]])
    if name.startswith("wo_joint_rc_"):
        for t in TRANSFER:
            if not (d / f"eval_{t['tag']}.json").exists():
                c = [PY, "-u", "-B", "-m", "temporac_cvpr.eval_saved", "--run", str(d), "--manifest", t["manifest"] or manifest,
                     "--features", t["features"] or r["features"], "--split", t["split"], "--tag", t["tag"], "--ckpt", "best.pt"]
                if t["aux"]:
                    c += ["--aux", *t["aux"]]
                cmds.append(c)
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free", type=int, default=6000)
    ap.add_argument("--per-gpu", type=int, default=1, help="max concurrent arms of ours per GPU")
    ap.add_argument("--only", default="", help="comma-separated arm names (disjoint subsets per server)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    contract = json.loads((CDIR / "RUN_CONTRACT.json").read_text())
    spec = json.loads(Path(contract["frozen_spec"]["path"]).read_text())
    plans = checked_plans(contract, spec)
    manifest = contract["manifest"]["path"]
    order = contract["lanes"]["gpu0"] + contract["lanes"]["gpu1"]
    if a.only:
        keep = set(a.only.split(","))
        assert keep <= set(order), keep - set(order)
        order = [n for n in order if n in keep]
    todo = [n for n in order if not arm_done(n, plans[n])]
    event(status="start", todo=todo, min_free=a.min_free, per_gpu=a.per_gpu)
    if a.dry:
        for n in todo:
            for c in arm_script(n, plans[n], 0, manifest):
                print(n, " ".join(c)[:300])
        return
    slots = [(g, k) for g in (0, 1) for k in range(a.per_gpu)]
    running: dict[tuple, tuple[str, subprocess.Popen, list]] = {}
    while todo or running:
        for slot in slots:
            gpu = slot[0]
            if slot in running:
                name, p, rest = running[slot]
                if p.poll() is None:
                    continue
                if p.returncode != 0:
                    event(status="failed", run=name, gpu=gpu, returncode=p.returncode, remaining_cmds=len(rest))
                    del running[slot]
                    continue  # the arm stays incomplete; it is not retried automatically
                if rest:
                    cmd = rest.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1",
                               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="2")
                    tag = cmd[cmd.index("--tag") + 1] if "--tag" in cmd else "train"
                    log = open(OUT / name / ("stdout.log" if tag == "train" else f"eval_{tag}.log"), "a")
                    running[slot] = (name, subprocess.Popen(cmd, cwd=str(SRC), env=env, stdout=log, stderr=subprocess.STDOUT,
                                                            stdin=subprocess.DEVNULL), rest)
                    event(status="step", run=name, gpu=gpu, step=tag)
                else:
                    event(status="arm_complete", run=name, gpu=gpu)
                    del running[slot]
            elif todo and free_mib(gpu) >= a.min_free:
                name = todo.pop(0)
                d = OUT / name
                if d.exists() and not (d / "result.json").exists():  # partial training: archive, restart fresh
                    arch = ROOT / "runs/_partial_ks_ablation" / f"{name}_{int(time.time())}"
                    arch.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(d), str(arch))
                    event(status="archived_partial", run=name, to=str(arch))
                d.mkdir(parents=True, exist_ok=True)
                cmds = arm_script(name, plans[name], gpu, manifest)
                # first command is launched on the next pass through the generic "rest" path
                running[slot] = (name, subprocess.Popen(["true"]), cmds)
                event(status="arm_start", run=name, gpu=gpu, free_mib=free_mib(gpu), n_cmds=len(cmds))
                time.sleep(90)  # let the new arm allocate before the next free-memory check
        time.sleep(60)
    event(status="queue_done")


if __name__ == "__main__":
    main()
