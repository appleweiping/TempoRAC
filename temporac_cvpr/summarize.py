"""Summarise a run group: best-epoch validation (and test if scored) per run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(group_dir: str) -> None:
    rows = []
    for d in sorted(Path(group_dir).iterdir()):
        if not d.is_dir():
            continue
        res = d / "result.json"
        log = d / "log.jsonl"
        if res.exists():
            r = json.loads(res.read_text())
            v = r["val"]
            t = r.get("test", {})
            rows.append((d.name, "done", r["best_epoch"], v["mae"], v["obo"], v["rmse"], t.get("mae"), t.get("obo"),
                         r.get("params")))
        elif log.exists():
            recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            vals = [x["val"] for x in recs if "val" in x]
            if vals:
                b = min(range(len(vals)), key=lambda i: (vals[i]["mae"], -vals[i]["obo"]))
                rows.append((d.name, f"ep{len(recs)}", b, vals[b]["mae"], vals[b]["obo"], vals[b]["rmse"], None, None, None))
            else:
                rows.append((d.name, "running", None, None, None, None, None, None, None))
        else:
            rows.append((d.name, "no-log", None, None, None, None, None, None, None))
    f = lambda x: "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    print(f"{'run':34s} {'state':8s} {'best':>4s} {'valMAE':>7s} {'valOBO':>7s} {'valRMSE':>7s} {'tMAE':>7s} {'tOBO':>7s} params")
    for r in rows:
        print(f"{r[0]:34s} {r[1]:8s} {f(r[2]):>4s} {f(r[3]):>7s} {f(r[4]):>7s} {f(r[5]):>7s} {f(r[6]):>7s} {f(r[7]):>7s} {f(r[8])}")


if __name__ == "__main__":
    main(sys.argv[1])
