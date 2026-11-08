"""Summarise a run group from per-epoch validation logs.

Selection metric: official positive-count MAE (``mae_p``); for older logs that lack
it, ``mae_pos`` (positive-count, gt denominator) is used and the column is marked *.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def best_from_log(recs: list[dict]):
    vals = [(i, r["val"]) for i, r in enumerate(recs) if "val" in r]
    if not vals:
        return None
    k = "mae_p" if "mae_p" in vals[0][1] else "mae_pos"
    ko = "obo_p" if "obo_p" in vals[0][1] else "obo"
    i, v = min(vals, key=lambda x: (x[1][k], -x[1][ko]))
    return recs[i]["epoch"], v[k], v[ko], v.get("mae"), v.get("obo"), k != "mae_p"


def main(group_dir: str) -> None:
    print(f"{'run':30s} {'ep':>4s} {'best':>4s} {'vMAEp':>7s} {'vOBOp':>7s} {'vMAEall':>7s} {'vOBOall':>7s} "
          f"{'tMAEp':>7s} {'tOBOp':>7s}")
    for d in sorted(Path(group_dir).iterdir()):
        log = d / "log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        b = best_from_log(recs)
        res = d / "result.json"
        t = json.loads(res.read_text()).get("test", {}) if res.exists() else {}
        f = lambda x: "-" if x is None else f"{x:.4f}"
        state = "done" if res.exists() else f"{len(recs)}"
        if b is None:
            print(f"{d.name:30s} {state:>4s}")
            continue
        star = "*" if b[5] else " "
        print(f"{d.name:30s} {state:>4s} {b[0]:>4d} {f(b[1])}{star}{f(b[2]):>7s} {f(b[3]):>7s} {f(b[4]):>7s} "
              f"{f(t.get('mae_p')):>7s} {f(t.get('obo_p')):>7s}")


if __name__ == "__main__":
    main(sys.argv[1])
