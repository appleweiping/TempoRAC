"""Per-variant summary (mean +- SD over seeds) of validation metrics and test-time views.

Finished runs: every split/view key in result.json (val, val_shift, val_flip, val_flipshift,
val_tta).  Unfinished runs: best validation epoch so far from log.jsonl (marked '*').
Never reads test keys unless --test is given.
Usage: python -m temporac_cvpr.summ_views GROUP_DIR [GROUP_DIR ...] [--test]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    show_test = "--test" in sys.argv
    done_only = "--done" in sys.argv  # finished runs only (no best-so-far of running seeds)
    groups = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    data = defaultdict(lambda: defaultdict(list))
    partial = defaultdict(int)
    for g in groups:
        for d in sorted(p for p in g.iterdir() if p.is_dir() and not p.name.startswith(("SKIP_", "_"))):
            base = re.sub(r"_s\d+$", "", d.name)  # (no backslash inside f-strings on Python 3.10)
            v = f"{g.name}/{base}"
            res = d / "result.json"
            per_target = sorted(d.glob("result_*.json"))  # multi-target joint runs: variant@target
            if res.exists():
                files = [(f"{v}@{f.stem[len('result_'):]}", f) for f in per_target] or [(v, res)]
                for vv, f in files:
                    r = json.loads(f.read_text())
                    for k, m in r.items():
                        if isinstance(m, dict) and "mae_p" in m and (show_test or not k.startswith("test")):
                            data[vv][k].append((m["mae_p"], m["obo_p"]))
            elif (d / "log.jsonl").exists() and not done_only:
                L = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if '"val"' in x]
                if L and "val_all" in L[0]:
                    for t in L[0]["val_all"]:
                        b = min(L, key=lambda x: (x["val_all"][t]["mae_p"], -x["val_all"][t]["obo_p"]))
                        data[f"{v}@{t}"]["val"].append((b["val_all"][t]["mae_p"], b["val_all"][t]["obo_p"]))
                        partial[f"{v}@{t}"] += 1
                elif L:
                    b = min(L, key=lambda x: (x["val"]["mae_p"], -x["val"]["obo_p"]))
                    data[v]["val"].append((b["val"]["mae_p"], b["val"]["obo_p"]))
                    partial[v] += 1
    keys = ["val", "val_shift", "val_flip", "val_flipshift", "val_tta"]
    if show_test:
        keys += ["test", "test_tta"]
    print("variant".ljust(28), "n".rjust(4), " ".join(k.rjust(19) for k in keys))
    for v in sorted(data):
        cells = []
        for k in keys:
            x = np.array(data[v].get(k, []))
            cells.append("--".rjust(19) if not len(x) else
                         f"{x[:, 0].mean():.4f}±{x[:, 0].std(ddof=1) if len(x) > 1 else 0:.3f}/{x[:, 1].mean():.2f}".rjust(19))
        n = len(data[v]["val"])
        print(v.ljust(28), (f"{n}" + ("*" * (partial[v] > 0))).rjust(4), " ".join(cells))


if __name__ == "__main__":
    main()
