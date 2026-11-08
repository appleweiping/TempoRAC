"""Aggregate multi-seed runs: variant = run name without the trailing _s<seed>.

Reports mean +- sample SD over seeds for validation (best epoch) and, when scored,
test metrics; optional extra eval files (eval_<tag>.json) are aggregated too.
Usage: python -m temporac_cvpr.aggregate GROUP_DIR [GROUP_DIR ...] [--tags test_warp] [--latex]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect(dirs: list[str], tags: list[str]):
    var = defaultdict(lambda: defaultdict(list))
    for g in dirs:
        for d in sorted(Path(g).iterdir()):
            res = d / "result.json"
            if not res.exists():
                continue
            m = re.match(r"(.+)_s(\d+)$", d.name)
            name = m.group(1) if m else d.name
            r = json.loads(res.read_text())
            for split in ("val", "test"):
                if split in r:
                    for k in ("mae_p", "obo_p", "mae", "obo", "obz_p", "rmse_p"):
                        if k in r[split]:
                            var[name][f"{split}.{k}"].append(r[split][k])
            for t in tags:
                e = d / f"eval_{t}.json"
                if e.exists():
                    em = json.loads(e.read_text())["metrics"]
                    for k in ("mae_p", "obo_p"):
                        var[name][f"{t}.{k}"].append(em[k])
            var[name]["params"].append(r.get("params", 0))
    return var


def fmt(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "-"
    return f"{v.mean():.4f}±{v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tags = []
    if "--tags" in sys.argv:
        tags = sys.argv[sys.argv.index("--tags") + 1].split(",")
        args = [a for a in args if a != sys.argv[sys.argv.index("--tags") + 1]]
    var = collect(args, tags)
    cols = ["val.mae_p", "val.obo_p", "test.mae_p", "test.obo_p"] + [f"{t}.{k}" for t in tags for k in ("mae_p", "obo_p")]
    print(f"{'variant':32s} {'n':>2s} " + " ".join(f"{c:>17s}" for c in cols))
    for name, m in sorted(var.items(), key=lambda kv: np.mean(kv[1].get("val.mae_p", [9]))):
        n = len(m.get("val.mae_p", []))
        print(f"{name:32s} {n:>2d} " + " ".join(f"{fmt(m.get(c, [])):>17s}" for c in cols))


if __name__ == "__main__":
    main()
