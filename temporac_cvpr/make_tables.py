"""Generate LaTeX table rows from multi-seed run directories (no hand transcription).

Each row is mean +- sample SD over seeds of a metric from result.json (split 'test'
or 'val') or from eval_<tag>.json.  Writes a small JSON of the numbers next to the
LaTeX so every printed value can be traced back.
Usage:
  python -m temporac_cvpr.make_tables ablation GROUP_DIR OUT.tex
  python -m temporac_cvpr.make_tables summary GROUP_DIR [--tags quva,x_ucfrep,test_warp]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ABLATION_ROWS = [  # (variant, label)
    ("full", r"\method (full)"),
    ("wo_pyramid_r2", r"\quad single stride $s{=}2$"),
    ("wo_pyramid_r4", r"\quad single stride $s{=}4$"),
    ("wo_pyramid_r8", r"\quad single stride $s{=}8$"),
    ("wo_overlap", r"\quad non-overlapping clips (no averaging)"),
    ("plus_count_loss", r"\quad + normalised count loss"),
    ("depth2", r"\quad 2 encoder blocks"),
]


def collect(group: Path, tags=()):
    out = defaultdict(lambda: defaultdict(list))
    for d in sorted(group.iterdir()):
        r = d / "result.json"
        if not r.exists():
            continue
        name = re.sub(r"_s\d+$", "", d.name)
        res = json.loads(r.read_text())
        for split in ("val", "test"):
            for k, v in res.get(split, {}).items():
                out[name][f"{split}.{k}"].append(v)
        for t in tags:
            e = d / f"eval_{t}.json"
            if e.exists():
                for k, v in json.loads(e.read_text())["metrics"].items():
                    out[name][f"{t}.{k}"].append(v)
    return out


def ms(v, digits=3):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "--", None
    m = v.mean()
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{m:.{digits}f}{{\\scriptsize$\\pm${sd:.{digits}f}}}", m


def ablation(group: Path, out_tex: Path, tags=("test_warp",)):
    data = collect(group, tags)
    cols = [("test.mae_p", "min"), ("test.obo_p", "max"), ("test_warp.mae_p", "min"), ("test_warp.obo_p", "max")]
    best = {}
    for c, mode in cols:
        vals = [np.mean(data[v][c]) for v, _ in ABLATION_ROWS if data[v][c]]
        best[c] = (min if mode == "min" else max)(vals) if vals else None
    lines, numbers = [], {}
    for v, label in ABLATION_ROWS:
        cells = []
        for c, _ in cols:
            s, m = ms(data[v][c])
            if m is not None and best[c] is not None and abs(m - best[c]) < 1e-12:
                s = r"\best{" + s + "}"
            cells.append(s)
            numbers.setdefault(v, {})[c] = data[v][c]
        n = len(data[v]["test.mae_p"])
        lines.append(f"{label} & {n} & " + " & ".join(cells) + r" \\")
    out_tex.write_text("\n".join(lines) + "\n")
    out_tex.with_suffix(".json").write_text(json.dumps(numbers, indent=1))
    print("\n".join(lines))


def summary(group: Path, tags):
    data = collect(group, tags)
    keys = ["test.mae_p", "test.obo_p"] + [f"{t}.{k}" for t in tags for k in ("mae_p", "obo_p")]
    print("variant".ljust(24), " ".join(k.rjust(20) for k in keys))
    for v in sorted(data):
        print(v.ljust(24), " ".join(ms(data[v][k])[0].replace("{\\scriptsize$\\pm$", "+-").replace("}", "").rjust(20)
                                    for k in keys))


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "ablation":
        ablation(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        tags = []
        if "--tags" in sys.argv:
            tags = sys.argv[sys.argv.index("--tags") + 1].split(",")
        summary(Path(sys.argv[2]), tags)
