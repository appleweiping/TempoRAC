"""Main-table row and ablation table from the single test evaluation of every final run.

Protocol: two-view average (original + exact 2x pyramid shift) for every model that was trained with
the shift; the row without pyramid shift uses its original view.  Mean +- sample SD over seeds.
Writes paper/cvpr27 fragments next to a JSON of all numbers.
Usage: python -m temporac_cvpr.final_tables LLM_GROUP_DIR OUT_DIR
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC/runs")
DS = [("rc", "rc_test", "RepCount-A"), ("ucf", "ucf_test", "UCFRep"), ("cx", "cx_test", "Countix")]


def ms(xs):
    xs = np.asarray(xs, float)
    return (float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, len(xs)) if len(xs) else None


def full_rows(group: Path) -> dict:
    out = {}
    for short, split, _ in DS:
        vals = [json.loads(Path(f).read_text())["metrics"]["density_tta"]
                for f in sorted(glob.glob(str(group / "s*_count" / f"eval_llm_{split}.json")))]
        out[short] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    q = [json.loads(Path(f).read_text())["metrics"]["density_tta"]
         for f in sorted(glob.glob(str(group / "s*_count" / "eval_llm_quva.json")))]
    out["quva"] = (ms([v["mae_p"] for v in q]), ms([v["obo_p"] for v in q]))
    return out


def saved_rows(pattern: str, key: str = "metrics_tta") -> dict:
    out = {}
    for short, split, _ in DS:
        vals = [json.loads(Path(f).read_text())[key] for f in sorted(glob.glob(str(R / pattern / f"eval_test_{short}.json")))]
        out[short] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def joint_rows() -> dict:
    out = {}
    for short, split, _ in DS:
        vals = [json.loads(Path(f).read_text())["metrics_tta"]
                for f in sorted(glob.glob(str(R / "adapt_abl_single" / f"wo_joint_{short}_s*" / f"eval_test_{short}.json")))]
        out[short] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def result_rows(pattern: str) -> dict:  # runs trained with the test split (final_v2): result_<val>.json["test_tta"]
    out = {}
    for short, split, _ in DS:
        val = {"rc": "rc_valid", "ucf": "ucf_dev", "cx": "cx_val"}[short]
        vals = [json.loads(Path(f).read_text())["test_tta"] for f in sorted(glob.glob(str(R / pattern / f"result_{val}.json")))]
        out[short] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def cell(x, best=None, digits=3):
    if x is None:
        return "--"
    m, s, n = x
    t = f"{m:.{digits}f}{{\\scriptsize$\\pm${s:.{digits}f}}}"
    return f"\\best{{{t}}}" if best is not None and abs(m - best) < 1e-12 else t


def main() -> None:
    group, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = [  # (label, numbers)
        (r"\method (full)", full_rows(group)),
        (r"\quad $-$ LLM stage", saved_rows("adapt_dev/wo_llm_adapted_s*")),
        (r"\quad $-$ encoder adaptation\textsuperscript{\dag}", result_rows("final_v2/wo_llm_s*")),
        (r"\quad\quad $+$ frozen-MLLM FiLM conditioning\textsuperscript{\dag}", result_rows("final_v3/full_s*")),
        (r"\quad $-$ multi-rate pyramid ($s{=}4$ only)\textsuperscript{\dag}", saved_rows("adapt_abl/wo_pyramid_s*")),
        (r"\quad $-$ pyramid shift\textsuperscript{\dag}", saved_rows("adapt_abl/wo_shift_s*", "metrics")),
        (r"\quad $-$ second depth (block 12 only)\textsuperscript{\dag}", saved_rows("adapt_abl/wo_depth_s*")),
        (r"\quad $-$ dataset token\textsuperscript{\dag}", saved_rows("adapt_abl/wo_token_s*")),
        (r"\quad $-$ balanced sampling\textsuperscript{\dag}", saved_rows("adapt_abl/wo_balance_s*")),
        (r"\quad $-$ joint training (one model per dataset)\textsuperscript{\dag}", joint_rows()),
    ]
    numbers = {lab: {k: v for k, v in r.items()} for lab, r in rows}
    (out / "final_numbers.json").write_text(json.dumps(numbers, indent=1))
    best = {}
    for short, _, _ in DS:
        for i in (0, 1):
            vals = [r[short][i][0] for _, r in rows if r.get(short) and r[short][i]]
            best[(short, i)] = (min if i == 0 else max)(vals) if vals else None
    lines = []
    for lab, r in rows:
        cells = []
        for short, _, _ in DS:
            for i in (0, 1):
                cells.append(cell(r[short][i], best[(short, i)]))
        n = max((r[s][0][2] for s, _, _ in DS if r[s][0]), default=0)
        lines.append(f"{lab} & {n} & " + " & ".join(cells) + r" \\")
    (out / "ablation_rows.tex").write_text("\n".join(lines) + "\n")
    f = rows[0][1]
    main_row = " & ".join(cell(f[s][i]) for s, _, _ in DS for i in (0, 1))
    (out / "main_row.tex").write_text(main_row + "\n")
    print("\n".join(lines))
    print("MAIN", main_row)
    print("QUVA", f["quva"])


if __name__ == "__main__":
    main()
