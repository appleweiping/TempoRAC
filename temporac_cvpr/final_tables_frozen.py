"""Tables for the frozen framework (two-depth multi-rate pyramid + pyramid shift + joint training; frozen ST-MAE;
no language model).  Each model was scored once on the test splits at the end of its training run
(``--with-test``); the checkpoint of each benchmark is the one selected on that benchmark's validation split.

Main model: final_v2/wo_llm_s0..2 + frozen_a/full_s3..4 (identical configuration).  Ablations: frozen_a/wo_*.
Writes paper/cvpr27/tab fragments: main_row (RepCount-A), app_row (UCFRep, Countix, QUVA zero-shot),
ablation_rows (RepCount-A), app_ablation_rows (all three benchmarks), plus final_numbers_frozen.json.
Usage: python -m temporac_cvpr.final_tables_frozen OUT_DIR
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC/runs")
DS = [("rc", "rc_valid"), ("ucf", "ucf_dev"), ("cx", "cx_val")]
MAIN = ["final_v2/wo_llm_s0", "final_v2/wo_llm_s1", "final_v2/wo_llm_s2", "frozen_a/full_s3", "frozen_a/full_s4"]


def ms(xs):
    xs = np.asarray(xs, float)
    return (float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, len(xs)) if len(xs) else None


def multi(runs: list[str], key: str = "test_tta") -> dict:
    out = {}
    for s, val in DS:
        vals = [json.loads((R / r / f"result_{val}.json").read_text())[key] for r in runs
                if (R / r / f"result_{val}.json").exists()]
        out[s] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def joint_rows() -> dict:
    out = {}
    for s, _ in DS:
        vals = [json.loads(Path(f).read_text())["test_tta"]
                for f in sorted(glob.glob(str(R / "frozen_a" / f"wo_joint_{s}_s*" / "result.json")))]
        out[s] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def quva(runs: list[str]) -> tuple:
    vals = [json.loads((R / r / "eval_quva.json").read_text())["metrics_tta"] for r in runs if (R / r / "eval_quva.json").exists()]
    return ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals])


def cell(x, best=None, d=3):
    if x is None:
        return "--"
    m, s, n = x
    t = f"{m:.{d}f}{{\\scriptsize$\\pm${s:.{d}f}}}"
    return f"\\best{{{t}}}" if best is not None and abs(m - best) < 1e-12 else t


def glob_runs(pattern: str) -> list[str]:
    return sorted(str(Path(p).relative_to(R)) for p in glob.glob(str(R / pattern)) if Path(p, "result.json").exists())


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        (r"\method", multi(MAIN)),
        (r"\quad single view at test (no $2\times$ view)", multi(MAIN, "test")),
        (r"\quad $-$ multi-rate pyramid ($s{=}4$ only)", multi(glob_runs("frozen_a/wo_pyramid_s*"))),
        (r"\quad $-$ pyramid shift (training and test)", multi(glob_runs("frozen_a/wo_shift_s*"), "test")),
        (r"\quad $-$ second depth (block 12 only)", multi(glob_runs("frozen_a/wo_depth_s*"))),
        (r"\quad $-$ dataset token", multi(glob_runs("frozen_a/wo_token_s*"))),
        (r"\quad $-$ balanced sampling", multi(glob_runs("frozen_a/wo_balance_s*"))),
        (r"\quad $-$ joint training (one model per dataset)", joint_rows()),
    ]
    q = quva(MAIN)
    numbers = {lab: r for lab, r in rows}
    numbers["QUVA zero-shot"] = q
    (out / "final_numbers_frozen.json").write_text(json.dumps(numbers, indent=1))

    def nseeds(r):
        return max((r[s][0][2] for s, _ in DS if r[s][0]), default=0)

    best = {(s, i): (min if i == 0 else max)([r[s][i][0] for _, r in rows if r[s][i]]) for s, _ in DS for i in (0, 1)}
    main_lines = [f"{lab} & {nseeds(r)} & {cell(r['rc'][0], best[('rc', 0)])} & {cell(r['rc'][1], best[('rc', 1)])}"
                  + r" \\" for lab, r in rows]
    app_lines = [f"{lab} & {nseeds(r)} & " + " & ".join(cell(r[s][i], best[(s, i)]) for s, _ in DS for i in (0, 1))
                 + r" \\" for lab, r in rows]
    (out / "ablation_rows.tex").write_text("\n".join(main_lines) + "\n")
    (out / "app_ablation_rows.tex").write_text("\n".join(app_lines) + "\n")
    f = rows[0][1]
    (out / "main_row.tex").write_text(f"{cell(f['rc'][0])} & {cell(f['rc'][1])}\n")
    (out / "app_row.tex").write_text(" & ".join([cell(f["ucf"][0]), cell(f["ucf"][1]), cell(f["cx"][0]), cell(f["cx"][1]),
                                                 cell(q[0])]) + "\n")
    print("\n".join(app_lines))
    print("QUVA", q)


if __name__ == "__main__":
    main()
