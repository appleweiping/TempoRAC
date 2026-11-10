"""Main-table row and ablation table from the single test evaluation of every final model.

Full model: the stage-1 decoders (adapt_dev/wo_llm_adapted_s0..4) whose exact tempo views are weighted by
the language tempo selector (tempo_select.py apply -> runs/tsel/<SELECTOR>_final.json); mean +- sample SD
over the five decoder seeds.  Rows marked with a dagger change the decoder and are compared with the
"plain two-view average" row (the decoder without the language model).  Writes paper fragments + JSON.
Usage: python -m temporac_cvpr.final_tables SELECTOR [NODESC_SELECTOR KNN_SELECTOR MLP_SELECTOR] OUT_DIR
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC/runs")
DS = [("rc", "RepCountA"), ("ucf", "UCFRep"), ("cx", "Countix")]
VALS = {"rc": "rc_valid", "ucf": "ucf_dev", "cx": "cx_val"}


def ms(xs):
    xs = np.asarray(xs, float)
    return (float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, len(xs)) if len(xs) else None


def selector_rows(name: str) -> dict:
    f = R / "tsel" / f"{name}_final.json"
    if not f.exists():
        return {s: (None, None) for s, _ in DS + [("quva", "QUVA")]}
    res = json.loads(f.read_text())["results"]
    out = {}
    for s, ds in DS:
        v = res["test"]["per_ds"].get(ds)
        out[s] = ((v["mae"], v["mae_sd"], v["seeds"]), (v["obo"], v["obo_sd"], v["seeds"])) if v else (None, None)
    q = res.get("quva", {}).get("per_ds", {}).get("QUVA")
    out["quva"] = ((q["mae"], q["mae_sd"], q["seeds"]), (q["obo"], q["obo_sd"], q["seeds"])) if q else (None, None)
    return out


def saved_rows(pattern: str, key: str = "metrics_tta") -> dict:
    out = {}
    for s, _ in DS:
        vals = [json.loads(Path(f).read_text())[key] for f in sorted(glob.glob(str(R / pattern / f"eval_test_{s}.json")))]
        out[s] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def joint_rows() -> dict:
    out = {}
    for s, _ in DS:
        vals = [json.loads(Path(f).read_text())["metrics_tta"]
                for f in sorted(glob.glob(str(R / "adapt_abl_single" / f"wo_joint_{s}_s*" / f"eval_test_{s}.json")))]
        out[s] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def result_rows(pattern: str) -> dict:  # runs trained with the test split (final_v2/v3): result_<val>.json["test_tta"]
    out = {}
    for s, _ in DS:
        vals = [json.loads(Path(f).read_text())["test_tta"] for f in sorted(glob.glob(str(R / pattern / f"result_{VALS[s]}.json")))]
        out[s] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def llm_refine_rows() -> dict:  # video-token LLM refining a frozen decoder (llm_frozen, eval_llm, two-view density)
    out = {}
    for s, _ in DS:
        vals = [json.loads(Path(f).read_text())["metrics"]["density_tta"]
                for f in sorted(glob.glob(str(R / "llm_frozen" / "s*_count" / f"eval_llm_{s}_test.json")))]
        out[s] = (ms([v["mae_p"] for v in vals]), ms([v["obo_p"] for v in vals]))
    return out


def cell(x, best=None, digits=3):
    if x is None:
        return "--"
    m, s, n = x
    t = f"{m:.{digits}f}{{\\scriptsize$\\pm${s:.{digits}f}}}"
    return f"\\best{{{t}}}" if best is not None and abs(m - best) < 1e-12 else t


def main() -> None:
    sel = sys.argv[1]
    others = sys.argv[2:-1]
    out = Path(sys.argv[-1])
    out.mkdir(parents=True, exist_ok=True)
    rows = [(r"\method (full)", selector_rows(sel))]
    labels = {"nodesc": r"\quad selector without the description (counts only)",
              "knn": r"\quad description nearest neighbours instead of the language model",
              "mlp": r"\quad MLP on the description embedding instead of the language model"}
    for o in others:
        kind = "nodesc" if "nodesc" in o else ("knn" if o.startswith("knn") else "mlp")
        rows.append((labels[kind], selector_rows(o)))
    rows += [
        (r"\quad $-$ language model (plain two-view average)", saved_rows("adapt_dev/wo_llm_adapted_s*")),
        (r"\quad language model refining the frozen decoder instead", llm_refine_rows()),
        (r"\quad $-$ encoder adaptation\textsuperscript{\dag}", result_rows("final_v2/wo_llm_s*")),
        (r"\quad\quad $+$ frozen-MLLM FiLM conditioning\textsuperscript{\dag}", result_rows("final_v3/full_s*")),
        (r"\quad $-$ multi-rate pyramid ($s{=}4$ only)\textsuperscript{\dag}", saved_rows("adapt_abl/wo_pyramid_s*")),
        (r"\quad $-$ pyramid shift\textsuperscript{\dag}", saved_rows("adapt_abl/wo_shift_s*", "metrics")),
        (r"\quad $-$ second depth (block 12 only)\textsuperscript{\dag}", saved_rows("adapt_abl/wo_depth_s*")),
        (r"\quad $-$ dataset token\textsuperscript{\dag}", saved_rows("adapt_abl/wo_token_s*")),
        (r"\quad $-$ balanced sampling\textsuperscript{\dag}", saved_rows("adapt_abl/wo_balance_s*")),
        (r"\quad $-$ joint training (one model per dataset)\textsuperscript{\dag}", joint_rows()),
    ]
    (out / "final_numbers.json").write_text(json.dumps({lab: r for lab, r in rows}, indent=1))
    best = {}
    for s, _ in DS:
        for i in (0, 1):
            vals = [r[s][i][0] for _, r in rows if r.get(s) and r[s][i]]
            best[(s, i)] = (min if i == 0 else max)(vals) if vals else None
    lines = []
    for lab, r in rows:
        cells = [cell(r[s][i], best[(s, i)]) for s, _ in DS for i in (0, 1)]
        n = max((r[s][0][2] for s, _ in DS if r[s][0]), default=0)
        lines.append(f"{lab} & {n} & " + " & ".join(cells) + r" \\")
    (out / "ablation_rows.tex").write_text("\n".join(lines) + "\n")
    f = rows[0][1]
    (out / "main_row.tex").write_text(" & ".join(cell(f[s][i]) for s, _ in DS for i in (0, 1)) + "\n")
    print("\n".join(lines))
    print("QUVA", f.get("quva"))


if __name__ == "__main__":
    main()
