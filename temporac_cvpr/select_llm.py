"""Pre-registered selection of the language-model stage variant (validation only; rule logged 11-11).

For every variant group and every finished seed s, compare GROUP/s{s}_count with the stage-1 decoder
it was initialised from (adapt_dev/wo_llm_adapted_s{s}):
  primary   = relative change of the two-view validation MAE at the selected epoch (val_tta mae_p),
  secondary = relative change of the mean original-view validation MAE over the last 3 epochs,
each averaged over the three validation splits, then over seeds.  A variant qualifies only if its
primary mean is below 0 (better than the decoder); the qualifying variant with the lower primary mean
is selected.
Usage: python -m temporac_cvpr.select_llm RUNS_ROOT GROUP [GROUP ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

T = ["rc_valid", "cx_val", "ucf_dev"]


def tta(d: Path) -> list[float]:
    return [json.loads((d / f"result_{t}.json").read_text())["val_tta"]["mae_p"] for t in T]


def last3(d: Path) -> list[float]:
    L = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]
    return [float(np.mean([e["val_all"][t]["mae_p"] for e in L[-3:]])) for t in T]


def main() -> None:
    root = Path(sys.argv[1])
    summary = {}
    for g in sys.argv[2:]:
        prim, sec = [], []
        for d in sorted((root / g).glob("s*_count")):
            if not (d / "result.json").exists():
                continue
            s = d.name.split("_")[0][1:]
            b = root / "adapt_dev" / f"wo_llm_adapted_s{s}"
            a1, b1, a3, b3 = tta(d), tta(b), last3(d), last3(b)
            r1 = [x / y - 1 for x, y in zip(a1, b1)]
            r3 = [x / y - 1 for x, y in zip(a3, b3)]
            prim.append(np.mean(r1))
            sec.append(np.mean(r3))
            print(f"{g} s{s}: two-view {np.round(a1, 4)} vs base {np.round(b1, 4)} rel {np.round(r1, 3)} mean {np.mean(r1):+.4f} | "
                  f"last3 {np.round(a3, 4)} vs {np.round(b3, 4)} mean {np.mean(r3):+.4f}")
        if prim:
            summary[g] = (float(np.mean(prim)), float(np.mean(sec)), len(prim))
            print(f"== {g}: seeds {len(prim)}  primary {np.mean(prim):+.4f}  secondary {np.mean(sec):+.4f}")
    ok = {g: v for g, v in summary.items() if v[0] < 0}
    print("SELECTED:", min(ok, key=lambda g: ok[g][0]) if ok else "none (fallback: windowed small-count LLM design)")


if __name__ == "__main__":
    main()
