"""Diagnostic (training + validation rows only): view error by true cycle length, OOF vs validation.

If the per-bin view errors agree and only the tempo distributions differ, a selector that sees the tempo
transfers; if the per-bin errors themselves differ, the OOF decoders behave differently on training
videos (e.g. memorised near-duplicates) and OOF rows are a biased training source.
Usage: python -m temporac_cvpr.oof_tempo OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BINS = [0, 0.4, 0.8, 1.6, 3.2, 1e9]  # true cycle duration, seconds


def mae(p, g):
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1))) if len(g) else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in (Path(sys.argv[1]) / "rows.jsonl").read_text().splitlines() if l.strip()]
    for ds in ("RepCountA", "Countix", "UCFRep"):
        print(f"== {ds}   bins of true cycle duration (s): n / MAE 1x / 2x / avg")
        for role in ("oof", "val"):
            S = [r for r in rows if r["role"] == role and r["ds"] == ds]
            per = np.array([r["dur"] / r["gt"] for r in S])
            g = np.array([r["gt"] for r in S]); c1 = np.array([r["c"]["1x"] for r in S]); c2 = np.array([r["c"]["2x"] for r in S])
            cells = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                m = (per >= lo) & (per < hi)
                cells.append(f"[{lo:.1f},{hi if hi < 1e8 else 'inf'}) {m.mean():.2f} {mae(c1[m], g[m]):.3f}/{mae(c2[m], g[m]):.3f}/{mae((c1[m] + c2[m]) / 2, g[m]):.3f}")
            print(f"  {role:3s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
