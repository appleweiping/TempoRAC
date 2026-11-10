"""Per-video comparison of two multi-target runs on validation (two-view average predictions).

Reports MAE, harmonic errors (prediction/ground-truth ratio > 1.6 or < 0.65) fixed or introduced by
run B relative to run A, win/loss counts, and error/bias by count range.  Validation splits only.
Usage: python -m temporac_cvpr.compare_preds RUN_A RUN_B [--splits rc_valid,ucf_dev,cx_val]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAN = Path("/root/autodl-tmp/TempoRAC/data/UNION/manifest_dev.json")


def tta(run: Path, t: str) -> dict:
    P = json.loads((run / f"predictions_{t}.json").read_text())
    return {k: 0.5 * (P["val"][k] + P["val_shift"][k]) for k in P["val"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--splits", default="rc_valid,ucf_dev,cx_val")
    args = ap.parse_args()
    man = {}
    for it in json.loads(MAN.read_text()):
        man[it["split"] + "__" + it["video_id"]] = it
    for t in args.splits.split(","):
        A, B = tta(Path(args.a), t), tta(Path(args.b), t)
        ks = [k for k in A if man[k]["count"] > 0]
        g = np.array([man[k]["count"] for k in ks], float)
        a, b = np.array([A[k] for k in ks]), np.array([B[k] for k in ks])
        ea, eb = np.abs(np.round(a) - g) / (g + .1), np.abs(np.round(b) - g) / (g + .1)
        cat = lambda r: np.where(r > 1.6, 2, np.where(r < 0.65, -1, 0))
        ca, cb = cat(a / g), cat(b / g)
        print(f"== {t} n={len(ks)} A MAE {ea.mean():.4f} B MAE {eb.mean():.4f}")
        print(f"   harmonic errors: A {int((ca != 0).sum())} (x2 {int((ca == 2).sum())}, x0.5 {int((ca == -1).sum())})"
              f"  B {int((cb != 0).sum())} (x2 {int((cb == 2).sum())}, x0.5 {int((cb == -1).sum())})"
              f" | fixed by B {int(((ca != 0) & (cb == 0)).sum())}  introduced by B {int(((ca == 0) & (cb != 0)).sum())}")
        d = eb - ea
        print(f"   per video: B better {int((d < -1e-9).sum())}  worse {int((d > 1e-9).sum())}  same {int((np.abs(d) <= 1e-9).sum())}")
        for lo, hi in [(0, 4), (4, 11), (11, 10 ** 6)]:
            m = (g >= lo) & (g < hi)
            if m.any():
                print(f"   count [{lo},{hi if hi < 10 ** 6 else 'inf'}) n={int(m.sum()):4d}  A {ea[m].mean():.3f} "
                      f"(bias {np.mean(np.round(a[m]) - g[m]):+.2f})  B {eb[m].mean():.3f} (bias {np.mean(np.round(b[m]) - g[m]):+.2f})")


if __name__ == "__main__":
    main()
