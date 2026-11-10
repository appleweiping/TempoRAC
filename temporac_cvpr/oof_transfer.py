"""Diagnostic (training + validation rows only): does "which view is right" transfer from out-of-fold
training predictions to validation predictions?

Per dataset: OOF MAE of 1x / 2x / average / oracle; a class-label gate fitted on OOF rows (fraction of
OOF videos of the class where 2x had the lower error, stake-weighted) applied to validation rows; the
per-class agreement between OOF and validation (Pearson r of the per-class 2x-better fractions).
Usage: python -m temporac_cvpr.oof_transfer OUT_DIR
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC")


def mae(p, g):
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1)))


def main() -> None:
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines() if l.strip()]
    man = {f"{it['split']}__{it['video_id']}": it for it in json.loads((R / "data/UNION/manifest_dev.json").read_text())}

    def cls(r):
        it = man[r["key"]]
        if it.get("action"):
            return str(it["action"]).lower()
        km = it.get("kinetics_member") or ""
        return km.split("/")[1] if km.count("/") >= 2 else "?"

    for ds in ("RepCountA", "Countix", "UCFRep"):
        O = [r for r in rows if r["role"] == "oof" and r["ds"] == ds]
        V = [r for r in rows if r["role"] == "val" and r["ds"] == ds]
        g = np.array([r["gt"] for r in O]); c1 = np.array([r["c"]["1x"] for r in O]); c2 = np.array([r["c"]["2x"] for r in O])
        e1 = np.abs(np.round(c1) - g) / (g + .1); e2 = np.abs(np.round(c2) - g) / (g + .1)
        print(f"== {ds} OOF n={len(O)}: 1x {mae(c1, g):.4f} 2x {mae(c2, g):.4f} avg {mae((c1 + c2) / 2, g):.4f} "
              f"oracle {float(np.mean(np.minimum(e1, e2))):.4f}  2x-better {np.mean(e2 < e1):.2f} 1x-better {np.mean(e1 < e2):.2f}")
        num, den = defaultdict(float), defaultdict(float)
        for r, a, b in zip(O, e1, e2):
            num[cls(r)] += abs(a - b) * (b < a)
            den[cls(r)] += abs(a - b)
        gate = {k: num[k] / den[k] for k in den if den[k] > 0}
        gv = np.array([r["gt"] for r in V]); v1 = np.array([r["c"]["1x"] for r in V]); v2 = np.array([r["c"]["2x"] for r in V])
        w = np.array([gate.get(cls(r), 0.5) for r in V])
        f1 = np.abs(np.round(v1) - gv) / (gv + .1); f2 = np.abs(np.round(v2) - gv) / (gv + .1)
        print(f"   VAL (5 seeds pooled): avg {mae((v1 + v2) / 2, gv):.4f}  class gate fitted on OOF {mae((1 - w) * v1 + w * v2, gv):.4f}"
              f"  oracle {float(np.mean(np.minimum(f1, f2))):.4f}  2x-better {np.mean(f2 < f1):.2f} 1x-better {np.mean(f1 < f2):.2f}")
        vn, vd = defaultdict(float), defaultdict(float)
        for r, a, b in zip(V, f1, f2):
            vn[cls(r)] += abs(a - b) * (b < a)
            vd[cls(r)] += abs(a - b)
        common = [k for k in gate if vd.get(k, 0) > 0]
        if len(common) > 2:
            x = np.array([gate[k] for k in common]); y = np.array([vn[k] / vd[k] for k in common])
            print(f"   per-class 2x-better (stake-weighted) OOF vs VAL: r = {np.corrcoef(x, y)[0, 1]:+.2f} over {len(common)} classes")


if __name__ == "__main__":
    main()
