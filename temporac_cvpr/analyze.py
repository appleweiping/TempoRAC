"""Stratified error analysis and paired comparisons from saved per-video predictions.

Strata are computed from ground-truth annotations only (never from predictions):
  * tempo variability: coefficient of variation of annotated cycle lengths
  * tempo: median annotated cycle length in seconds
  * count magnitude
Usage: python -m temporac_cvpr.analyze --manifest M.json --split valid --runs DIR1 DIR2 ... [--ref DIR1]
Each run DIR may be a single run (predictions.json) or several seeds via comma: a,b,c.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temporac_cvpr.data import cycles_from_bounds, tempo_cv


def video_stats(manifest: list[dict], split: str, fps_lookup: dict) -> dict:
    out = {}
    for it in manifest:
        if it["split"] != split or it["count"] is None or float(it["count"]) <= 0:
            continue  # official protocol: positive-count videos only
        key = f"{it['split']}__{it['video_id']}"
        cyc = cycles_from_bounds(it.get("bounds", []))
        fps = fps_lookup.get(key) or it.get("fps_csv") or 30.0
        lens = np.abs(cyc[:, 1] - cyc[:, 0]) if len(cyc) else np.array([])
        out[key] = {
            "count": float(it["count"]),
            "cv": tempo_cv(cyc),
            "period_s": float(np.median(lens) / fps) if len(lens) else float("nan"),
            "action": it.get("action", ""),
        }
    return out


def make_strata(stats: dict) -> dict:
    """Tertile strata on the evaluated split (GT-derived only)."""
    out = {}
    for field, short in (("cv", "cv"), ("period_s", "per"), ("count", "cnt")):
        v = np.array([s[field] for s in stats.values() if not np.isnan(s[field])])
        q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
        out[short] = [(f"{short}<{q1:.2f}", lambda s, f=field, a=q1: s[f] < a),
                      (f"{short}[{q1:.2f},{q2:.2f})", lambda s, f=field, a=q1, b=q2: a <= s[f] < b),
                      (f"{short}>={q2:.2f}", lambda s, f=field, b=q2: s[f] >= b)]
    return out


def per_video_err(preds: dict, stats: dict) -> dict:
    e = {}
    for k, s in stats.items():
        p = np.round(preds[k])
        e[k] = (abs(p - s["count"]) / (s["count"] + 0.1), float(abs(p - s["count"]) <= 1))
    return e


def load_preds(spec: str, split: str) -> list[dict]:
    out = []
    for d in spec.split(","):
        p = json.loads((Path(d) / "predictions.json").read_text())
        out.append(p["val" if split in ("valid", "dev") else split] if split not in p else p[split])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    fps = {}
    if args.features:
        for f in Path(args.features, "meta").glob(f"{args.split}__*.json"):
            m = json.loads(f.read_text())
            fps[f.stem] = m.get("fps")
    stats = video_stats(manifest, args.split, fps)
    keys = sorted(stats)
    STRATA = make_strata(stats)
    print("strata sizes:", {g: {n: sum(fn(stats[k]) for k in keys) for n, fn in v} for g, v in STRATA.items()})
    errs = {}
    for spec in args.runs:
        seeds = [per_video_err(p, stats) for p in load_preds(spec, args.split)]
        mae = np.mean([[s[k][0] for k in keys] for s in seeds], 0)
        obo = np.mean([[s[k][1] for k in keys] for s in seeds], 0)
        errs[spec] = (mae, obo)
        row = [f"{Path(spec.split(',')[0]).name}[{len(seeds)}]", f"all {mae.mean():.3f}/{obo.mean():.3f}"]
        for g, v in STRATA.items():
            for n, fn in v:
                idx = np.array([fn(stats[k]) for k in keys])
                row.append(f"{n} {mae[idx].mean():.3f}/{obo[idx].mean():.3f}")
        print(" | ".join(row))
    if len(args.runs) > 1:
        ref = args.runs[0]
        rng = np.random.default_rng(0)
        n = len(keys)
        idx = rng.integers(0, n, size=(args.boot, n))
        for spec in args.runs[1:]:
            d = errs[spec][0] - errs[ref][0]
            bs = d[idx].mean(1)
            print(f"MAE diff {Path(spec.split(',')[0]).name} - {Path(ref.split(',')[0]).name}: {d.mean():+.4f} "
                  f"95%CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  P(diff<0)={np.mean(bs < 0):.3f}")


if __name__ == "__main__":
    main()
