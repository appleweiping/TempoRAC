"""Dump predicted vs target densities for selected validation videos (diagnostic only).

For each video: count, predicted sum, the dominant period of the target and of the
prediction (autocorrelation peak on the base grid), and the mean density level in the
annotated repetition span.  Answers whether under-counting comes from a lowered flat
level (frequency regressed to the mean) or from missed cycles (gaps in the density).

Usage: python -m temporac_cvpr.density_probe RUN_DIR MANIFEST FEATURES SPLIT [--min-count 20] [--n 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.data import collate, load_videos
from temporac_cvpr.models import build_model
from temporac_cvpr.train import load_rates


def dom_period(x: np.ndarray, lo: int = 2, hi: int = 200) -> float:
    x = x - x.mean()
    if len(x) < 2 * lo + 2 or not np.any(x):
        return float("nan")
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(hi, len(x) // 2)
    if hi <= lo:
        return float("nan")
    return float(lo + np.argmax(ac[lo:hi]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("manifest")
    ap.add_argument("features")
    ap.add_argument("split")
    ap.add_argument("--min-count", type=float, default=20)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    import resource  # one memory-mapped file per video and rate

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 16) if hard > 0 else 1 << 16, hard))
    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    man = json.loads(Path(args.manifest).read_text())
    vids = load_videos(man, args.features, load_rates(cfg), {args.split}, cfg["pool_to"],
                       tuple(cfg.get("aux_features", ())))
    preds = json.loads((Path(args.run) / "predictions.json").read_text())["val"]
    sel = [v for v in vids if v.count >= args.min_count]
    sel.sort(key=lambda v: (preds[v.key] - v.count) / v.count)  # most under-counted first
    sel = sel[: args.n]
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(ck["state"])
    rows = []
    for v in sel:
        b = collate([v], cfg["rates"], cfg["sigma_frac"], cfg["density_kind"], torch.device("cuda"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            d = model(b)["density"].float()[0].cpu().numpy()
        y = b["density"][0].float().cpu().numpy()  # the training target (respects per-video target kinds)
        G = min(len(d), len(y))
        d, y = d[:G], y[:G]
        on = y > 1e-4
        a, z = (np.argmax(on), G - np.argmax(on[::-1])) if on.any() else (0, G)
        rows.append({
            "key": v.key, "count": v.count, "pred_sum": float(d.sum()), "T": v.T, "cells": G,
            "span_cells": int(z - a), "period_target": dom_period(y[a:z]), "period_pred": dom_period(d[a:z]),
            "level_target": float(y[a:z].mean()), "level_pred": float(d[a:z].mean()),
            "mass_outside_span": float(d[:a].sum() + d[z:].sum()),
            "cv_pred_in_span": float(d[a:z].std() / (d[a:z].mean() + 1e-9)),
        })
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rows[-1].items()}), flush=True)
    (Path(args.run) / f"density_probe_{args.split}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
