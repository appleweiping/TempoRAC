"""Mechanism analysis for a trained TempoRAC run.

For every base-grid cell inside an annotated cycle, relate the routing weights
g_s(t) to the ground-truth local period (cycle length, seconds).  Reports mean
routing weight per GT-period bin, routing entropy, and -- for density-level
mixing -- the count error of each rate expert alone vs. the routed mixture.
GT is used only for analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.data import BASE, load_videos, period_target, to_device
from temporac_cvpr.models import build_model
from temporac_cvpr.train import EvalLoader


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    device = torch.device("cuda")
    vids = load_videos(json.loads(Path(args.manifest).read_text()), args.features, cfg["rates"], {args.split},
                       cfg["pool_to"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["state"])
    loader = EvalLoader(vids, cfg, 2)
    per_s, routes, ents = [], [], []
    expert_counts, mix_counts, gts = [], [], []
    for b, batch in zip(loader.batches, loader.loader):
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        g = out["route"].float().cpu().numpy()
        for i, v in enumerate(b):
            G = int(batch["mask"][i].sum())
            p, on = period_target(v.cycles, v.T)
            sel = on[:G] > 0
            per_s.append(p[:G][sel] * BASE / v.fps)
            routes.append(g[i, :G][sel])
            gi = np.clip(g[i, :G], 1e-8, 1)
            ents.append(float(-(gi * np.log(gi)).sum(-1).mean()))
            if "expert_density" in out:
                expert_counts.append(out["expert_density"][i, :G].float().sum(0).cpu().numpy())
                mix_counts.append(float(out["density"][i, :G].float().sum()))
                gts.append(v.count)
    per_s = np.concatenate(per_s)
    routes = np.concatenate(routes)
    edges = np.quantile(per_s, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rep = {"rates": cfg["rates"], "mean_route_entropy": float(np.mean(ents)), "bins": []}
    for k in range(5):
        m = (per_s >= edges[k]) & (per_s <= edges[k + 1])
        rep["bins"].append({"period_s": [float(edges[k]), float(edges[k + 1])], "cells": int(m.sum()),
                            "mean_route": routes[m].mean(0).round(4).tolist()})
    if expert_counts:
        ec = np.array(expert_counts)
        gt = np.array(gts)
        pos = gt > 0
        f = lambda c: float(np.mean(np.abs(np.round(c[pos]) - gt[pos]) / (gt[pos] + 0.1)))
        rep["expert_alone_mae_p"] = {str(s): f(ec[:, j]) for j, s in enumerate(cfg["rates"])}
        rep["mixture_mae_p"] = f(np.array(mix_counts))
        rep["oracle_best_expert_mae_p"] = float(np.mean(np.min(
            np.abs(np.round(ec[pos]) - gt[pos, None]), 1) / (gt[pos] + 0.1)))
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
