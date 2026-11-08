"""Evaluate a saved run (best.pt) on another split, e.g. the tempo-warp stress test.

Writes RUN/eval_<tag>.json with per-video predictions and metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from temporac_cvpr.data import load_videos
from temporac_cvpr.models import build_model
from temporac_cvpr.train import predict, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    manifest = []
    for m in args.manifest:
        manifest += json.loads(Path(m).read_text())
    cache = {}
    for run in args.run:
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
        cfg = ck["cfg"]
        key = (tuple(cfg["rates"]), cfg["pool_to"])
        if key not in cache:
            cache[key] = load_videos(manifest, args.features, cfg["rates"], {args.split}, cfg["pool_to"])
        vids = cache[key]
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["state"])
        preds = predict(model, vids, cfg, device)
        res = {"split": args.split, "n": len(vids), "metrics": score(preds, vids), "predictions": preds,
               "best_epoch": ck.get("epoch")}
        tag = args.tag or args.split
        (Path(run) / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
        print(Path(run).name, tag, json.dumps(res["metrics"]), flush=True)


if __name__ == "__main__":
    main()
