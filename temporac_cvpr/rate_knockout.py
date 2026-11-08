"""Rate knock-out analysis: which pyramid level does a trained model rely on, per tempo?

For each saved run, evaluate on a split with all rates, and with each rate's pooled
tokens set to zero at inference.  Reports MAE_p per GT tempo tertile (median cycle
length, seconds) for every condition, averaged over runs.  GT only stratifies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.analyze import video_stats
from temporac_cvpr.data import load_videos, to_device
from temporac_cvpr.models import build_model
from temporac_cvpr.train import EvalLoader


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta-features", required=True, help="feature dir with meta (fps)")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    man = json.loads(Path(args.manifest).read_text())
    fps = {f.stem: json.loads(f.read_text()).get("fps") for f in Path(args.meta_features, "meta").glob(f"{args.split}__*.json")}
    st = video_stats(man, args.split, fps)
    keys = sorted(st)
    q = np.quantile([st[k]["period_s"] for k in keys], [1 / 3, 2 / 3])
    tert = {k: (0 if st[k]["period_s"] < q[0] else 1 if st[k]["period_s"] < q[1] else 2) for k in keys}
    device = torch.device("cuda")
    res = {}
    for run in args.runs:
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
        cfg = ck["cfg"]
        vids = load_videos(man, args.features, cfg["rates"], {args.split}, cfg["pool_to"])
        model = build_model(cfg).to(device).eval()
        model.load_state_dict(ck["state"])
        loader = EvalLoader(vids, cfg, 2)
        conds = [("all", [])] + [(f"-r{s}", [s]) for s in cfg["rates"]]
        for name, ko in conds:
            preds = {}
            for b, batch in zip(loader.batches, loader.loader):
                batch = to_device(batch, device)
                if ko:
                    batch["knockout"] = ko
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch)
                for v, c in zip(b, out["density"].float().sum(1).cpu().numpy()):
                    preds[v.key] = float(c)
            err = {k: abs(round(preds[k]) - st[k]["count"]) / (st[k]["count"] + 0.1) for k in keys}
            row = [float(np.mean([err[k] for k in keys if tert[k] == t])) for t in range(3)] + [float(np.mean(list(err.values())))]
            res.setdefault(name, []).append(row)
    summary = {"tertiles_s": [float(q[0]), float(q[1])], "columns": ["fast", "medium", "slow", "all"],
               "mae_p": {k: np.mean(v, 0).round(4).tolist() for k, v in res.items()}, "n_runs": len(args.runs)}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
