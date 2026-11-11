"""Rate knock-out analysis of the final (joint, two-depth, count-adapted) decoders, validation splits only.

For every benchmark and each of the five decoders (checkpoint selected on that benchmark's validation split),
predict the original view with all rates and with one rate's pooled tokens zeroed at inference (the model's
`knockout` hook), and report MAE per tempo tertile of the annotated cycle duration (seconds).
Usage: python -m temporac_cvpr.rate_knockout2 OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.data import load_videos, to_device
from temporac_cvpr.models import build_model
from temporac_cvpr.train import EvalLoader, load_rates

R = Path("/root/autodl-tmp/TempoRAC")
VAL = {"RepCountA": "rc_valid", "Countix": "cx_val", "UCFRep": "ucf_dev"}


@torch.no_grad()
def main() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    man = json.loads((R / "data/UNION/manifest_dev.json").read_text())
    feats = R / "features/UNION_stmaeA_b8_p4"
    device = torch.device("cuda")
    summary = {}
    for ds, split in VAL.items():
        res, tert_q = {}, None
        for s in range(5):
            run = R / "runs/adapt_dev" / f"wo_llm_adapted_s{s}"
            ck = torch.load(run / f"best_{split}.pt", map_location="cpu", weights_only=True)
            cfg = ck["cfg"]
            vids = load_videos(man, str(feats), load_rates(cfg), {split}, cfg["pool_to"], tuple(cfg.get("aux_features", ())))
            vids = [v for v in vids if v.count > 0]
            for v in vids:
                v.domain = cfg["domains"][v.dataset]
            per = np.array([v.T / v.fps / v.count for v in vids])  # annotated cycle duration (s), stratification only
            q = np.quantile(per, [1 / 3, 2 / 3])
            tert_q = q.tolist()
            tert = np.where(per < q[0], 0, np.where(per < q[1], 1, 2))
            model = build_model(cfg).to(device).eval()
            model.load_state_dict(ck["state"])
            loader = EvalLoader(vids, cfg, 2)
            for name, ko in [("all", [])] + [(f"-r{r}", [r]) for r in cfg["rates"]]:
                preds = {}
                for b, batch in zip(loader.batches, loader.loader):
                    batch = to_device(batch, device)
                    if ko:
                        batch["knockout"] = ko
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(batch)
                    for v, c in zip(b, out["density"].float().sum(1).cpu().numpy()):
                        preds[v.key] = float(c)
                err = np.array([abs(round(preds[v.key]) - v.count) / (v.count + 0.1) for v in vids])
                res.setdefault(name, []).append([float(err[tert == t].mean()) for t in range(3)] + [float(err.mean())])
        summary[ds] = {"tertiles_s": tert_q, "columns": ["fast", "medium", "slow", "all"],
                       "mae_p": {k: np.mean(v, 0).round(4).tolist() for k, v in res.items()}}
        print(ds, json.dumps(summary[ds]), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
