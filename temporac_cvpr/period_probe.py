"""Accuracy of the local period read from a trained model's tempogram (diagnostic).

Compares, per annotated cycle centre, the tempogram's dominant lag (soft-argmax in
log-lag space as in TempoGate, and the hard local-max argmax) with the annotated
cycle length.  GT is used only to score the estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.data import BASE, load_videos, to_device
from temporac_cvpr.models import build_model, to_base
from temporac_cvpr.train import EvalLoader


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    device = torch.device("cuda")
    vids = load_videos(json.loads(Path(args.manifest).read_text()), args.features, cfg["rates"], {args.split}, cfg["pool_to"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["state"])
    loader = EvalLoader(vids, cfg, 2)
    r0 = cfg["rates"][0]
    soft, hard, gt = [], [], []
    for b, batch in zip(loader.batches, loader.loader):
        batch = to_device(batch, device)
        G = batch["mask"].shape[1]
        xs = to_base(model.pools[str(r0)](batch["feats"][r0].float()), r0, G)
        A, sup = model.tempo(xs, batch["mask"])
        w = sup.float()
        n = w.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (A * w).sum(-1, keepdim=True) / n
        sd = (((A - mu) ** 2) * w).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-3)
        Az = ((A - mu) / sd).masked_fill(w <= 0, -1e4)
        Az[..., 0] = -1e4
        lags = torch.arange(1, A.shape[-1] + 1, device=device).float()
        pr = torch.softmax(Az / 0.5, -1)
        ps = torch.exp((pr * torch.log(lags)).sum(-1))
        # hard: strongest local maximum
        loc = (Az[..., 1:-1] >= Az[..., :-2]) & (Az[..., 1:-1] >= Az[..., 2:])
        Am = torch.full_like(Az, -1e4)
        Am[..., 1:-1] = torch.where(loc, Az[..., 1:-1], torch.full_like(Az[..., 1:-1], -1e4))
        ph = Am.argmax(-1).float() + 1
        for i, v in enumerate(b):
            for a0, a1 in v.cycles:
                c = int(round((a0 + a1) / 2 / BASE))
                if c >= G:
                    continue
                soft.append(float(ps[i, c]))
                hard.append(float(ph[i, c]))
                gt.append((a1 - a0) / BASE)
    gt, soft, hard = map(np.array, (gt, soft, hard))
    ok = gt >= 2
    rep = {"cycles": int(ok.sum())}
    for name, e in (("soft", soft), ("hard_localmax", hard)):
        r = e[ok] / gt[ok]
        rep[name] = {"within_x1.25": float(np.mean((r > 0.8) & (r < 1.25))),
                     "within_x1.5": float(np.mean((r > 2 / 3) & (r < 1.5))),
                     "half": float(np.mean((r > 0.4) & (r <= 2 / 3))), "double": float(np.mean((r >= 1.5) & (r < 2.5))),
                     "log_corr": float(np.corrcoef(np.log(e[ok]), np.log(gt[ok]))[0, 1])}
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
