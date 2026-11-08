"""Does the tempogram separate the true period from its half? (diagnostic, GT used for analysis only)

For every annotated cycle of every video in a split, read the z-scored tempogram of a
trained model at the annotated period p and at p/2 (centre cell of the cycle), and
report how often A(p) > A(p/2) overall and for videos the model over-counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.data import BASE, load_videos, to_device
from temporac_cvpr.models import build_model, TempoConsistentDecoder
from temporac_cvpr.train import EvalLoader


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--signal", default="state", choices=["state", "velocity", "phase"])
    args = ap.parse_args()
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    ck = torch.load(Path(args.run) / "best.pt", map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    device = torch.device("cuda")
    vids = load_videos(json.loads(Path(args.manifest).read_text()), args.features, cfg["rates"], {args.split}, cfg["pool_to"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["state"])
    preds = json.loads((Path(args.run) / "predictions.json").read_text())["val"]
    loader = EvalLoader(vids, cfg, 2)
    rows = []
    for b, batch in zip(loader.batches, loader.loader):
        batch = to_device(batch, device)
        G = batch["mask"].shape[1]
        from temporac_cvpr.models import to_base

        xs = to_base(model.pools[str(cfg["rates"][0])](batch["feats"][cfg["rates"][0]].float()), cfg["rates"][0], G)
        if args.signal == "velocity":  # signed temporal difference of the pooled features
            xs = torch.cat([xs[:, 1:] - xs[:, :-1], torch.zeros_like(xs[:, :1])], 1)
        elif args.signal == "phase":  # state and velocity jointly (unit-normalised halves)
            v = torch.cat([xs[:, 1:] - xs[:, :-1], torch.zeros_like(xs[:, :1])], 1)
            xs = torch.cat([torch.nn.functional.normalize(xs, dim=-1), torch.nn.functional.normalize(v, dim=-1)], -1)
        A, sup = model.tempo(xs, batch["mask"])
        w = sup.float()
        n = w.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (A * w).sum(-1, keepdim=True) / n
        sd = (((A - mu) ** 2) * w).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-3)
        Az = ((A - mu) / sd * w).float()
        for i, v in enumerate(b):
            if not v.count or v.count <= 0:
                continue
            ratio = round(preds[v.key]) / v.count
            for a0, a1 in v.cycles:
                p = (a1 - a0) / BASE
                c = int(round((a0 + a1) / 2 / BASE))
                if p < 2 or p > A.shape[-1] or c >= G:
                    continue
                full, _ = TempoConsistentDecoder._read(Az[i:i + 1, c:c + 1], torch.tensor([[p]], device=device))
                half, _ = TempoConsistentDecoder._read(Az[i:i + 1, c:c + 1], torch.tensor([[p / 2]], device=device))
                rows.append((float(full), float(half), ratio, v.key))
    r = np.array([(a, b, c) for a, b, c, _ in rows])
    over = r[:, 2] >= 1.5
    print(json.dumps({
        "cycles": len(r), "P[A(p) > A(p/2)] all": float(np.mean(r[:, 0] > r[:, 1])),
        "mean A(p), A(p/2) all": [float(r[:, 0].mean()), float(r[:, 1].mean())],
        "cycles in over-counted videos (ratio>=1.5)": int(over.sum()),
        "P[A(p) > A(p/2)] over-counted": float(np.mean(r[over, 0] > r[over, 1])) if over.any() else None,
        "mean A(p), A(p/2) over-counted": [float(r[over, 0].mean()), float(r[over, 1].mean())] if over.any() else None,
    }, indent=1))


if __name__ == "__main__":
    main()
