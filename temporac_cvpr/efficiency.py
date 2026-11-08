"""Head efficiency: parameters, FLOPs and latency on a fixed-length input.

FLOPs are counted with torch.utils.flop_counter for one video of T frames
(default 900 = RepCount-A mean length) at batch size 1; latency is the median of
repeated CUDA-synchronised forward passes in bf16.  Encoder (feature extraction)
cost is reported separately from the token counts per rate.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from temporac_cvpr.models import build_model
from temporac_cvpr.train import DEFAULT

VIT_B_GFLOPS_PER_CLIP = 180.0  # ST-MAE ViT-B/16, 16x224x224 clip (approximate, forward)


def dummy_batch(cfg: dict, T: int, device) -> dict:
    G = math.ceil(T / 2)
    P = cfg["pool_to"] ** 2
    feats, masks = {}, {}
    for s in cfg["rates"]:
        L = math.ceil(G / s)
        feats[s] = torch.randn(1, L, P, 768, device=device, dtype=torch.float16)
        masks[s] = torch.ones(1, L, dtype=torch.bool, device=device)
    return {"feats": feats, "masks": masks, "mask": torch.ones(1, G, dtype=torch.bool, device=device)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True, help="JSON list of {name, cfg}")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda")
    rows = []
    for item in json.loads(args.cfgs):
        cfg = dict(DEFAULT)
        cfg.update(item["cfg"])
        model = build_model(cfg).to(device).eval()
        params = sum(p.numel() for p in model.parameters())
        batch = dummy_batch(cfg, args.frames, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fc = FlopCounterMode(display=False)
            with fc:
                model(batch)
            flops = fc.get_total_flops()
            for _ in range(5):
                model(batch)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        clips = sum(math.ceil(args.frames / (4 * s)) for s in cfg["rates"])
        rows.append({"name": item["name"], "params_M": params / 1e6, "head_GFLOPs": flops / 1e9,
                     "latency_ms": 1000 * sorted(ts)[len(ts) // 2], "encoder_clips": clips,
                     "encoder_GFLOPs": clips * VIT_B_GFLOPS_PER_CLIP})
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
