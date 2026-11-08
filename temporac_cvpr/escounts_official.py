"""Official ESCounts RepCount checkpoint, exemplar-free (shot_num=0), on our data.

Token extraction mirrors ESCounts ``demo.py`` / ``save_swim_features.py``: ImageNet
normalisation, 64 zero frames padded after normalisation, windows starting every
16 frames, 16 frames sampled from each 64-frame window, fp16 autocast, 14x14 tokens
kept.  The encoder is our verified re-implementation of the same ST-MAE ViT-B
(block-level equality with pySlowFast, see check_encoder.py); ESCounts' demo
mapping loads the identical weights.  Decoding uses the official test-time
"two-offset" procedure (windows 0::4 and 2::4, the latter shifted by 4 tokens,
predictions averaged), adaptive pooling to ceil(14*0.4)=6, density / 100.

Writes per-video raw counts so that metrics can be recomputed with any convention.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.extract_features import MEAN, STD, decode_video
from temporac_cvpr.metrics import count_metrics
from temporac_cvpr.vit_encoder import load_stmae_vitb


@torch.no_grad()
def escounts_tokens(encoder, frames_u8: torch.Tensor, device, batch: int = 32) -> torch.Tensor:
    T = frames_u8.shape[0]
    x = frames_u8.to(device).permute(3, 0, 1, 2).float().div_(255.0)  # C T H W
    x = (x - MEAN.view(3, 1, 1, 1).to(device)) / STD.view(3, 1, 1, 1).to(device)
    x = torch.cat([x, torch.zeros(3, 64, 224, 224, device=device)], 1)
    starts = list(range(0, T, 16))
    out = []
    for b0 in range(0, len(starts), batch):
        clips = []
        for j in starts[b0:b0 + batch]:
            idx = np.linspace(j, j + 64, 17)[:16].astype(int)
            clips.append(x[:, idx])
        data = torch.stack(clips)
        with torch.autocast("cuda", dtype=torch.float16):
            tok = encoder(data)  # B 8 14 14 C
        out.append(tok.permute(0, 4, 1, 2, 3).half().cpu())  # B C T H W
    return torch.cat(out)


def two_offset(tokens: torch.Tensor, n_frames: int):
    low, up = 0, math.ceil(n_frames / 8)  # RepCount: segment_start=0, segment_end=num_frames
    wins = []
    for offset, a, b in ((0, low, up), (2, max(low - 4, 0), max(up - 4, 0))):
        sel = tokens[offset::4]
        s, c, t, h, w = sel.shape
        arr = sel.permute(1, 0, 2, 3, 4).reshape(c, s * t, h, w)[:, a:b].float()
        if arr.shape[1] == 0:
            raise ValueError("empty offset window")
        factor = math.ceil(w * 0.4)
        arr = torch.nn.functional.adaptive_avg_pool3d(arr, (arr.shape[1], factor, factor))
        wins.append(arr)
    return wins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="valid,test")
    ap.add_argument("--source", required=True, help="official EveryShotCounts checkout")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torchvision.transforms._functional_tensor as _ft  # torchvision>=0.17 renamed module
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
    # slowfast.models.head_helper imports detectron2.layers.ROIAlign at import time only
    # (detection heads, unused by ESCounts); provide the torchvision op as a stand-in.
    import torchvision.ops

    d2 = types.ModuleType("detectron2")
    d2l = types.ModuleType("detectron2.layers")
    d2l.ROIAlign = torchvision.ops.RoIAlign
    d2.layers = d2l
    sys.modules["detectron2"], sys.modules["detectron2.layers"] = d2, d2l
    sys.path.insert(0, args.source)
    import video_mae_cross_full_attention as vm
    from slowfast.utils.parser import load_config

    cfg = load_config(argparse.Namespace(opts=None, output_dir=str(out)),
                      path_to_config=str(Path(args.source) / "configs/pretrain_config.yaml"))
    decoder = vm.SupervisedMAE(cfg=cfg, use_precomputed=True, token_pool_ratio=0.4, iterative_shots=True,
                               encodings="mae", window_size=(4, 7, 7)).to(device)
    sd = torch.load(args.decoder, map_location="cpu", weights_only=True)["model_state_dict"]
    decoder.load_state_dict(sd, strict=True)
    decoder.eval()
    encoder, _ = load_stmae_vitb(args.encoder)
    encoder = encoder.to(device)

    items = [it for it in json.loads(Path(args.manifest).read_text()) if it["split"] in set(args.splits.split(","))]
    rec_path = out / "records.jsonl"
    done = {}
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    f = open(rec_path, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        key = f"{it['split']}__{it['video_id']}"
        if key in done:
            continue
        frames, info = decode_video(it["file"])
        tok = escounts_tokens(encoder, torch.from_numpy(frames), device)
        w1, w2 = two_offset(tok, frames.shape[0])
        thw = [list(w1.shape[-3:])]
        ph = torch.zeros((1, 1, 768), device=device)
        with torch.no_grad():
            x1 = w1.permute(1, 2, 3, 0).reshape(1, -1, w1.shape[0]).to(device)
            x2 = w2.permute(1, 2, 3, 0).reshape(1, -1, w2.shape[0]).to(device)
            p1 = decoder(x1, ph, thw, shot_num=0)
            p2 = decoder(x2, ph, thw, shot_num=0)
            if p1.shape != p2.shape:
                p2 = torch.cat([torch.zeros(1, 4, dtype=p2.dtype, device=device), p2], 1)
            pred = (p1 + p2) / 2
            raw = float(pred.float().sum() / 100.0)
        r = {"key": key, "split": it["split"], "video_id": it["video_id"], "count": it["count"], "raw_count": raw,
             "frames": int(frames.shape[0]), "windows": int(tok.shape[0])}
        f.write(json.dumps(r) + "\n")
        f.flush()
        done[key] = r
        if n % 20 == 0:
            print(f"{n + 1}/{len(items)} {key} pred={raw:.2f} gt={it['count']} {(time.time() - t0) / 60:.1f}min", flush=True)
    res = {}
    for s in args.splits.split(","):
        rs = [r for r in done.values() if r["split"] == s]
        res[s] = count_metrics([r["raw_count"] for r in rs], [r["count"] for r in rs])
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
