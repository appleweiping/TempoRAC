"""Multi-rate frozen ST-MAE token extraction on a fixed raw-frame grid.

For every sampling rate ``s`` (frame stride inside a clip) we slide 16-frame clips
spanning ``16 s`` raw frames with a hop of ``4 s`` raw frames (75% overlap, the
ESCounts setting for s = 4).  A clip starting at raw frame ``j`` yields 8 temporal
tokens; token ``k`` covers raw frames ``[j + 2 s k, j + 2 s (k + 1))`` and is written
to grid cell ``j / (2 s) + k`` of a rate-specific grid with cell width ``2 s`` raw
frames.  Overlapping clips are averaged.  Spatial 14x14 tokens are average-pooled
to ``P x P`` (default 7x7).  Output per video and rate: float16 ``[G_s, P*P, 768]``.

Pre-processing follows ESCounts: shorter side 224, centre crop 224, ImageNet
mean/std.  Videos are decoded completely; no frame cap or sample cap is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from temporac_cvpr.vit_encoder import load_stmae_vitb

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def decode_video(path: str, size: int = 224) -> tuple[np.ndarray, dict]:
    import decord

    vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=2)
    h0, w0 = vr[0].shape[:2]
    n0 = len(vr)
    fps = float(vr.get_avg_fps())
    scale = size / min(h0, w0)
    nh, nw = max(size, round(h0 * scale)), max(size, round(w0 * scale))
    del vr
    vr = decord.VideoReader(path, ctx=decord.cpu(0), width=nw, height=nh, num_threads=2)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # T H W 3 uint8
    top, left = (nh - size) // 2, (nw - size) // 2
    frames = np.ascontiguousarray(frames[:, top : top + size, left : left + size])
    info = {"orig_h": h0, "orig_w": w0, "frames_decoded": int(frames.shape[0]), "len_reader": n0, "fps": fps,
            "resized_h": nh, "resized_w": nw}
    return frames, info


class VideoSet(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            frames, info = decode_video(it["file"])
            if "frame_range" in it:  # trimmed protocol (e.g. UCFRep annotated segment)
                a, b = it["frame_range"]
                info["frame_range_requested"] = [a, b]
                info["frame_range_clamped"] = b > frames.shape[0] - 1
                b = min(b, frames.shape[0] - 1)
                frames = np.ascontiguousarray(frames[a : b + 1])
                info["frames_full_video"] = info["frames_decoded"]
                info["frames_decoded"] = int(frames.shape[0])
            if "frame_index_file" in it:  # count-preserving tempo warp (stress test)
                idx = np.load(it["frame_index_file"])
                assert idx.max() < frames.shape[0], (it["video_id"], idx.max(), frames.shape[0])
                info["frames_source"] = int(frames.shape[0])
                frames = np.ascontiguousarray(frames[idx])
                info["frames_decoded"] = int(frames.shape[0])
            return it, torch.from_numpy(frames), info, None
        except Exception as e:  # recorded, never silently dropped
            return it, None, None, repr(e)


@torch.no_grad()
def encode_rate(model, frames_u8: torch.Tensor, s: int, pool: int, device, batch: int) -> np.ndarray:
    T = frames_u8.shape[0]
    starts = list(range(0, T, 4 * s))
    G = math.ceil(T / (2 * s))
    acc = torch.zeros(G, pool * pool, 768, dtype=torch.float32, device=device)
    cnt = torch.zeros(G, dtype=torch.float32, device=device)
    for b0 in range(0, len(starts), batch):
        bs = starts[b0 : b0 + batch]
        idx = torch.tensor([[min(j + s * f, T - 1) for f in range(16)] for j in bs], device=frames_u8.device)
        clip = frames_u8[idx.flatten()]  # (B*16) H W 3, already on device
        clip = clip.view(len(bs), 16, 224, 224, 3).permute(0, 4, 1, 2, 3).float().div_(255.0)
        clip = (clip - MEAN.to(device)) / STD.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = model(clip)  # B 8 14 14 C
        tok = tok.float().permute(0, 1, 4, 2, 3).reshape(len(bs) * 8, 768, 14, 14)
        tok = F.adaptive_avg_pool2d(tok, pool).reshape(len(bs), 8, 768, pool * pool).permute(0, 1, 3, 2)
        for bi, j in enumerate(bs):
            g0 = j // (2 * s)
            g1 = min(g0 + 8, G)
            acc[g0:g1] += tok[bi, : g1 - g0]
            cnt[g0:g1] += 1
    assert torch.all(cnt > 0)
    return (acc / cnt.view(-1, 1, 1)).to(torch.float16).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rates", default="1,2,4")
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rates = [int(r) for r in args.rates.split(",")]
    items = json.loads(Path(args.manifest).read_text())
    items = [it for k, it in enumerate(items) if k % args.num_shards == args.shard]
    out = Path(args.out)
    for s in rates:
        (out / f"r{s}").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    fkey = lambda it: f"{it.get('fsplit', it['split'])}__{it['video_id']}"
    # a video is to-do if its meta or any requested rate file is missing (adding a rate later is supported)
    todo = [it for it in items if not (out / "meta" / f"{fkey(it)}.json").exists()
            or any(not (out / f"r{s}" / f"{fkey(it)}.npy").exists() for s in rates)]
    print(f"shard {args.shard}: {len(items)} items, {len(todo)} to do", flush=True)
    device = torch.device("cuda")
    model, rep = load_stmae_vitb(args.ckpt)
    print("encoder", rep, flush=True)
    model = model.to(device)
    dl = torch.utils.data.DataLoader(VideoSet(todo), batch_size=None, num_workers=args.workers,
                                     prefetch_factor=2, persistent_workers=False)
    t0 = time.time()
    failures = []
    for n, (it, frames, info, err) in enumerate(dl):
        key = fkey(it)
        if err is not None:
            failures.append({"key": key, "error": err})
            print("DECODE_FAIL", key, err, flush=True)
            continue
        mpath = out / "meta" / f"{key}.json"
        old = json.loads(mpath.read_text()) if mpath.exists() else None
        meta = dict(info)
        meta.update({"video_id": it["video_id"], "split": it["split"], "dataset": it["dataset"],
                     "rates": dict(old["rates"]) if old else {}})
        if old and old.get("frames_decoded") != info["frames_decoded"]:
            raise RuntimeError(f"decoded length changed for {key}: {old.get('frames_decoded')} vs {info['frames_decoded']}")
        frames = frames.to(device, non_blocking=True)
        for s in rates:
            if (out / f"r{s}" / f"{key}.npy").exists():
                continue
            arr = encode_rate(model, frames, s, args.pool, device, args.batch)
            np.save(out / f"r{s}" / f"{key}.npy", arr)
            meta["rates"][str(s)] = {"cells": int(arr.shape[0]), "cell_frames": 2 * s}
        (out / "meta" / f"{key}.json").write_text(json.dumps(meta))
        if n % 20 == 0:
            el = time.time() - t0
            print(f"[{args.shard}] {n + 1}/{len(todo)} {key} T={info['frames_decoded']} {el / 60:.1f}min", flush=True)
    (out / f"failures_shard{args.shard}.json").write_text(json.dumps(failures, indent=1))
    print("EXTRACT_DONE", args.shard, "failures", len(failures), flush=True)


if __name__ == "__main__":
    main()
