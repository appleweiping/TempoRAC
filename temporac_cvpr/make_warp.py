"""Count-preserving piecewise tempo warps for the non-stationary-tempo stress test.

Each source video is split at two random interior points into three segments that
are replayed at different speeds v_k (log-uniform in [1/vmax, vmax], with at least
one segment faster than ``fast_min`` and one slower than 1/``fast_min``).  Output
frame j of segment k shows source frame floor(s_k + j * v_k).  Cycle boundaries are
mapped through the inverse time map; the repetition count is unchanged.  The warp
is fixed by a seed and applied identically to every evaluated method.

Writes ``warp_<name>.json`` (manifest; split ``<split>_warp``) and one ``.npy`` frame
index array per video.  Analysis only: never used for training or selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def make_profile(T: int, rng, vmax: float, fast_min: float):
    while True:
        u = np.sort(rng.uniform(0.2, 0.8, size=2))
        if u[1] - u[0] >= 0.15:
            break
    cuts = [0, int(u[0] * T), int(u[1] * T), T]
    while True:
        v = np.exp(rng.uniform(-math.log(vmax), math.log(vmax), size=3))
        if v.max() >= fast_min and v.min() <= 1.0 / fast_min:
            break
    return cuts, v


def warp_indices(T: int, cuts, v):
    idx, seg_out = [], []
    o = 0
    for k in range(3):
        s, e = cuts[k], cuts[k + 1]
        n = max(1, int(round((e - s) / v[k])))
        j = np.arange(n)
        src = np.minimum(np.floor(s + j * v[k]).astype(int), e - 1)
        idx.append(src)
        seg_out.append((s, e, o, v[k]))
        o += n
    return np.concatenate(idx), seg_out


def map_frame(b: float, seg_out) -> float:
    for s, e, o, vk in seg_out:
        if b < e or (s, e, o, vk) == seg_out[-1]:
            return o + (b - s) / vk
    raise AssertionError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="pw3")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--vmax", type=float, default=2.0)
    ap.add_argument("--fast-min", type=float, default=1.4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / args.name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    items = []
    for it in json.loads(Path(args.manifest).read_text()):
        if it["split"] != args.split:
            continue
        T = int(it.get("num_frames_csv") or 0)
        if T <= 0:
            raise ValueError(f"num_frames missing for {it['video_id']}")
        cuts, v = make_profile(T, rng, args.vmax, args.fast_min)
        idx, seg_out = warp_indices(T, cuts, v)
        p = out / args.name / f"{it['video_id']}.npy"
        np.save(p, idx.astype(np.int32))
        new = dict(it)
        new["split"] = f"{args.split}_warp"
        new["fsplit"] = f"{args.split}_{args.name}"
        new["frame_index_file"] = str(p)
        new["bounds"] = [float(map_frame(b, seg_out)) for b in it.get("bounds", [])]
        new["num_frames_csv"] = float(len(idx))
        new["warp"] = {"cuts": cuts, "speeds": [float(x) for x in v], "seed": args.seed, "out_frames": int(len(idx))}
        items.append(new)
    (out / f"warp_{args.name}.json").write_text(json.dumps(items, indent=1))
    sp = np.array([x["warp"]["speeds"] for x in items])
    print("videos", len(items), "speed range", sp.min().round(3), sp.max().round(3),
          "mean |log v|", np.abs(np.log(sp)).mean().round(3))


if __name__ == "__main__":
    main()
