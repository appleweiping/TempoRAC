"""Validation-only check of a tempo-normalising view choice driven by the model's own count.

The decoder is reliable when a repetition spans a medium number of frames in the view it reads
(see view_tempo.py): the original view over-counts slow repetitions (half cycles), the exact 2x view
under-counts fast ones.  Like Context's context-aware scaling, we estimate the period from the
model's own original-view count, est = L / c_1x (L = length of the counted segment in frames), and
weight the 2x view by w = sigmoid((log2 est - log2 tau) / temp).  tau/temp are one global pair for
all datasets (no per-dataset choice).  No labels are used at inference.
Usage: python -m temporac_cvpr.view_gate MANIFEST FEAT_ROOT RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SPLITS = ["rc_valid", "cx_val", "ucf_dev"]


def seg_len(it: dict, feat_root: Path) -> float:
    fkey = f"{it.get('fsplit', it['split'])}__{it['video_id']}"
    meta = json.loads((feat_root / "meta" / f"{fkey}.json").read_text())
    T = int(meta["frames_decoded"])
    fr = it.get("frame_range")
    if fr:
        n_ann = it.get("num_frames_csv")
        scale = T / float(n_ann) if n_ann and abs(n_ann - T) > 1 else 1.0
        return max(float(fr[1] - fr[0] + 1) * scale, 1.0)
    return float(T)


def mae(p, g):
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1)))


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    feat_root = Path(sys.argv[2])
    runs = [Path(r) for r in sys.argv[3:]]
    items = {f"{it['split']}__{it['video_id']}": it for it in manifest if it["split"] in SPLITS}
    L = {k: seg_len(it, feat_root) for k, it in items.items()}
    data = {}
    for split in SPLITS:
        rows = []
        for r in runs:
            f = r / f"predictions_{split}.json"
            if not f.exists():
                continue
            P = json.loads(f.read_text())
            for k, p1 in P["val"].items():
                g = float(items[k]["count"])
                if g > 0 and k in P["val_shift"]:
                    rows.append((L[k], g, p1, P["val_shift"][k], hash(r)))
        data[split] = np.asarray(rows)
    print("baseline (1x / 2x / avg):", {s: tuple(round(mae(x, a[:, 1]), 4) for x in (a[:, 2], a[:, 3], (a[:, 2] + a[:, 3]) / 2))
                                        for s, a in data.items()})
    res = []
    for tau in (16, 24, 32, 40, 48, 64, 96):
        for temp in (0.01, 0.25, 0.5, 1.0):
            out = {}
            for s, a in data.items():
                est = a[:, 0] / np.maximum(a[:, 2], 0.5)
                w = 1 / (1 + np.exp(-(np.log2(est) - np.log2(tau)) / temp))
                out[s] = mae((1 - w) * a[:, 2] + w * a[:, 3], a[:, 1])
            base = {s: mae((a[:, 2] + a[:, 3]) / 2, a[:, 1]) for s, a in data.items()}
            rel = np.mean([out[s] / base[s] - 1 for s in out])
            res.append((rel, tau, temp, out))
    for rel, tau, temp, out in sorted(res, key=lambda x: x[0])[:12]:
        print(f"tau={tau:>3} temp={temp:<5} mean rel vs avg {rel:+.3f}  " + "  ".join(f"{s} {v:.4f}" for s, v in out.items()))


if __name__ == "__main__":
    main()
