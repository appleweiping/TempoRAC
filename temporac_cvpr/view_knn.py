"""Validation-only feasibility check of a label-free semantic tempo prior.

Same leave-one-out estimator as view_class.py, but the "class" is replaced by the k nearest
validation videos in the space of the MLLM's own description of the repeated action (BGE embedding of
desc_embed.py; no labels at inference).  w(2x view) = fraction of neighbours on which the 2x view had
the lower error.
Usage: python -m temporac_cvpr.view_knn MANIFEST DESC_ROOT[,DESC_ROOT...] RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SPLITS = ["rc_valid", "cx_val", "ucf_dev"]


def mae(p, g):
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1)))


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    roots = [Path(r) for r in sys.argv[2].split(",")]
    runs = [Path(r) for r in sys.argv[3:]]
    items = {f"{it['split']}__{it['video_id']}": it for it in manifest if it["split"] in SPLITS}

    def emb(k: str, field: str) -> np.ndarray:
        it = items[k]
        fkey = f"{it.get('fsplit', it['split'])}__{it['video_id']}"
        for r in roots:
            f = r / f"{fkey}.npz"
            if f.exists():
                return np.load(f)[field].astype(np.float32)
        raise FileNotFoundError(fkey)

    for field in ("desc_h", "act_h", "rep_h"):
        for split in SPLITS:
            out = {k: [] for k in ("avg", "k5", "k10", "k20", "k40")}
            for r in runs:
                f = r / f"predictions_{split}.json"
                if not f.exists():
                    continue
                P = json.loads(f.read_text())
                keys = [k for k in P["val"] if k in P["val_shift"] and float(items[k]["count"]) > 0]
                g = np.array([float(items[k]["count"]) for k in keys])
                p1 = np.array([P["val"][k] for k in keys])
                p2 = np.array([P["val_shift"][k] for k in keys])
                e1 = np.abs(np.round(p1) - g) / (g + .1)
                e2 = np.abs(np.round(p2) - g) / (g + .1)
                E = np.stack([emb(k, field) for k in keys])
                E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
                S = E @ E.T
                np.fill_diagonal(S, -np.inf)
                order = np.argsort(-S, axis=1)
                lab = np.where(e1 == e2, np.nan, (e2 < e1).astype(float))
                out["avg"].append(mae((p1 + p2) / 2, g))
                for kk in (5, 10, 20, 40):
                    w = np.array([np.nanmean(lab[order[i, :kk]]) if np.isfinite(lab[order[i, :kk]]).any() else 0.5
                                  for i in range(len(keys))])
                    out[f"k{kk}"].append(mae((1 - w) * p1 + w * p2, g))
            print(f"{field:7s} {split:9s} " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in out.items()))


if __name__ == "__main__":
    main()
