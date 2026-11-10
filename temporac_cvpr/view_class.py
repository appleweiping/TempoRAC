"""Validation-only feasibility check: does the action (semantics) predict which view is right?

For every validation video, the weight of the 2x view is set from the OTHER validation videos of the
same action class (leave-one-out): w = fraction of those videos on which the 2x view had the lower
error (ties -> 0.5; classes with no other video -> 0.5, i.e. the plain average).  This is an honest
estimate of a gate that knows only the action, learned from other videos; if it beats the plain
average on all three datasets, a semantic tempo prior (from the MLLM description) is worth building,
trained on out-of-fold training predictions.
Usage: python -m temporac_cvpr.view_class MANIFEST RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ["rc_valid", "cx_val", "ucf_dev"]


def mae(p, g):
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1)))


def action(it: dict) -> str:
    if it.get("action"):
        return str(it["action"]).lower()
    km = it.get("kinetics_member") or ""
    return km.split("/")[1] if km.count("/") >= 2 else "?"


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    items = {f"{it['split']}__{it['video_id']}": it for it in manifest if it["split"] in SPLITS}
    for split in SPLITS:
        res = {"avg": [], "1x": [], "2x": [], "loo": [], "hard": []}
        n_cls = None
        for r in sys.argv[2:]:
            f = Path(r) / f"predictions_{split}.json"
            if not f.exists():
                continue
            P = json.loads(f.read_text())
            keys = [k for k in P["val"] if k in P["val_shift"] and float(items[k]["count"]) > 0]
            g = np.array([float(items[k]["count"]) for k in keys])
            p1 = np.array([P["val"][k] for k in keys])
            p2 = np.array([P["val_shift"][k] for k in keys])
            e1 = np.abs(np.round(p1) - g) / (g + .1)
            e2 = np.abs(np.round(p2) - g) / (g + .1)
            cls = [action(items[k]) for k in keys]
            by = defaultdict(list)
            for i, c in enumerate(cls):
                by[c].append(i)
            n_cls = len(by)
            w = np.full(len(keys), 0.5)
            for c, idx in by.items():
                for i in idx:
                    o = [j for j in idx if j != i and e1[j] != e2[j]]
                    if o:
                        w[i] = float(np.mean([e2[j] < e1[j] for j in o]))
            res["avg"].append(mae((p1 + p2) / 2, g))
            res["1x"].append(mae(p1, g))
            res["2x"].append(mae(p2, g))
            res["loo"].append(mae((1 - w) * p1 + w * p2, g))
            wh = np.where(w > 0.5, 1.0, np.where(w < 0.5, 0.0, 0.5))
            res["hard"].append(mae((1 - wh) * p1 + wh * p2, g))
        print(f"{split:9s} classes {n_cls}: " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
