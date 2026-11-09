"""Offline check of the period-gated 2x-tempo test view (validation predictions only).

The shifted pyramid is only a valid view when the accelerated cycle stays >= 8 frames,
the same aliasing guard as in training.  Without labels, the cycle length is estimated
from the original view as T / predicted count; the two views are averaged only when
that estimate is >= thr frames (thr = 2 x 8 = 16 is the training guard).
Usage: python -m temporac_cvpr.tta_gate RUN_GLOB [RUN_GLOB ...]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC")
STEM = {"RepCountA": "RepCountA", "Countix": "Countix", "UCFRep": "UCFRep"}


def manifest_index() -> dict:
    idx = {}
    for m in (R / "data/UNION/manifest_dev.json", R / "data/RepCountA/manifest_all.json",
              R / "data/UCFRep/manifest_dev.json", R / "data/Countix/manifest.json"):
        for it in json.loads(m.read_text()):
            idx.setdefault(f"{it['split']}__{it['video_id']}", it)
    return idx


def frames(it: dict, cache: dict) -> int:
    k = (it["dataset"], it["video_id"])
    if k not in cache:
        fs = it.get("fsplit") or it["split"]
        cache[k] = json.loads((R / "features" / f"{STEM[it['dataset']]}_stmae_p4" / "meta" /
                               f"{fs}__{it['video_id']}.json").read_text())["frames_decoded"]
    return cache[k]


def score(P: dict, keys: list, man: dict) -> tuple[float, float]:
    g = np.array([man[k]["count"] for k in keys])
    p = np.array([P[k] for k in keys])
    m = g > 0
    d = np.abs(np.round(p) - g)
    return float((d / (g + 0.1))[m].mean()), float((d <= 1)[m].mean())


def main() -> None:
    man, cache = manifest_index(), {}
    for pat in sys.argv[1:]:
        res = defaultdict(list)
        for d in sorted(glob.glob(str(R / "runs" / pat))):
            for f in sorted(Path(d).glob("predictions*.json")):
                P = json.loads(f.read_text())
                if "val_shift" not in P:
                    continue
                o, s = P["val"], P["val_shift"]
                keys = list(o)
                res["orig"].append(score(o, keys, man))
                res["avg"].append(score({k: 0.5 * (o[k] + s[k]) for k in keys}, keys, man))
                for thr in (16, 24, 32):
                    gate = {k: (0.5 * (o[k] + s[k]) if frames(man[k], cache) / max(o[k], 0.5) >= thr else o[k])
                            for k in keys}
                    res[f"gate{thr}"].append(score(gate, keys, man))
        if res:
            print(pat.ljust(26), len(res["orig"]),
                  "  ".join(f"{k}={np.mean([x[0] for x in v]):.4f}/{np.mean([x[1] for x in v]):.2f}"
                            for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
