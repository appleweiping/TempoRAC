"""Validation-only feasibility check of a *tempo prior* for view selection.

Instead of learning "which view won" from the decoder's held-out predictions (few, noisy labels), predict
the cycle duration of a video from what is repeated, supervised by the annotated cycle durations of the
training videos (decoder-independent: no stacking, no train/val decoder shift), and weight each view by
how well the cycle duration it implies, P_k = duration / c_k, agrees with the prediction mu:
    w_k ~ exp(-(log P_k - mu)^2 / (2 sigma^2)).
Stand-ins for the language model's prediction here: the median log cycle duration of the k nearest training
videos in the description-embedding space, and (reference, uses labels) of the training videos of the class.
Usage: python -m temporac_cvpr.tempo_prior_check OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from temporac_cvpr.tempo_select import embeddings, mae

R = Path("/root/autodl-tmp/TempoRAC")


def main() -> None:
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "1x" in r["c"] and "2x" in r["c"]]
    man = {f"{it['split']}__{it['video_id']}": it for it in json.loads((R / "data/UNION/manifest_dev.json").read_text())}

    def cls(r):
        it = man[r["key"]]
        if it.get("action"):
            return f"{r['ds']}:{str(it['action']).lower()}"
        km = it.get("kinetics_member") or ""
        return f"{r['ds']}:{km.split('/')[1] if km.count('/') >= 2 else '?'}"

    tr = [r for r in rows if r["role"] == "oof"]  # one row per training video (annotated count + duration)
    va = [r for r in rows if r["role"] == "val"]
    ltr = np.log(np.array([r["dur"] / r["gt"] for r in tr]))
    Etr, Eva = embeddings(tr, "desc_h"), embeddings(va, "desc_h")
    S = Eva @ Etr.T
    order = np.argsort(-S, axis=1)
    lva = np.log(np.array([r["dur"] / r["gt"] for r in va]))
    by_cls = {}
    for r, l in zip(tr, ltr):
        by_cls.setdefault(cls(r), []).append(l)
    mu_cls = np.array([np.median(by_cls[cls(r)]) if cls(r) in by_cls else np.median(ltr) for r in va])
    g = np.array([r["gt"] for r in va])
    c1 = np.array([r["c"]["1x"] for r in va]); c2 = np.array([r["c"]["2x"] for r in va])
    d = np.array([r["dur"] for r in va])
    l1, l2 = np.log(d / np.maximum(c1, .5)), np.log(d / np.maximum(c2, .5))
    ds = np.array([r["ds"] for r in va])
    seeds = np.array([r["seed"] for r in va])

    def score(w):
        res = []
        for name in ("RepCountA", "Countix", "UCFRep"):
            m = ds == name
            a = np.mean([mae((1 - w[m & (seeds == s)]) * c1[m & (seeds == s)] + w[m & (seeds == s)] * c2[m & (seeds == s)], g[m & (seeds == s)])
                         for s in range(5)])
            b = np.mean([mae((c1[m & (seeds == s)] + c2[m & (seeds == s)]) / 2, g[m & (seeds == s)]) for s in range(5)])
            res.append((name, a, a / b - 1))
        return res

    def weights(mu, sig):
        a = -(l1 - mu) ** 2 / (2 * sig ** 2)
        b = -(l2 - mu) ** 2 / (2 * sig ** 2)
        return 1 / (1 + np.exp(a - b))  # weight of the 2x view

    cands = {"class median (labels; reference)": mu_cls}
    for k in (8, 16, 32, 64):
        cands[f"description kNN k={k}"] = np.median(ltr[order[:, :k]], axis=1)
    for name, mu in cands.items():
        err = np.median(np.abs(mu - lva))
        print(f"== {name}: median |log error| of the cycle duration {err:.3f} (x{np.exp(err):.2f})")
        for sig in (0.2, 0.35, 0.5, 0.75, 1.0):
            r = score(weights(mu, sig))
            print(f"   sigma {sig:<4} " + "  ".join(f"{n} {a:.4f} ({x:+.3f})" for n, a, x in r) + f"  mean {np.mean([x for _, _, x in r]):+.4f}")


if __name__ == "__main__":
    main()
