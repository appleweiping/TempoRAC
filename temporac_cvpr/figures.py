"""Paper figures (run on the server; outputs vector PDFs).

observation : (a) per-video median cycle length and within-video variation on
              RepCount-A train; (b) MAE of single-rate counters per tempo tertile
              (mean over seeds of the given run groups, given split).
Usage:
  python -m temporac_cvpr.figures observation --manifest M --features F --split test \
      --runs single_r2=DIR_GLOB single_r4=... single_r8=... --out fig/observation.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from temporac_cvpr.analyze import video_stats

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def per_video_mae(pred_files: list[str], split_key: str, stats: dict) -> dict:
    vals = {k: [] for k in stats}
    for f in pred_files:
        p = json.loads(Path(f).read_text())
        p = p.get(split_key) or p.get("predictions")
        for k in stats:
            g = stats[k]["count"]
            vals[k].append(abs(round(p[k]) - g) / (g + 0.1))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def observation(args) -> None:
    man = json.loads(Path(args.manifest).read_text())
    fps = {f.stem: json.loads(f.read_text()).get("fps") for f in Path(args.features, "meta").glob("train__*.json")}
    tr = video_stats(man, "train", fps)
    per = np.array([s["period_s"] for s in tr.values()])
    cv = np.array([s["cv"] for s in tr.values()])
    fps_e = {f.stem: json.loads(f.read_text()).get("fps") for f in Path(args.features, "meta").glob(f"{args.split}__*.json")}
    ev = video_stats(man, args.split, fps_e)
    keys = sorted(ev)
    q = np.quantile([ev[k]["period_s"] for k in keys], [1 / 3, 2 / 3])
    bins = [("fast", lambda s: s["period_s"] < q[0]), ("medium", lambda s: q[0] <= s["period_s"] < q[1]),
            ("slow", lambda s: s["period_s"] >= q[1])]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax[0].scatter(per, cv, s=4, alpha=0.45, color=COLORS[0], linewidths=0)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("median cycle length per video (s)")
    ax[0].set_ylabel("within-video CV of cycle length")
    ax[0].set_title("(a) tempo across and within videos", fontsize=8)
    names = [r.split("=")[0] for r in args.runs]
    width = 0.8 / len(names)
    for j, spec in enumerate(args.runs):
        name, pat = spec.split("=")
        files = sorted(glob.glob(pat))
        m = per_video_mae(files, "test" if args.split == "test" else "val", ev)
        ys = [np.mean([m[k] for k in keys if fn(ev[k])]) for _, fn in bins]
        ax[1].bar(np.arange(3) + (j - (len(names) - 1) / 2) * width, ys, width, color=COLORS[j], label=f"{name} ({len(files)} seeds)")
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels([f"{b} (<{q[0]:.1f}s)" if i == 0 else (f"{b}" if i == 1 else f"{b} (>{q[1]:.1f}s)")
                           for i, (b, _) in enumerate(bins)])
    ax[1].set_ylabel("MAE")
    ax[1].set_title("(b) best sampling rate depends on tempo", fontsize=8)
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("--manifest")
    ap.add_argument("--features")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    {"observation": observation}[args.kind](args)


if __name__ == "__main__":
    main()
