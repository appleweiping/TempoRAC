"""Validation-only diagnostic: error of the original and the 2x-shifted view as a function of tempo.

For every validation video the annotated median cycle length (decoded frames) is computed from the
manifest bounds (Countix: span / count when no bounds).  Per tempo bin we report the protocol MAE of
the original view, the 2x view and their average, and the mean log2 ratio pred/gt (negative =
under-count).  If the 2x view is much worse than the original on fast repetitions and better on slow
ones, an exact 0.5x view (pyramid shifted one level down, needs an s=1 stream) should help the fast end.
Usage: python -m temporac_cvpr.view_tempo MANIFEST FEAT_ROOT RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SPLITS = ["rc_valid", "cx_val", "ucf_dev"]
BINS = [0, 12, 24, 48, 96, 1e9]


def period_frames(it: dict, feat_root: Path) -> float:
    fkey = f"{it.get('fsplit', it['split'])}__{it['video_id']}"
    meta = json.loads((feat_root / "meta" / f"{fkey}.json").read_text())
    T = int(meta["frames_decoded"])
    n_ann = it.get("num_frames_csv")
    scale = T / float(n_ann) if n_ann and abs(n_ann - T) > 1 else 1.0
    b = it.get("bounds") or []
    if len(b) >= 1:
        b = np.asarray(b, float).reshape(-1, 2) * scale
        return float(np.median(np.abs(b[:, 1] - b[:, 0])))
    span = it.get("span") or [0, T]
    return float(abs(span[1] - span[0]) * scale) / max(float(it["count"]), 1.0)


def mae(p, g):
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1)))


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    if isinstance(manifest, dict):
        manifest = manifest.get("videos", manifest.get("items"))
    feat_root = Path(sys.argv[2])
    runs = [Path(r) for r in sys.argv[3:]]
    items = {f"{it['split']}__{it['video_id']}": it for it in manifest if it["split"] in SPLITS}
    per = {k: period_frames(it, feat_root) for k, it in items.items()}
    for split in SPLITS:
        rows = []  # (period, gt, p1, p2)
        for r in runs:
            f = r / f"predictions_{split}.json"
            if not f.exists():
                continue
            P = json.loads(f.read_text())
            v1, v2 = P.get("val", {}), P.get("val_shift", {})
            for k, p1 in v1.items():
                g = float(items[k]["count"]) if k in items else float("nan")
                if not g > 0 or k not in v2:
                    continue
                rows.append((per[k], g, p1, v2[k]))
        if not rows:
            continue
        a = np.asarray(rows)
        print(f"== {split}  n={len(a)} (video x seed)  keys={sorted(P.keys())}")
        print(f"{'period(fr)':>12} {'n':>5} {'mae_1x':>7} {'mae_2x':>7} {'mae_avg':>7} {'log2r_1x':>8} {'log2r_2x':>8}")
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if not m.any():
                continue
            g, p1, p2 = a[m, 1], a[m, 2], a[m, 3]
            print(f"{lo:>5.0f}-{hi if hi < 1e8 else 'inf':>5} {m.sum():>5} {mae(p1, g):7.3f} {mae(p2, g):7.3f} "
                  f"{mae((p1 + p2) / 2, g):7.3f} {np.mean(np.log2(np.maximum(p1, .5) / g)):8.3f} "
                  f"{np.mean(np.log2(np.maximum(p2, .5) / g)):8.3f}")
        g, p1, p2 = a[:, 1], a[:, 2], a[:, 3]
        print(f"{'all':>11} {len(a):>5} {mae(p1, g):7.3f} {mae(p2, g):7.3f} {mae((p1 + p2) / 2, g):7.3f}")


if __name__ == "__main__":
    main()
