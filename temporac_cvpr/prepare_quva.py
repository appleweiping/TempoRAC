"""QUVA Repetition (Runia et al., CVPR 2018) manifest: 100 videos, test-only.

Annotations are the frame indices of the end of every cycle; the count is their
number (as stated in the dataset README).  Cycle bounds for analysis use the
previous end as start (the first start extrapolated by one cycle length).  The
dataset is used for zero-shot evaluation only (no training, no selection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../QUVARepetitionDataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    items = []
    for f in sorted((root / "videos").glob("*.mp4")):
        ends = np.load(root / "annotations" / f"{f.stem}.npy").astype(float).reshape(-1)
        first = max(0.0, ends[0] - (ends[1] - ends[0])) if len(ends) > 1 else 0.0
        starts = np.concatenate([[first], ends[:-1]])
        bounds = np.stack([starts, ends], 1).reshape(-1).tolist()
        data = f.read_bytes()
        items.append({"dataset": "QUVA", "split": "test", "video_id": f.stem, "file": str(f), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(), "count": float(len(ends)), "bounds": bounds,
                      "action": f.stem.split("_", 1)[1]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(items, indent=0))
    print("QUVA", len(items), "counts", min(i["count"] for i in items), max(i["count"] for i in items))


if __name__ == "__main__":
    main()
