"""Build the Countix manifest (ESCounts train/val/test split, Kinetics-700 clips).

Each row is a Kinetics clip trimmed to its annotated repetition segment
[rep_start, rep_end] (seconds, relative to the clip start, as encoded in the
ESCounts file names).  Countix provides counts but no per-cycle boundaries; like
ESCounts we place cycles uniformly over the segment for the density target, while
evaluation uses the annotated count only.  Rows whose clip could not be matched
unambiguously or downloaded are listed in the report and excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data/Countix")
    args = ap.parse_args()
    import decord

    root = Path(args.root)
    matched = json.loads((root / "countix_matched.json").read_text())
    rec = json.loads((root / "download_receipt.json").read_text())["files"]
    items, rep = [], {"missing_file": [], "short_segment": []}
    fps_cache = {}
    for m in matched:
        f = root / "videos" / Path(m["member"]).name
        if not f.exists() or f.stat().st_size != m["size"]:
            rep["missing_file"].append(m["name"])
            continue
        if str(f) not in fps_cache:
            vr = decord.VideoReader(str(f), num_threads=1)
            fps_cache[str(f)] = (float(vr.get_avg_fps()), len(vr))
        fps, n = fps_cache[str(f)]
        a, b = int(round(m["rep_start"] * fps)), min(int(round(m["rep_end"] * fps)), n - 1)
        if b - a + 1 < 8:
            rep["short_segment"].append(m["name"])
            continue
        N = b - a + 1
        c = int(round(m["count"]))
        edges = np.linspace(0, N - 1, c + 1)
        bounds = np.stack([edges[:-1], edges[1:]], 1).reshape(-1).tolist()
        items.append({"dataset": "Countix", "split": m["split"], "video_id": m["name"], "file": str(f),
                      "kinetics_member": m["member"], "sha256": rec.get(m["member"]), "count": float(m["count"]),
                      "frame_range": [a, b], "bounds": bounds, "fps_csv": fps, "num_frames_csv": float(N),
                      "num_frames_escounts": m["num_frames_csv"]})
    # The ESCounts CSVs contain a few exact duplicate rows (same clip and repetition span
    # in the same split); keep one copy so every evaluated video is counted once.
    seen, dedup = set(), []
    for x in items:
        k = (x["split"], x["video_id"])
        if k in seen:
            rep.setdefault("duplicates_removed", []).append(x["video_id"])
            continue
        seen.add(k)
        dedup.append(x)
    items = dedup
    for s in ("train", "val", "test"):
        rep[s] = sum(1 for x in items if x["split"] == s)
    nf = np.array([x["num_frames_csv"] - x["num_frames_escounts"] for x in items])
    rep["segment_frames_minus_escounts"] = {"mean": float(nf.mean()), "abs_max": float(np.abs(nf).max()),
                                             "within_2": float(np.mean(np.abs(nf) <= 2))}
    (root / "manifest.json").write_text(json.dumps(items, indent=0))
    (root / "prepare_report.json").write_text(json.dumps({k: (v if not isinstance(v, list) else {"n": len(v), "ids": v[:30]})
                                                          for k, v in rep.items()}, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}))


if __name__ == "__main__":
    main()
