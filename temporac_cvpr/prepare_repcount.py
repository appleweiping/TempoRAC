"""Materialise RepCount-A (LLSP, 758/131/152) from the Lance mirror and build a manifest.

The Lance mirror (lmms-lab-eval/repcounta-lance) stores the original media bytes of
``RepCountA.tar.gz`` together with the official count and the L1..L302 cycle bounds.
We write every video to disk unchanged, hash it, and cross-check counts/bounds
against the independent ESCounts CSV copy of the official annotation.

Usage (server):
  python -m temporac_cvpr.prepare_repcount --lance ROOT/data/raw/repcounta-lance/data \
      --out ROOT/data/RepCountA --escounts-csv ROOT/third_party/EveryShotCounts/datasets/repcount
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def bounds_from_row(row: pd.Series) -> list[float]:
    cols = [c for c in row.index if c.startswith("L") and c[1:].isdigit()]
    cols.sort(key=lambda c: int(c[1:]))
    vals = [float(row[c]) for c in cols if not (isinstance(row[c], float) and math.isnan(row[c]))]
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--escounts-csv", required=True)
    ap.add_argument("--splits", default="train,valid,test")
    args = ap.parse_args()

    import lance  # pylance

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    report = {"splits": {}}
    wanted = set(args.splits.split(","))
    tag = "_".join(sorted(wanted))
    for lance_split, split in SPLITS.items():
        if split not in wanted:
            continue
        ds = lance.dataset(str(Path(args.lance) / f"{lance_split}.lance"))
        cols = [f.name for f in ds.schema]
        meta_cols = [c for c in cols if c != "video_blob"]
        meta = ds.to_table(columns=meta_cols).to_pandas()
        esc = pd.read_csv(Path(args.escounts_csv) / f"{split}_with_fps.csv")
        esc = esc.set_index("name")
        vdir = out / split
        vdir.mkdir(parents=True, exist_ok=True)
        n_count_mismatch, n_bound_mismatch, n_missing_esc = 0, 0, 0
        for i in range(len(meta)):
            row = meta.iloc[i]
            # video_blob is a Lance blob column: read the raw media bytes through BlobFile.
            with ds.take_blobs("video_blob", indices=[i])[0] as bf:
                blob = bf.read()
            assert len(blob) == int(row["video_size_bytes"]), (i, len(blob), row["video_size_bytes"])
            name = str(row["source_name"])
            path = vdir / name
            if not path.exists() or path.stat().st_size != len(blob):
                path.write_bytes(blob)
            sha = hashlib.sha256(blob).hexdigest()
            bounds = json.loads(row["cycle_bounds_json"]) if isinstance(row["cycle_bounds_json"], str) else []
            bounds = [float(b) for b in bounds if b is not None and not (isinstance(b, float) and math.isnan(b))]
            count = None if pd.isna(row["count"]) else float(row["count"])
            rec = {
                "dataset": "RepCountA",
                "split": split,
                "video_id": Path(name).stem,
                "file": str(path),
                "bytes": len(blob),
                "sha256": sha,
                "action": str(row.get("action_type", "")),
                "count": count,
                "bounds": bounds,
            }
            if name in esc.index:
                e = esc.loc[name]
                rec["fps_csv"] = float(e["fps"]) if "fps" in e and not pd.isna(e["fps"]) else None
                rec["num_frames_csv"] = float(e["num_frames"]) if "num_frames" in e and not pd.isna(e["num_frames"]) else None
                ec = None if pd.isna(e["count"]) else float(e["count"])
                if (ec is None) != (count is None) or (ec is not None and abs(ec - count) > 1e-6):
                    n_count_mismatch += 1
                    rec["count_escounts"] = ec
                eb = bounds_from_row(e)
                if len(eb) != len(bounds) or (len(eb) and np.max(np.abs(np.array(eb) - np.array(bounds))) > 1e-6):
                    n_bound_mismatch += 1
                    rec["bounds_escounts"] = eb
            else:
                n_missing_esc += 1
            manifest.append(rec)
        report["splits"][split] = {
            "rows": len(meta),
            "count_mismatch_vs_escounts": n_count_mismatch,
            "bound_mismatch_vs_escounts": n_bound_mismatch,
            "missing_in_escounts": n_missing_esc,
            "escounts_rows": len(esc),
        }
        print(split, report["splits"][split], flush=True)
    (out / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=1))
    (out / f"prepare_report_{tag}.json").write_text(json.dumps(report, indent=1))
    print("PREPARE_DONE", len(manifest))


if __name__ == "__main__":
    main()
