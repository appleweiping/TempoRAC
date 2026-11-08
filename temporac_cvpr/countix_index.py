"""Index the Kinetics-700 HF mirror zips remotely and match Countix clips.

Reads only each zip's central directory (HTTP range requests), builds
youtube_id -> [(zip, member, size)], and matches every Countix row
(official train annotation from allenai/Molmo2-Countix, ESCounts train/val/test
splits) to its Kinetics clip.  Writes the index and a coverage report; no media
is downloaded here.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://hf-mirror.com/datasets/atalaydenknalbant/Kinetics-700/resolve/main/Kinetics700_part_{:03d}.zip"
PAT = re.compile(r"^(train|val|validation|test)/(?:.*/)?(.{11})_(\d{6})_(\d{6})\.mp4$")


def list_zip(i: int):
    from remotezip import RemoteZip

    url = BASE.format(i)
    with RemoteZip(url) as z:
        return i, [(info.filename, info.file_size, info.compress_size) for info in z.infolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--escounts", required=True, help="EveryShotCounts/datasets/countix")
    ap.add_argument("--official-train", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "kinetics700_index.json"
    if idx_path.exists():
        entries = json.loads(idx_path.read_text())
    else:
        with ThreadPoolExecutor(8) as ex:
            res = dict(ex.map(list_zip, range(1, 23)))
        entries = [(i, n, s, c) for i, lst in sorted(res.items()) for n, s, c in lst]
        idx_path.write_text(json.dumps(entries))
    by_id = defaultdict(list)
    for zi, name, size, csize in entries:
        m = PAT.match(name)
        if m:
            by_id[m.group(2)].append({"zip": zi, "member": name, "size": size, "split": m.group(1),
                                      "start": int(m.group(3)), "end": int(m.group(4))})
    rep = {"zip_entries": len(entries), "mp4_ids": len(by_id),
           "multi_clip_ids": sum(1 for v in by_id.values() if len(v) > 1)}

    off = pd.read_csv(args.official_train)
    off_key = {}
    for _, r in off.iterrows():
        off_key[r["video_id"]] = r
    matched = []
    for split, fn in (("train", "new_train.csv"), ("val", "new_val.csv"), ("test", "new_test.csv")):
        df = pd.read_csv(Path(args.escounts) / fn)
        n_ok = n_amb = n_miss = 0
        for _, r in df.iterrows():
            stem = Path(r["video_id"]).stem
            yid = stem[:11]
            a, b = (float(x) for x in stem[12:].split("_"))
            cands = by_id.get(yid, [])
            pick = None
            if yid in off_key:  # official kinetics window disambiguates
                o = off_key[yid]
                pick = next((c for c in cands if c["start"] == int(o["kinetics_start"])), None)
            if pick is None and len(cands) == 1:
                pick = cands[0]
            if pick is None and len(cands) > 1:
                # keep only clips whose 10 s window can contain the relative repetition span
                ok = [c for c in cands if b <= (c["end"] - c["start"]) + 0.5]
                pick = ok[0] if len(ok) == 1 else None
                n_amb += pick is None
            if pick is None:
                n_miss += not cands
                continue
            n_ok += 1
            L = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
            bounds = [float(r[c]) for c in sorted(L, key=lambda c: int(c[1:])) if not pd.isna(r[c])]
            matched.append({"split": split, "name": stem, "youtube_id": yid, "rep_start": a, "rep_end": b,
                            "count": float(r["counts"]), "num_frames_csv": float(r["num_frames"]),
                            "bounds_csv": bounds, **pick})
        rep[split] = {"rows": len(df), "matched": n_ok, "ambiguous": n_amb, "missing": n_miss}
        print(split, rep[split], flush=True)
    rep["bytes_needed"] = int(sum(m["size"] for m in matched))
    (out / "countix_matched.json").write_text(json.dumps(matched, indent=0))
    (out / "countix_index_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
