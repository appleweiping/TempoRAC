"""Build the UCFRep manifest (Context/ESCounts protocol, 421 train / 105 val).

Protocol: each video is trimmed to its annotated repetitive segment
``[start_frame, end_frame]`` (1-based, inclusive) and the target is the number of
annotated cycles inside it, exactly as in the ESCounts UCFRep loader.  Counts and
cycle bounds from the ESCounts CSV are cross-checked against the official Context
``.mat`` annotations (``temporal_bound``; count = len(bounds) - 1).

The official 105-video ``val`` split is the community evaluation split and is
named ``test`` here; it is never used for model selection.  A development split is
carved out of the 421 training videos by whole UCF101 groups (clips of one group
are cut from the same source video), stratified per class, with a fixed seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--escounts-csv", required=True)
    ap.add_argument("--context-ann", required=True)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--dev-seed", type=int, default=2026)
    args = ap.parse_args()
    from scipy.io import loadmat

    out = Path(args.out)
    vdir = out / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.ucf_zip)
    names = {Path(n).name: n for n in zf.namelist() if n.endswith(".avi")}
    items, report = [], {}
    for csv_split, fsplit, mat_split in (("new_train", "train", "train"), ("new_val", "test", "val")):
        df = pd.read_csv(Path(args.escounts_csv) / f"{csv_split}.csv")
        L = sorted([c for c in df.columns if c.startswith("L") and c[1:].isdigit()], key=lambda c: int(c[1:]))
        mism = []
        for _, r in df.iterrows():
            fname = Path(r["name"]).name
            stem = Path(fname).stem
            data = zf.read(names[fname])
            p = vdir / fname
            if not p.exists() or p.stat().st_size != len(data):
                p.write_bytes(data)
            b = np.array([r[c] for c in L], dtype=float)
            b = b[~np.isnan(b)]
            start, end = int(r["start_frame"]), int(r["end_frame"])
            m = loadmat(Path(args.context_ann) / mat_split / f"{stem}.mat")["label"][0, 0]
            tb = np.asarray(m["temporal_bound"], dtype=float).reshape(-1)
            ms, me = int(np.asarray(m["start_frame"]).reshape(-1)[0]), int(np.asarray(m["end_frame"]).reshape(-1)[0])
            pairs = np.stack([tb[:-1], tb[1:]], 1).reshape(-1)
            ok = (len(tb) - 1 == int(r["counts"])) and ms == start and me == end and len(pairs) == len(b) \
                and np.allclose(pairs, b)
            if not ok:
                mism.append(stem)
            items.append({
                "dataset": "UCFRep", "fsplit": fsplit, "split": fsplit, "video_id": stem, "file": str(p),
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "action": Path(r["name"]).parent.name, "group": stem.split("_")[-2],
                "count": float(r["counts"]),
                # 0-based inclusive decoded-frame range of the annotated segment
                "frame_range": [start - 1, end - 1],
                "bounds": [float(x - start) for x in b],
                "num_frames_csv": float(end - start + 1),
            })
        report[fsplit] = {"rows": len(df), "mismatch_vs_context_mat": len(mism), "mismatch_ids": mism[:20]}
        print(fsplit, report[fsplit], flush=True)

    # group-level overlap between official train and test (reported, not altered)
    key = lambda it: (it["action"], it["group"])
    tr_groups = {key(it) for it in items if it["fsplit"] == "train"}
    te_groups = {key(it) for it in items if it["fsplit"] == "test"}
    report["train_test_shared_class_groups"] = len(tr_groups & te_groups)

    rng = random.Random(args.dev_seed)
    by_class = defaultdict(list)
    for g in sorted(tr_groups):
        by_class[g[0]].append(g)
    dev_groups = set()
    for c, gs in sorted(by_class.items()):
        gs = gs[:]
        rng.shuffle(gs)
        n_vid = {g: sum(1 for it in items if it["fsplit"] == "train" and key(it) == g) for g in gs}
        total = sum(n_vid.values())
        acc = 0
        for g in gs:
            if acc >= args.dev_frac * total or len(gs) - len([x for x in gs if x in dev_groups]) <= 1:
                break
            dev_groups.add(g)
            acc += n_vid[g]
    full, dev = [], []
    for it in items:
        full.append(dict(it))
        d = dict(it)
        if it["fsplit"] == "train":
            d["split"] = "dev" if key(it) in dev_groups else "train"
        dev.append(d)
    report["dev_protocol"] = {s: sum(1 for d in dev if d["split"] == s) for s in ("train", "dev", "test")}
    (out / "manifest_full.json").write_text(json.dumps(full, indent=1))
    (out / "manifest_dev.json").write_text(json.dumps(dev, indent=1))
    (out / "prepare_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), "\nPREPARE_DONE", len(items))


if __name__ == "__main__":
    main()
