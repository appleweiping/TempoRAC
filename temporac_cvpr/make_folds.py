"""K-fold cross-fitting manifests for out-of-fold (OOF) stage-1 predictions on the training videos.

A second stage that is trained on the stage-1 model's predictions for its own training videos sees
almost no errors (the decoder fits them); the standard remedy (stacking) is to train it on
out-of-fold predictions.  Fold f renames its held-out training items to <ds>_oof (rc_oof, cx_oof,
ucf_oof) so a stage-1 run trained on the rest predicts them with ``--with-test``; validation splits
are unchanged and still select the checkpoint.  Groups never straddle folds: UCFRep by UCF101 group
(clips of one group are cut from the same source video), Countix by YouTube id, RepCount-A by video.
Writes MANIFEST_DIR/manifest_fold{f}.json and folds.json.
Usage: python -m temporac_cvpr.make_folds MANIFEST OUT_DIR [K=5] [SEED=0]
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

TRAIN = {"rc_train": "rc_oof", "cx_train": "cx_oof", "ucf_train": "ucf_oof"}


def group_of(it: dict) -> str:
    if it["dataset"] == "UCFRep":
        return f"ucf:{it['action']}:{it['group']}"
    if it["dataset"] == "Countix":
        return f"cx:{it['video_id'].rsplit('_', 2)[0]}"
    return f"rc:{it['video_id']}"


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    m = json.loads(src.read_text())
    groups = defaultdict(list)
    for i, it in enumerate(m):
        if it["split"] in TRAIN:
            groups[(it["dataset"], group_of(it))].append(i)
    fold_of = {}
    for ds in sorted({d for d, _ in groups}):
        gs = sorted(g for d, g in groups if d == ds)
        random.Random(f"{seed}:{ds}").shuffle(gs)
        # greedy balance by number of videos
        load = [0] * K
        for g in sorted(gs, key=lambda g: -len(groups[(ds, g)])):
            f = min(range(K), key=lambda k: (load[k], k))
            fold_of[(ds, g)] = f
            load[f] += len(groups[(ds, g)])
        print(ds, "videos per fold", load, "groups", len(gs))
    out.mkdir(parents=True, exist_ok=True)
    assign = {}
    for f in range(K):
        mf = []
        for i, it in enumerate(m):
            it2 = dict(it)
            if it["split"] in TRAIN and fold_of[(it["dataset"], group_of(it))] == f:
                it2["split"] = TRAIN[it["split"]]
                assign[f"{it['split']}__{it['video_id']}"] = f
            mf.append(it2)
        (out / f"manifest_fold{f}.json").write_text(json.dumps(mf))
    (out / "folds.json").write_text(json.dumps({"K": K, "seed": seed, "fold_of": assign}, indent=0))
    print("held-out items", len(assign))


if __name__ == "__main__":
    main()
