"""Union of the public RAC training sets for joint training of the counting head.

Writes one manifest whose ``split`` values are prefixed with the dataset
(``rc_train``, ``rc_valid``, ``rc_test``, ``cx_train``, ``cx_val``, ``cx_test``,
``ucf_train``, ``ucf_dev``, ``ucf_test``) while ``fsplit`` keeps the original
feature-file split, and one feature root made of symlinks to the per-dataset
feature directories (no copies).  Only training splits are ever used for
optimisation; validation/test splits keep their per-dataset roles.

Usage: python -m temporac_cvpr.make_union TAG  (TAG = stmae_p4 | vjepaL_p4)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/TempoRAC")
SOURCES = [  # prefix, manifest, feature dir stem
    ("rc", R / "data/RepCountA/manifest_all.json", "RepCountA"),
    ("cx", R / "data/Countix/manifest.json", "Countix"),
    ("ucf", R / "data/UCFRep/manifest_dev.json", "UCFRep"),  # group-wise dev split of the official train set
]


def main() -> None:
    tag = sys.argv[1]
    out_feat = R / "features" / f"UNION_{tag}"
    union, seen = [], set()
    for prefix, man, stem in SOURCES:
        src = R / "features" / f"{stem}_{tag}"
        for sub in os.listdir(src):
            if not (src / sub).is_dir():
                continue
            (out_feat / sub).mkdir(parents=True, exist_ok=True)
            for f in os.listdir(src / sub):
                dst = out_feat / sub / f
                if not dst.exists():
                    dst.symlink_to(src / sub / f)
        for it in json.loads(man.read_text()):
            it = dict(it)
            it["fsplit"] = it.get("fsplit", it["split"])
            it["split"] = f"{prefix}_{it['split']}"
            key = f"{it['fsplit']}__{it['video_id']}"
            assert key not in seen, f"feature key collision across datasets: {key}"
            seen.add(key)
            union.append(it)
    (R / "data/UNION").mkdir(parents=True, exist_ok=True)
    p = R / "data/UNION/manifest_dev.json"
    p.write_text(json.dumps(union))
    from collections import Counter

    print(p, len(union), dict(Counter(it["split"] for it in union)))
    print(out_feat, {d: len(os.listdir(out_feat / d)) for d in sorted(os.listdir(out_feat))})


if __name__ == "__main__":
    main()
