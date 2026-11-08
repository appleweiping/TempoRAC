"""Generate frozen final-evaluation specs (multi-seed; test scored once per model).

Usage:
  python -m temporac_cvpr.make_final_specs SPEC_IN.json OUTDIR
SPEC_IN = {"full": {...cfg...},
           "ablations": {"wo_x": {...overrides...}, ...},      # RepCount-A only
           "baselines": {"single_r4": {...cfg...}, ...},         # all datasets
           "seeds": [0,1,2,3,4]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = "/root/autodl-tmp/TempoRAC"
DATASETS = {
    "repcount": {"manifest": [R + "/data/RepCountA/manifest_all.json"], "features": R + "/features/RepCountA_stmae_p4",
                 "splits": {}, "ablations": True},
    "ucfrep": {"manifest": [R + "/data/UCFRep/manifest_dev.json"], "features": R + "/features/UCFRep_stmae_p4",
               "splits": {"train_split": "train", "val_split": "dev", "test_split": "test"}, "ablations": False},
    "countix": {"manifest": [R + "/data/Countix/manifest.json"], "features": R + "/features/Countix_stmae_p4",
                "splits": {"train_split": "train", "val_split": "val", "test_split": "test"}, "ablations": False},
}


def main() -> None:
    spec_in = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    full = spec_in["full"]
    seeds = spec_in.get("seeds", [0, 1, 2, 3, 4])
    for ds, d in DATASETS.items():
        variants = {"full": full}
        abl = spec_in.get("ablations", {})
        if d["ablations"]:
            variants.update({k: dict(full, **v) for k, v in abl.items()})
        elif "wo_pyramid_r4" in abl:  # the pyramid's contribution is checked on every dataset
            variants["wo_pyramid_r4"] = dict(full, **abl["wo_pyramid_r4"])
        variants.update(spec_in.get("baselines", {}))
        runs = []
        for name, cfg in variants.items():
            c = dict(cfg)
            feats = c.pop("features", None)  # per-run feature directory (e.g. the no-overlap ablation)
            if ds != "repcount":
                c.setdefault("workers", 3)
            for s in seeds:
                r = {"name": f"{name}_s{s}", "seed": s, "cfg": c, "with_test": True}
                if feats:
                    r["features"] = feats
                runs.append(r)
        for i, r in enumerate(runs):
            r["gpu"] = i % 2
        spec = {"root": R, "manifest": d["manifest"], "features": d["features"], "group": f"final_{ds}", "runs": runs,
                **d["splits"]}
        (out / f"final_{ds}.json").write_text(json.dumps(spec, indent=1))
        print(ds, len(runs))


if __name__ == "__main__":
    main()
