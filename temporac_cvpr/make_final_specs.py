"""Generate frozen final-evaluation specs (multi-seed, test scored once per model).

Variants (same frozen tokens, same budget):
  full            TempoRAC: multi-rate pyramid {2,4,8} + tempogram-biased attention + tempo-warp training
  wo_pyramid      single finest rate (2) with the same tempogram bias and warp training
  wo_tgbias       pyramid + warp training, plain attention
  wo_warp         pyramid + tempogram bias, no warp training
  base_concat     pyramid only (= remove both B and C)
  single_r2/r4/r8 single-rate counters (ESCounts-style exemplar-free decoder)
  tsm             TransRAC-style multi-scale banded TSM head on the pyramid
  repnet          RepNet-style period/periodicity head on rate 2
Usage: python -m temporac_cvpr.make_final_specs FULL_CFG_JSON SEEDS OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = "/root/autodl-tmp/TempoRAC"
DATASETS = {
    "repcount": {"manifest": [R + "/data/RepCountA/manifest_all.json"], "features": R + "/features/RepCountA_stmae_p4",
                 "splits": {}},
    "ucfrep": {"manifest": [R + "/data/UCFRep/manifest_dev.json"], "features": R + "/features/UCFRep_stmae_p4",
               "splits": {"train_split": "train", "val_split": "dev", "test_split": "test"}},
    "countix": {"manifest": [R + "/data/Countix/manifest.json"], "features": R + "/features/Countix_stmae_p4",
                "splits": {"train_split": "train", "val_split": "val", "test_split": "test"}},
}


COMPONENT_OFF = {  # how each TempoRAC component is removed in the ablation
    "tg_bias": {"tg_bias": False},
    "tempo_gate": {"tempo_gate": False},
    "periodic_pool": {"periodic_pool": False},
    "tc_decode": {"tc_decode": False},
    "warp": {"warp_prob": 0.0},
}


def variants(full: dict) -> dict:
    v = {"full": dict(full)}
    v["wo_pyramid"] = dict(full, rates=[full["rates"][0]])
    on = [k for k in COMPONENT_OFF if (full.get(k) if k != "warp" else full.get("warp_prob", 0) > 0)]
    base = dict(full)
    for k in on:
        v[f"wo_{k}"] = dict(full, **COMPONENT_OFF[k])
        base.update(COMPONENT_OFF[k])
    v["base_concat"] = base
    for s in (2, 4, 8):
        v[f"single_r{s}"] = {"head": "single", "rates": [s], "pool_to": 4}
    v["tsm"] = {"head": "tsm", "rates": list(full["rates"]), "pool_to": 4}
    v["repnet"] = {"head": "repnet", "rates": [full["rates"][0]], "pool_to": 4}
    return v


def main() -> None:
    full = json.loads(sys.argv[1])
    seeds = [int(s) for s in sys.argv[2].split(",")]
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for ds, d in DATASETS.items():
        runs = []
        for name, cfg in variants(full).items():
            for s in seeds:
                runs.append({"name": f"{name}_s{s}", "seed": s, "cfg": cfg, "with_test": True})
        for i, r in enumerate(runs):
            r["gpu"] = i % 2
        spec = {"root": R, "manifest": d["manifest"], "features": d["features"], "group": f"final_{ds}", "runs": runs,
                **d["splits"]}
        (out / f"final_{ds}.json").write_text(json.dumps(spec, indent=1))
        print(ds, len(runs))


if __name__ == "__main__":
    main()
