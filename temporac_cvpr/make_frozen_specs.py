"""Specs for the frozen framework (= final_v2/wo_llm: two-depth multi-rate pyramid, pyramid shift, joint training
with dataset token and balanced sampling; original ST-MAE features; no language model).

final_v2's ablations and single-dataset runs were trained *with* the frozen-MLLM FiLM conditioning, so they do not
ablate this framework.  This writes the same runs with every MLLM key removed, plus seeds 3-4 of the main model
(seeds 0-2 are final_v2/wo_llm_s0..2).  Two disjoint specs: GPU0 (4 concurrent) and GPU1 (small-memory runs,
2 concurrent, <= 5 GB in total).  Usage: python -m temporac_cvpr.make_frozen_specs
"""
from __future__ import annotations

import json
from pathlib import Path

S = Path("/root/autodl-tmp/TempoRAC/code/temporac_cvpr/specs")
MLLM = ("mllm_root", "mllm_glob", "mllm_vis_layer", "mllm_film", "mllm_drop", "mllm_stream", "mllm_glob_dim")


def strip(r: dict) -> dict:
    r = json.loads(json.dumps(r))
    for k in MLLM:
        r["cfg"].pop(k, None)
    return r


def main() -> None:
    v2 = json.loads((S / "final_v2.json").read_text())
    single = json.loads((S / "final_v2_single.json").read_text())
    runs = []
    for r in v2["runs"]:
        n = r["name"]
        if n.startswith("wo_llm_s") or n.startswith("full_s"):
            continue
        runs.append(strip(r))
    for sd in (3, 4):
        base = next(x for x in v2["runs"] if x["name"] == "wo_llm_s0")
        r = strip(base)
        r["name"], r["seed"] = f"full_s{sd}", sd
        runs.append(r)
    for r in single["runs"]:
        x = strip(r)
        x.setdefault("features", single["features"])
        runs.append(x)
    assert not any(k in r["cfg"] for r in runs for k in MLLM)
    common = {k: v for k, v in v2.items() if k != "runs"}
    common["group"] = "frozen_a"
    g1_names = {r["name"] for r in runs if r["name"].startswith(("wo_depth", "wo_pyramid"))}
    for g, sel in ((0, [r for r in runs if r["name"] not in g1_names]), (1, [r for r in runs if r["name"] in g1_names])):
        for r in sel:
            r["gpu"] = g
        (S / f"frozen_a_g{g}.json").write_text(json.dumps({**common, "runs": sel}, indent=1))
        print(f"frozen_a_g{g}.json", len(sel), [r["name"] for r in sel])


if __name__ == "__main__":
    main()
