"""Hyper-parameter sensitivity sweep around a frozen configuration (validation only).

One factor at a time, 3 seeds each; the centre configuration is shared.
Usage: python -m temporac_cvpr.make_hparam_specs CENTER_CFG_JSON OUT.json
"""
from __future__ import annotations

import json
import sys

R = "/root/autodl-tmp/TempoRAC"
SWEEPS = {
    "sigma_frac": [0.125, 0.25, 0.5],
    "max_lag": [80, 160, 240],
    "tempo_win": [24, 48, 96],
    "d": [128, 256, 384],
    "rates": [[2, 4], [2, 4, 8], [1, 2, 4, 8]],
    "lr": [1e-4, 2e-4, 4e-4],
}


def main() -> None:
    center = json.loads(sys.argv[1])
    runs = []
    for key, values in SWEEPS.items():
        for v in values:
            if v == center.get(key):
                continue  # the centre point is run once, separately
            cfg = dict(center, **{key: v})
            tag = "-".join(map(str, v)) if isinstance(v, list) else str(v)
            for s in range(3):
                runs.append({"name": f"hp_{key}_{tag}_s{s}", "seed": s, "cfg": cfg})
    for i, r in enumerate(runs):
        r["gpu"] = i % 2
    spec = {"root": R, "manifest": [R + "/data/RepCountA/manifest_all.json"],
            "features": R + "/features/RepCountA_stmae_p4", "group": "hparam", "runs": runs}
    with open(sys.argv[2], "w") as f:
        json.dump(spec, f, indent=1)
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
