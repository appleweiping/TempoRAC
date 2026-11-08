"""UCFRep literature-comparable protocol: retrain on all 421 training videos.

For every variant of the dev-selected UCFRep runs (group final_ucfrep), take the
median best epoch over seeds, retrain each seed on the full official training
split for that many epochs without any validation-based selection (last weights),
and score the 105-video evaluation split once.  Writes and launches a spec.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC")


def main() -> None:
    dev = R / "runs" / "final_ucfrep"
    spec_dev = json.loads((R / "code/temporac_cvpr/specs/final_ucfrep.json").read_text())
    cfg_by_variant = {re.sub(r"_s\d+$", "", r["name"]): r["cfg"] for r in spec_dev["runs"]}
    best = defaultdict(list)
    for d in dev.iterdir():
        res = d / "result.json"
        if res.exists():
            best[re.sub(r"_s\d+$", "", d.name)].append(json.loads(res.read_text())["best_epoch"] + 1)
    runs = []
    for v, eps in sorted(best.items()):
        E = int(np.median(eps))
        cfg = dict(cfg_by_variant[v], epochs=E)
        for s in range(5):
            runs.append({"name": f"{v}_s{s}", "seed": s, "cfg": cfg, "with_test": True, "gpu": len(runs) % 2})
        print(v, "epochs", E, "from", sorted(eps))
    spec = {"root": str(R), "manifest": [str(R / "data/UCFRep/manifest_full.json")],
            "features": str(R / "features/UCFRep_stmae_p4"), "group": "final_ucfrep_full421",
            "train_split": "train", "val_split": "none", "test_split": "test", "runs": runs}
    p = R / "code/temporac_cvpr/specs/final_ucfrep_full421.json"
    p.write_text(json.dumps(spec, indent=1))
    subprocess.run([str(R / "env/bin/python"), "-m", "temporac_cvpr.launch", str(p), "--max-per-gpu",
                    sys.argv[1] if len(sys.argv) > 1 else "1"], cwd=str(R / "code"), check=False)


if __name__ == "__main__":
    main()
