"""Validation-only: share of the RepCount-A validation error (original view) of RepCount-A-only decoders that comes
from videos over-counted by a factor in [1.15, 2.4] (corrected-protocol wo_joint_rc_s0..2)."""
import json

import numpy as np

R = "/root/autodl-tmp/TempoRAC"
rows = json.load(open(R + "/data/UNION_known_source_v1/manifest_dev.json"))
man = {"%s__%s" % (r["split"], r["video_id"]): float(r["count"]) for r in rows if r["split"] == "rc_valid"}
for s in (0, 1, 2):
    p = json.load(open("%s/runs/frozen_known_source_v1_ablation20261112/wo_joint_rc_s%d/predictions.json" % (R, s)))["val"]
    keys = [k for k in p if man.get(k, 0) > 0]
    g = np.array([man[k] for k in keys])
    c = np.round(np.array([p[k] for k in keys]))
    err = np.abs(c - g) / (g + 0.1)
    ratio = c / g
    over = (ratio >= 1.15) & (ratio <= 2.4)
    print("seed", s, "n", len(keys), "mae_p %.4f" % err.mean(), "share_over %.3f" % (err[over].sum() / err.sum()),
          "n_over", int(over.sum()), "share_under %.3f" % (err[ratio < 1 / 1.15].sum() / err.sum()))
