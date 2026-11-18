"""Annotation-only tempo statistics of RepCount-A quoted in the introduction and the dataset paragraph.

Run on the first server (2026-11-17) from /root/autodl-tmp/TempoRAC with /root/miniconda3/bin/python3; no model involved.
Per video with a positive count: cycle durations (end - start) / CSV fps from the annotated bounds; the per-video median
cycle duration (5th-95th percentile over videos) and the coefficient of variation of the cycle durations
(population std / mean; videos with at least two cycles). Output in tempo_stats.txt.
"""
import json

import numpy as np

for path in ["data/UNION/manifest_dev.json", "data/UNION_known_source_v1/manifest_dev.json"]:
    rows = [r for r in json.load(open(path))
            if r["dataset"] == "RepCountA" and (r.get("count") or 0) > 0 and r.get("bounds")]
    med, cv = [], []
    for r in rows:
        b = np.asarray(r["bounds"], float).reshape(-1, 2)
        d = (b[:, 1] - b[:, 0]) / r["fps_csv"]
        d = d[d > 0]
        if len(d) == 0:
            continue
        med.append(np.median(d))
        if len(d) >= 2:
            cv.append(d.std(ddof=0) / d.mean())
    med, cv = np.asarray(med), np.asarray(cv)
    print(path, "videos", len(med), "p5/p95 median cycle s", np.round(np.percentile(med, [5, 95]), 3),
          "| cv>0.25: %.3f of %d videos with >=2 cycles" % ((cv > 0.25).mean(), len(cv)))
