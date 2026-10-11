"""Aggregate three frozen video-first TempoRAC ablation score files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    if len(args.score) != 3:
        parser.error("exactly three ordered seed score files are required")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.score]
    variants = sorted(payloads[0]["metrics"])
    if any(sorted(payload["metrics"]) != variants for payload in payloads):
        raise ValueError("metric variants differ across seeds")
    rows: dict[str, dict[str, float | list[float]]] = {}
    for variant in variants:
        mae = np.asarray(
            [payload["metrics"][variant]["avg_mae"] for payload in payloads],
            dtype=np.float64,
        )
        obo = np.asarray(
            [payload["metrics"][variant]["avg_obo"] for payload in payloads],
            dtype=np.float64,
        )
        rows[variant] = {
            "avg_mae_by_seed": mae.tolist(),
            "avg_mae_mean": float(np.mean(mae)),
            "avg_mae_std_population": float(np.std(mae, ddof=0)),
            "avg_mae_std_sample": float(np.std(mae, ddof=1)),
            "avg_obo_by_seed": obo.tolist(),
            "avg_obo_mean": float(np.mean(obo)),
            "avg_obo_std_population": float(np.std(obo, ddof=0)),
            "avg_obo_std_sample": float(np.std(obo, ddof=1)),
        }
    report: dict[str, Any] = {
        "schema": "temporac.ablation-three-seed-video-first-summary.v1",
        "scientific_status": "SUPPLIED_TRACK_PARTIAL_CACHE_DIAGNOSTIC_NOT_PAPER_MAIN",
        "seed_order": [20260815, 20260816, 20260817],
        "aggregation": "person relative error -> within-video mean -> equal-video mean",
        "obo_definition": "absolute per-person count error <= 1",
        "score_receipts": [
            {"path": str(path), "sha256": _sha256(path)} for path in args.score
        ],
        "metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
