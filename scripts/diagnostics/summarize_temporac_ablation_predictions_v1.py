"""Summarize frozen TempoRAC ablation predictions without using labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

FULL = "full_local_soft_nola"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")

    by_seed: dict[int, dict[str, dict[tuple[int, int], int]]] = {}
    receipts: list[dict[str, Any]] = []
    for path in args.prediction:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_name = Path(payload["checkpoint"]).name
        seed_text = checkpoint_name.split("seed", 1)[1].split("_", 1)[0]
        seed = int(seed_text)
        variant_rows: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
        for row in payload["predictions"]:
            identity = (row["video_index"], row["person_index"])
            variant = row["variant"]
            if identity in variant_rows[variant]:
                raise ValueError("duplicate identity within variant")
            variant_rows[variant][identity] = row["prediction"]
        if seed in by_seed:
            raise ValueError("duplicate seed")
        by_seed[seed] = dict(variant_rows)
        receipts.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": _sha256(path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )

    seeds = sorted(by_seed)
    variants = sorted(next(iter(by_seed.values())))
    reference_identities = set(by_seed[seeds[0]][FULL])
    for seed in seeds:
        if sorted(by_seed[seed]) != variants:
            raise ValueError("variant inventory changes across seeds")
        for variant in variants:
            if set(by_seed[seed][variant]) != reference_identities:
                raise ValueError("identity population changes across seed/variant")

    per_seed: dict[str, dict[str, dict[str, float | int]]] = {}
    for seed in seeds:
        full = by_seed[seed][FULL]
        seed_rows: dict[str, dict[str, float | int]] = {}
        for variant in variants:
            values = np.asarray(list(by_seed[seed][variant].values()), dtype=np.float64)
            deltas = np.asarray(
                [abs(by_seed[seed][variant][identity] - full[identity]) for identity in full],
                dtype=np.float64,
            )
            seed_rows[variant] = {
                "mean_predicted_count": float(np.mean(values)),
                "median_predicted_count": float(np.median(values)),
                "p90_predicted_count": float(np.quantile(values, 0.9)),
                "maximum_predicted_count": int(np.max(values)),
                "zero_fraction": float(np.mean(values == 0)),
                "mean_absolute_delta_from_full": float(np.mean(deltas)),
                "changed_fraction_from_full": float(np.mean(deltas > 0)),
            }
        per_seed[str(seed)] = seed_rows

    cross_seed: dict[str, dict[str, float]] = {}
    ordered_identities = sorted(reference_identities)
    for variant in variants:
        matrix = np.asarray(
            [
                [by_seed[seed][variant][identity] for identity in ordered_identities]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        cross_seed[variant] = {
            "mean_identity_seed_std": float(np.mean(np.std(matrix, axis=0))),
            "all_seed_exact_agreement_fraction": float(
                np.mean(np.all(matrix == matrix[0:1], axis=0))
            ),
            "grand_mean_predicted_count": float(np.mean(matrix)),
        }
        if not all(math.isfinite(value) for value in cross_seed[variant].values()):
            raise ValueError("nonfinite summary statistic")

    report = {
        "schema": "temporac.ablation-label-free-summary.v1",
        "scientific_status": "LABEL_FREE_DIAGNOSTIC_NOT_ACCURACY_EVIDENCE",
        "seeds": seeds,
        "identity_count": len(reference_identities),
        "variants": variants,
        "prediction_receipts": receipts,
        "per_seed": per_seed,
        "cross_seed": cross_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
