"""Score frozen ablation predictions with video-first AvgMAE and AvgOBO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty level")
    return math.fsum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-label-subset", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")

    prediction_payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    label_payload = json.loads(args.labels.read_text(encoding="utf-8"))
    predictions = prediction_payload.get("predictions")
    labels = label_payload.get("labels")
    if not isinstance(predictions, list) or not isinstance(labels, list):
        raise TypeError("prediction and label payloads must contain row lists")
    label_map: dict[tuple[int, int], int] = {}
    for row in labels:
        if not isinstance(row, dict) or set(row) != {"video_index", "person_index", "count_gt"}:
            raise TypeError("label rows have an unexpected schema")
        key = (row["video_index"], row["person_index"])
        if key in label_map or type(row["count_gt"]) is not int or row["count_gt"] <= 0:
            raise ValueError("labels must be unique positive-integer observations")
        label_map[key] = row["count_gt"]

    grouped: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    prediction_identities: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in predictions:
        if not isinstance(row, dict):
            raise TypeError("prediction rows must be mappings")
        key = (row["video_index"], row["person_index"])
        variant = row["variant"]
        estimate = row["prediction"]
        if key not in label_map and args.allow_label_subset:
            continue
        if (
            not isinstance(variant, str)
            or type(estimate) is not int
            or estimate < 0
            or key not in label_map
            or key in prediction_identities[variant]
        ):
            raise ValueError("prediction identity, variant, or estimate is invalid")
        prediction_identities[variant].add(key)
        grouped[variant][key[0]].append((estimate, label_map[key]))
    for variant, identities in prediction_identities.items():
        if identities != set(label_map):
            raise ValueError(f"variant {variant} does not cover the exact label population")

    metrics: dict[str, dict[str, float | int]] = {}
    for variant, videos in sorted(grouped.items()):
        per_video_mae: list[float] = []
        per_video_obo: list[float] = []
        for people in videos.values():
            per_video_mae.append(
                _mean([abs(estimate - target) / target for estimate, target in people])
            )
            per_video_obo.append(
                _mean([float(abs(estimate - target) <= 1) for estimate, target in people])
            )
        metrics[variant] = {
            "avg_mae": _mean(per_video_mae),
            "avg_obo": _mean(per_video_obo),
            "videos": len(videos),
            "identities": len(prediction_identities[variant]),
        }
    excluded = prediction_payload.get("identities", len(label_map)) - len(label_map)
    report: dict[str, Any] = {
        "schema": "temporac.ablation-video-first-score.v1",
        "scientific_status": (
            "PARTIAL_IDENTITY_DIAGNOSTIC_NOT_PAPER_MAIN"
            if excluded
            else "SUPPLIED_TRACK_PARTIAL_CACHE_DIAGNOSTIC_NOT_PAPER_MAIN"
        ),
        "aggregation": "person-relative error -> mean within video -> mean across videos",
        "obo_definition": "absolute count error <= 1 per person",
        "predictions_sha256": _sha256(args.predictions),
        "labels_sha256": _sha256(args.labels),
        "evaluated_identity_count": len(label_map),
        "excluded_prediction_identities": excluded,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
