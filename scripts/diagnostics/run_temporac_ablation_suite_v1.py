"""Run same-checkpoint TempoRAC ablations from safe per-identity NPZ caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pams.temporac.ablation_v1 import run_ablation_suite
from pams.temporac.response import TempoRACResponse

_CACHE_NAME = re.compile(r"v(?P<video>\d+)-p(?P<person>\d+)\.npz\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(path: Path, device: torch.device) -> TempoRACResponse:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")
    model = TempoRACResponse()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _labels(path: Path | None) -> dict[tuple[int, int], int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("label JSON must be a list")
    result: dict[tuple[int, int], int] = {}
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"video_index", "person_index", "count_gt"}:
            raise TypeError("each label row must have video_index, person_index, count_gt")
        key = (row["video_index"], row["person_index"])
        count = row["count_gt"]
        if any(type(value) is not int for value in (*key, count)) or count <= 0 or key in result:
            raise ValueError("label identities/counts must be unique nonnegative integers")
        result[key] = count
    return result


def _video_first_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_variant_video: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_variant_video[row["variant"]][row["video_index"]].append(
            (row["prediction"], row["count_gt"])
        )
    result: dict[str, dict[str, float]] = {}
    for variant, videos in sorted(by_variant_video.items()):
        video_mae: list[float] = []
        video_obo: list[float] = []
        for people in videos.values():
            video_mae.append(
                float(np.mean([abs(prediction - target) / target for prediction, target in people]))
            )
            video_obo.append(
                float(np.mean([abs(prediction - target) <= 1 for prediction, target in people]))
            )
        result[variant] = {
            "avg_mae": float(np.mean(video_mae)),
            "avg_obo": float(np.mean(video_obo)),
            "videos": len(videos),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)
    labels = _labels(args.labels_json)
    prediction_rows: list[dict[str, Any]] = []
    cache_files = sorted(args.cache_dir.glob("v*-p*.npz"))
    if args.limit is not None:
        cache_files = cache_files[: args.limit]
    variants: set[str] = set()
    global_available = True
    for cache_path in cache_files:
        match = _CACHE_NAME.fullmatch(cache_path.name)
        if match is None:
            raise ValueError(f"unexpected cache filename: {cache_path.name}")
        identity = (int(match.group("video")), int(match.group("person")))
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {"response_input", "bounds", "edge_mask", "local_gates", "windows"}
            if not required.issubset(cached.files):
                raise ValueError(f"{cache_path} lacks required safe arrays")
            response_input = np.array(cached["response_input"], dtype=np.float32, copy=True)
            bounds = np.array(cached["bounds"], dtype=np.int32, copy=True)
            edge_mask = np.array(cached["edge_mask"], dtype=np.uint8, copy=True)
            local_gates = np.array(cached["local_gates"], dtype=np.float64, copy=True)
            windows = np.array(cached["windows"], dtype=np.int32, copy=True)
            global_gates = (
                np.array(cached["global_gates"], dtype=np.float64, copy=True)
                if "global_gates" in cached.files
                else None
            )
        global_available = global_available and global_gates is not None
        with torch.inference_mode():
            probabilities = model.cached_responses(
                torch.from_numpy(response_input).to(device), bounds
            ).edge_probabilities
        outputs = run_ablation_suite(
            probabilities.detach().cpu().numpy(),
            local_gates,
            windows,
            edge_mask,
            bounds,
            global_gates=global_gates,
        )
        for name, output in outputs.items():
            variants.add(name)
            row: dict[str, Any] = {
                "cache_file": cache_path.name,
                "video_index": identity[0],
                "person_index": identity[1],
                "variant": name,
                "prediction": output.count,
            }
            if labels is not None:
                if identity not in labels:
                    raise ValueError(f"labels lack identity {identity}")
                row["count_gt"] = labels[identity]
            prediction_rows.append(row)
    if labels is not None and set(labels) != {
        (row["video_index"], row["person_index"]) for row in prediction_rows
    }:
        raise ValueError("label identity set differs from evaluated cache identity set")
    report: dict[str, Any] = {
        "schema": "temporac.ablation-diagnostic.v1",
        "scientific_status": "SUPPLIED_TRACK_PARTIAL_CACHE_DIAGNOSTIC_NOT_PAPER_MAIN",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "identities": len(cache_files),
        "variants_completed": sorted(variants),
        "variants_skipped": ([] if global_available else ["global_tempo_soft_nola"]),
        "metric_protocol": (
            "video-first person-relative AvgMAE/AvgOBO" if labels is not None else "prediction-only"
        ),
        "metrics": _video_first_metrics(prediction_rows) if labels is not None else None,
        "predictions": prediction_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "predictions"}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
