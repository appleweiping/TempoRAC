"""FULL-v1 proposed CPU reference. Independent of frozen TempoRAC metrics.

See docs/MULTIREP_METRIC_REFERENCE_20261104.md for schema and limitations.
No decoder, rounding, track truncation, support masking, or data access occurs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

CONTRACT_STATUS = "proposed-reference-not-official"
TEMPORAL_THRESHOLDS = tuple((50 + 5 * i) / 100 for i in range(10))
SUPPORT = {"observed", "interpolated", "extrapolated", "missing"}


class ContractError(ValueError):
    """An input cannot be scored under the proposed contract."""


def _id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ContractError(f"invalid ID: {value!r}")
    return value


def _key(value):
    return (0, value) if isinstance(value, int) else (1, value)


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ContractError(f"invalid {name}: {value}")
    return value


def _list(value, name):
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be a list/tuple")
    return value


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    return value


def _field(row, name):
    if name not in row:
        raise ContractError(f"missing field: {name}")
    return row[name]


def _interval(value, frames):
    value = _list(value, "interval")
    if len(value) != 2:
        raise ContractError("interval must have exactly two endpoints")
    start, end = (_number(v, "interval endpoint") for v in value)
    if not 0 <= start < end <= frames:
        raise ContractError("require 0 <= start < end <= source frame count")
    return (start, end)


def inclusive_frame_span(start, end, frames):
    """Explicit inclusive integer frames [a,b] -> source boundaries [a,b+1]."""
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end, frames)):
        raise ContractError("inclusive frame indices and frame count must be integers")
    return _interval((start, end + 1), frames)


def map_boundaries(values, output_boundaries, source_boundaries, frames, *, inverse=False):
    """Explicit monotone boundary map, linear between verified control points.

    Caller supplies the measured mapping; stride is never guessed. Inverse maps
    source boundaries back to output coordinates using the identical controls.
    This does not manufacture/resample full source-frame bbox/support arrays.
    """
    frames = _number(frames, "frames", 1)
    out = tuple(
        _number(x, "output boundary") for x in _list(output_boundaries, "output boundaries")
    )
    src = tuple(
        _number(x, "source boundary") for x in _list(source_boundaries, "source boundaries")
    )
    if len(out) != len(src) or len(out) < 2 or out[0] != 0 or src[0] != 0 or src[-1] != frames:
        raise ContractError("mapping must align and span output/source origin through source T")
    if any(b <= a for seq in (out, src) for a, b in zip(seq, seq[1:], strict=False)):
        raise ContractError("mapping controls must be strictly increasing")
    x, y = (src, out) if inverse else (out, src)
    result = []
    for value in _list(values, "boundaries"):
        value = _number(value, "boundary")
        if not x[0] <= value <= x[-1]:
            raise ContractError("boundary outside verified map")
        j = next(j for j in range(len(x) - 1) if value <= x[j + 1])
        result.append(y[j] + (value - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j]))
    return result


def _track(row, frames, prediction):
    row = dict(_mapping(row, "track"))
    ident = _id(_field(row, "prediction_id" if prediction else "gt_id"))
    category = _id(row.get("category_id", 1))
    count = _number(_field(row, "count"), "count")
    if not prediction and count <= 0:
        raise ContractError("GT count must be positive; no epsilon patch allowed")
    bboxes = _list(_field(row, "bboxes"), "bboxes")
    support = _list(_field(row, "support"), "support")
    if len(bboxes) != frames or len(support) != frames:
        raise ContractError("bbox/support lengths must equal complete source frame count")
    boxes = []
    for box, state in zip(bboxes, support, strict=True):
        if (
            not isinstance(state, str)
            or state not in SUPPORT
            or (box is None) != (state == "missing")
        ):
            raise ContractError("invalid bbox/support relationship")
        if box is not None:
            box = _list(box, "bbox")
            if len(box) != 4:
                raise ContractError("bbox must be xywh")
            # Positions can be negative for a partially off-frame person.
            coords = tuple(_number(x, "bbox", -math.inf) for x in box)
            if coords[2] <= 0 or coords[3] <= 0:
                raise ContractError("bbox width/height must be positive")
            boxes.append(coords)
        else:
            boxes.append(None)
    periods = _field(row, "periods")
    parsed = None
    if periods is not None:
        parsed = []
        seen = set()
        for event in _list(periods, "periods"):
            event = _mapping(event, "event")
            event_id = _id(_field(event, "event_id"))
            if event_id in seen:
                raise ContractError("duplicate event ID in track")
            seen.add(event_id)
            interval = _interval(_field(event, "interval"), frames)
            score = (
                _number(_field(event, "score"), "event score", -math.inf) if prediction else None
            )
            parsed.append((event_id, interval, score))
    elif not prediction:
        raise ContractError("GT periods must be explicitly present (possibly empty)")
    result = dict(id=ident, category=category, count=count, bboxes=boxes, periods=parsed)
    if prediction:
        result["score"] = _number(_field(row, "track_score"), "track score", -math.inf)
        for name in ("decoder_sha", "config_sha", "checkpoint_sha"):
            value = _field(row, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in value)
            ):
                raise ContractError(f"invalid {name}; expected SHA256 hex")
        result["supplied_gt_id"] = row.get("supplied_gt_id")
        if result["supplied_gt_id"] is not None:
            _id(result["supplied_gt_id"])
    return result


def tube_iou(first, second):
    """Full-frame xywh tube IoU; one-sided missing frame contributes union."""
    if len(first) != len(second):
        raise ContractError("tube lengths differ")
    intersection = union = 0.0
    for a, b in zip(first, second, strict=True):
        aa = a[2] * a[3] if a is not None else 0.0
        ab = b[2] * b[3] if b is not None else 0.0
        inter = 0.0
        if a is not None and b is not None:
            inter = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
                0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            )
        intersection += inter
        union += aa + ab - inter
    return intersection / union if union > 0 else 0.0


def _tiou(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / ((a[1] - a[0]) + (b[1] - b[0]) - intersection)


def _period_ap(gt_events, predicted_events):
    # GT: (identity, event_id, interval); DT: (identity|None, tie_key, interval, score).
    n_gt = len(gt_events)
    if not n_gt:
        return {
            "status": "undefined-no-gt-periods",
            "n_gt": 0,
            "n_predictions": len(predicted_events),
            "thresholds": list(TEMPORAL_THRESHOLDS),
            "AP": None,
            "mAP": None,
            "AP50": None,
            "AP75": None,
        }
    ordered = sorted(predicted_events, key=lambda r: (-r[3], r[1]))
    aps = []
    for threshold in TEMPORAL_THRESHOLDS:
        used = set()
        precisions, recalls = [], []
        tp = 0
        for rank, (identity, _, interval, _) in enumerate(ordered, 1):
            candidates = [
                (j, _tiou(interval, g[2]), g[1])
                for j, g in enumerate(gt_events)
                if identity is not None and g[0] == identity and j not in used
            ]
            candidates.sort(key=lambda c: (-c[1], _key(c[2])))
            if candidates and candidates[0][1] >= threshold:
                used.add(candidates[0][0])
                tp += 1
            precisions.append(tp / rank)
            recalls.append(tp / n_gt)
        envelope = precisions + [0.0]
        for j in range(len(envelope) - 2, -1, -1):
            envelope[j] = max(envelope[j], envelope[j + 1])
        last = ap = 0.0
        for recall, precision in zip(recalls, envelope, strict=False):
            ap += (recall - last) * precision
            last = recall
        aps.append(ap)
    return {
        "status": "available",
        "n_gt": n_gt,
        "n_predictions": len(ordered),
        "thresholds": list(TEMPORAL_THRESHOLDS),
        "AP": aps,
        "mAP": sum(aps) / len(aps),
        "AP50": aps[0],
        "AP75": aps[5],
    }


def score(
    videos: Sequence[Mapping], prediction_videos: Sequence[Mapping], *, setting="predicted-track"
):
    """Score canonical complete population; malformed input fails the entire call.

    GT video: video_id, num_frames, tracks; GT track: gt_id, count, bboxes,
    support, periods=[{event_id, interval:[s,e]}], optional category_id=1.
    Prediction video: video_id, status='success'/'failure', tracks; failure needs
    failure_reason and zero tracks. Prediction track: prediction_id, track_score,
    count, bboxes, support, periods=[{event_id,interval,score}] or None, three
    *_sha provenance fields; supplied setting also requires supplied_gt_id.
    """
    if setting not in {"predicted-track", "supplied-track"}:
        raise ContractError("unknown setting")
    populations, all_gt = {}, set()
    for video in _list(videos, "videos"):
        video = _mapping(video, "video")
        video_id = _id(_field(video, "video_id"))
        frames = _field(video, "num_frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ContractError("num_frames must be a positive integer")
        if video_id in populations:
            raise ContractError("duplicate GT video ID")
        tracks = [_track(t, frames, False) for t in _list(_field(video, "tracks"), "GT tracks")]
        if not tracks:
            raise ContractError(
                "videos without repeating GT people require an explicit contract patch"
            )
        for track in tracks:
            if track["id"] in all_gt:
                raise ContractError("duplicate GT ID")
            all_gt.add(track["id"])
        populations[video_id] = (frames, tracks)
    if not populations:
        raise ContractError("empty declared GT population")
    predictions, failures = {}, {}
    for video in _list(prediction_videos, "prediction videos"):
        video = _mapping(video, "prediction video")
        video_id = _id(_field(video, "video_id"))
        if video_id not in populations or video_id in predictions:
            raise ContractError("unknown or duplicate prediction video ID")
        status = _field(video, "status")
        tracks = _list(_field(video, "tracks"), "prediction tracks")
        if status == "failure":
            reason = _field(video, "failure_reason")
            if not isinstance(reason, str) or not reason or tracks:
                raise ContractError("failure needs nonempty reason and zero tracks")
            failures[video_id] = reason
        elif status != "success":
            raise ContractError("unknown prediction video status")
        parsed = [_track(t, populations[video_id][0], True) for t in tracks]
        if len({t["id"] for t in parsed}) != len(parsed):
            raise ContractError("duplicate prediction ID within video")
        if setting == "supplied-track" and any("supplied_gt_id" not in t for t in tracks):
            raise ContractError("supplied track requires explicit identity mapping (null allowed)")
        predictions[video_id] = parsed
    rows, gt_events, dt_events, conditional_gt, conditional_dt = [], [], [], [], []
    period_ready = True
    for video_id in sorted(populations, key=_key):
        _, gt = populations[video_id]
        dt = predictions.get(video_id, [])
        if video_id not in predictions:
            failures[video_id] = "missing-prediction-video"
        gt_by_id = {g["id"]: g for g in gt}
        matches, used = {}, set()
        for pred in sorted(dt, key=lambda p: (-p["score"], _key(p["id"]))):
            if setting == "supplied-track":
                gid = pred["supplied_gt_id"]
                if gid is not None:
                    if (
                        gid not in gt_by_id
                        or gid in used
                        or pred["category"] != gt_by_id[gid]["category"]
                    ):
                        raise ContractError("invalid/duplicate/cross-category supplied identity")
                    matches[pred["id"]] = gid
                    used.add(gid)
            else:
                candidates = [
                    (tube_iou(pred["bboxes"], g["bboxes"]), g["id"])
                    for g in gt
                    if g["id"] not in used and g["category"] == pred["category"]
                ]
                candidates.sort(key=lambda r: (-r[0], _key(r[1])))
                if candidates and candidates[0][0] >= 0.5:
                    gid = candidates[0][1]
                    matches[pred["id"]] = gid
                    used.add(gid)
        inverse = {gid: pid for pid, gid in matches.items()}
        dt_by_id = {p["id"]: p for p in dt}
        raw, abs_errors, naes, obos, identity_obos = [], [], [], [], []
        for g in sorted(gt, key=lambda g: _key(g["id"])):
            pid = inverse.get(g["id"])
            count = dt_by_id[pid]["count"] if pid is not None else 0.0
            error = abs(count - g["count"])
            abs_errors.append(error)
            naes.append(error / g["count"])
            obos.append(int(error <= 1))
            identity_obos.append(int(pid is not None and error <= 1))
            raw.append(
                {
                    "gt_id": g["id"],
                    "prediction_id": pid,
                    "gt_count": g["count"],
                    "predicted_count": count,
                }
            )
            for eid, interval, _ in g["periods"]:
                event = ((video_id, g["id"]), eid, interval)
                gt_events.append(event)
                if pid is not None:
                    conditional_gt.append(event)
        extras = [p for p in dt if p["id"] not in matches]
        for p in dt:
            if p["periods"] is None:
                period_ready = False
                continue
            identity = (video_id, matches[p["id"]]) if p["id"] in matches else None
            for eid, interval, event_score in p["periods"]:
                event = (
                    identity,
                    (_key(video_id), _key(p["id"]), _key(eid)),
                    interval,
                    event_score,
                )
                dt_events.append(event)
                if identity is not None:
                    conditional_dt.append(event)
        rows.append(
            {
                "video_id": video_id,
                "n_gt": len(gt),
                "n_predictions": len(dt),
                "matched": len(matches),
                "missing_gt": len(gt) - len(matches),
                "extra_predictions": len(extras),
                "raw_counts": raw,
                "extra_raw_counts": [
                    {"prediction_id": p["id"], "count": p["count"]}
                    for p in sorted(extras, key=lambda p: _key(p["id"]))
                ],
                "failure_reason": failures.get(video_id),
                "AvgNAE_GT": sum(naes) / len(gt),
                "AvgOBO_GT": sum(obos) / len(gt),
                "FullCountNE": (sum(abs_errors) + sum(p["count"] for p in extras))
                / sum(g["count"] for g in gt),
                "FullIdentityOBO": sum(identity_obos) / (len(gt) + len(extras)),
            }
        )
    metric_names = ("AvgNAE_GT", "AvgOBO_GT", "FullCountNE", "FullIdentityOBO")
    unavailable = {
        "status": "unavailable-period-decoder",
        "AP": None,
        "mAP": None,
        "AP50": None,
        "AP75": None,
    }
    return {
        "contract": "FULL-v1",
        "contract_status": CONTRACT_STATUS,
        "setting": setting,
        "count_metrics": {name: sum(r[name] for r in rows) / len(rows) for name in metric_names},
        "population": {
            "videos": len(rows),
            "gt_tracks": sum(r["n_gt"] for r in rows),
            "predicted_tracks": sum(r["n_predictions"] for r in rows),
            "matched": sum(r["matched"] for r in rows),
            "missing_gt": sum(r["missing_gt"] for r in rows),
            "extra_predictions": sum(r["extra_predictions"] for r in rows),
            "failed_or_missing_videos": len(failures),
        },
        "videos": rows,
        "FullPeriodAP": _period_ap(gt_events, dt_events) if period_ready else dict(unavailable),
        "CONDITIONAL-v1-reference-diagnostic": _period_ap(conditional_gt, conditional_dt)
        if period_ready
        else dict(unavailable),
    }
