"""Train-selected conservative return-RMS fallback for complete chains.

This additive copy preserves the v3 algorithm and changes only the natural
return threshold from 0.075 to the train-only selected value 0.085.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_PERIOD = 6
MAX_PERIOD = 160
MAX_RETURN_RMS = 0.085


@dataclass(frozen=True, slots=True)
class FullChainLandmarkResultV4:
    landmarks: NDArray[np.int32]
    base_landmarks: int
    retained_event_fraction: float
    estimated_period: int | None


def _lag_scores(run: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = min(MAX_PERIOD, (run.shape[0] - 1) // 2)
    scores = np.full(maximum + 1, np.inf, dtype=np.float64)
    for lag in range(MIN_PERIOD, maximum + 1):
        difference = run[lag:] - run[:-lag]
        per_row = np.linalg.norm(difference, axis=1) / np.sqrt(run.shape[1])
        scores[lag] = float(np.median(per_row))
    return scores


def _return_chain(
    run: NDArray[np.float64],
    period: int,
) -> tuple[list[int], float, float] | None:
    best: tuple[tuple[float, float, int], list[int], float, float] | None = None
    radius = max(2, int(round(0.20 * period)))
    for reference in range(min(period, run.shape[0])):
        points = [reference]
        errors = [0.0]
        expected = reference + period
        while expected < run.shape[0]:
            left = max(points[-1] + max(3, int(round(0.55 * period))), expected - radius)
            right = min(run.shape[0], expected + radius + 1)
            if left >= right:
                break
            local = run[left:right] - run[reference]
            distance = np.linalg.norm(local, axis=1) / np.sqrt(run.shape[1])
            offset = int(np.argmin(distance))
            error = float(distance[offset])
            if error > MAX_RETURN_RMS:
                break
            point = left + offset
            points.append(point)
            errors.append(error)
            expected = point + period
        if len(points) < 3:
            continue
        intervals = np.diff(np.asarray(points, dtype=np.float64))
        interval_cv = float(np.std(intervals) / np.mean(intervals))
        maximum = max(errors)
        key = (-float(len(points)), maximum, reference)
        if best is None or key < best[0]:
            best = (key, points, maximum, interval_cv)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _origin_subset(
    geometry: NDArray[np.float64],
    points: NDArray[np.int32],
) -> NDArray[np.int32]:
    values = geometry[points]
    distance = np.linalg.norm(values[:, None] - values[None, :], axis=2) / np.sqrt(66.0)
    reference = min(
        range(points.size),
        key=lambda index: (
            -int(np.count_nonzero(distance[index] <= 0.05)),
            float(np.max(distance[index][distance[index] <= 0.05])),
            int(points[index]),
        ),
    )
    selected = np.array(points[distance[reference] <= 0.05], dtype=np.int32, copy=True)
    while selected.size >= 3:
        starts = geometry[selected[:-1]]
        median = np.median(starts, axis=0)
        rms = np.linalg.norm(starts - median, axis=1) / np.sqrt(66.0)
        if np.all(rms <= 0.05):
            break
        selected = np.delete(selected, int(np.argmax(rms)))
    return selected


def _valid_edge(geometry: NDArray[np.float64], left: int, right: int) -> bool:
    arc = np.linalg.norm(geometry[left + 1 : right + 1] - geometry[left:right], axis=1)
    return bool(
        arc.size >= 6
        and np.count_nonzero(arc > 0.0) >= 5
        and float(np.max(arc) / np.sum(arc)) < 0.25
    )


def _maximum_legal_subsequence(
    geometry: NDArray[np.float64],
    points: NDArray[np.int32],
) -> NDArray[np.int32]:
    if points.size < 3:
        return np.empty(0, dtype=np.int32)
    length = np.ones(points.size, dtype=np.int32)
    span = np.zeros(points.size, dtype=np.int32)
    predecessor = np.full(points.size, -1, dtype=np.int32)
    for right_index in range(points.size):
        for left_index in range(right_index):
            if not _valid_edge(geometry, int(points[left_index]), int(points[right_index])):
                continue
            candidate_length = int(length[left_index]) + 1
            candidate_span = int(points[right_index] - points[left_index]) + int(
                span[left_index]
            )
            if candidate_length > int(length[right_index]) or (
                candidate_length == int(length[right_index])
                and candidate_span > int(span[right_index])
            ):
                length[right_index] = candidate_length
                span[right_index] = candidate_span
                predecessor[right_index] = left_index
    endpoint = max(
        range(points.size),
        key=lambda index: (int(length[index]), int(span[index]), -int(points[index])),
    )
    if int(length[endpoint]) < 3:
        return np.empty(0, dtype=np.int32)
    indices: list[int] = []
    current = endpoint
    while current >= 0:
        indices.append(current)
        current = int(predecessor[current])
    indices.reverse()
    return np.asarray(points[indices], dtype=np.int32)


def full_chain_integer_landmarks_v4(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> FullChainLandmarkResultV4:
    """Choose the longest final legal chain across all valid lag minima."""

    values = np.asarray(geometry, dtype=np.float64)
    bounds = np.asarray(run_bounds, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 66 or not np.isfinite(values).all():
        raise ValueError("geometry must be finite [T,66]")
    if bounds.shape != (1, 2):
        raise ValueError("periodic candidate requires one [1,2] run")
    start, stop = (int(value) for value in bounds[0])
    if not 0 <= start < stop <= values.shape[0]:
        raise ValueError("run bound is out of range")

    run = values[start:stop]
    scores = _lag_scores(run)
    best: tuple[
        tuple[int, float, float, float, int, int],
        NDArray[np.int32],
        int,
        int,
    ] | None = None
    for period in range(MIN_PERIOD, scores.size):
        score = float(scores[period])
        if not (
            score <= MAX_RETURN_RMS
            and score <= float(scores[period - 1])
            and (period + 1 == scores.size or score < float(scores[period + 1]))
        ):
            continue
        chain = _return_chain(run, period)
        if chain is None:
            continue
        points, maximum, interval_cv = chain
        raw = np.asarray([start + point for point in points], dtype=np.int32)
        origin = _origin_subset(values, raw)
        legal = _maximum_legal_subsequence(values, origin)
        if legal.size < 3:
            continue
        key = (
            -int(legal.size),
            score,
            interval_cv,
            maximum,
            period,
            int(legal[0]),
        )
        if best is None or key < best[0]:
            best = (key, legal, int(raw.size), period)

    if best is None:
        empty = np.empty(0, dtype=np.int32)
        empty.flags.writeable = False
        return FullChainLandmarkResultV4(empty, 0, 0.0, None)
    _key, selected, base_landmarks, period = best
    selected.flags.writeable = False
    original_events = max(0, base_landmarks - 1)
    retained_events = max(0, int(selected.size) - 1)
    return FullChainLandmarkResultV4(
        landmarks=selected,
        base_landmarks=base_landmarks,
        retained_event_fraction=(retained_events / original_events if original_events else 0.0),
        estimated_period=period,
    )


__all__ = ["FullChainLandmarkResultV4", "full_chain_integer_landmarks_v4"]
