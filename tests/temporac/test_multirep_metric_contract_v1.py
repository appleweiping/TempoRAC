"""Artificial CPU fixtures only: no datasets, labels, models or network."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pams.temporac.multirep_metric_contract_v1 import (
    ContractError,
    inclusive_frame_span,
    map_boundaries,
    score,
    tube_iou,
)

SHA = "a" * 64


def gt(ident=19, count=1, box=(0, 0, 2, 2), periods=((0, 10),)):
    return dict(
        gt_id=ident,
        count=count,
        bboxes=[box] * 10,
        support=["observed"] * 10,
        periods=[dict(event_id=i, interval=p) for i, p in enumerate(periods)],
    )


def pred(ident="p", count=1, box=(0, 0, 2, 2), periods=((0, 10, 0.8),), **kwargs):
    return dict(
        prediction_id=ident,
        count=count,
        track_score=0.9,
        bboxes=[box] * 10,
        support=["observed"] * 10,
        decoder_sha=SHA,
        config_sha=SHA,
        checkpoint_sha=SHA,
        periods=None
        if periods is None
        else [dict(event_id=i, interval=p[:2], score=p[2]) for i, p in enumerate(periods)],
        **kwargs,
    )


def video(tracks, ident="v"):
    return dict(video_id=ident, num_frames=10, tracks=tracks)


def prediction(tracks, ident="v"):
    return dict(video_id=ident, status="success", tracks=tracks)


def counterexample(leading=False):
    gs = [video([gt(91), gt(7, box=(20, 0, 2, 2))])]
    ps = [pred()]
    if leading:
        ps.append(pred("false", box=(40, 0, 2, 2), periods=((0, 10, 0.99),)))
    return gs, [prediction(ps)]


def test_spine_counts_and_empty_prediction():
    result = score([video([gt(count=1), gt(700, count=10)])], [])
    assert result["count_metrics"] == dict(
        AvgNAE_GT=1, AvgOBO_GT=0.5, FullCountNE=1, FullIdentityOBO=0
    )
    assert result["population"]["missing_gt"] == 2
    assert result["population"]["failed_or_missing_videos"] == 1
    assert result["FullPeriodAP"]["AP50"] == 0
    assert result["CONDITIONAL-v1-reference-diagnostic"]["status"] == "undefined-no-gt-periods"


def test_full_ap_and_conditional_counterexamples():
    for leading, expected in [(False, 0.5), (True, 0.25)]:
        result = score(*counterexample(leading))
        assert result["FullPeriodAP"]["AP"] == [expected] * 10
        assert result["CONDITIONAL-v1-reference-diagnostic"]["AP"] == [1] * 10
        assert result["population"]["missing_gt"] == 1
        assert result["population"]["extra_predictions"] == int(leading)


def test_video_macro_is_not_person_pool_and_no_rounding():
    result = score(
        [video([gt()], "a"), video([gt(2), gt(3)], "b")],
        [prediction([pred(count=0.4, supplied_gt_id=19)], "a"), prediction([], "b")],
        setting="supplied-track",
    )
    assert result["count_metrics"]["AvgNAE_GT"] == pytest.approx(0.8)
    assert result["count_metrics"]["AvgNAE_GT"] != pytest.approx(2.6 / 3)
    assert result["videos"][0]["raw_counts"][0]["predicted_count"] == 0.4


def test_duplicates_extras_zero_counts_are_retained():
    tracks = [pred("first"), pred("duplicate", count=0)]
    result = score([video([gt()])], [prediction(tracks)])
    assert result["population"]["extra_predictions"] == 1
    assert result["count_metrics"]["FullIdentityOBO"] == 0.5
    assert result["FullPeriodAP"]["n_predictions"] == 2
    assert result["videos"][0]["extra_raw_counts"] == [dict(prediction_id="first", count=1)]
    # Equal track scores use prediction IDs; duplicate sorts first and count is 0.
    assert result["videos"][0]["raw_counts"][0]["predicted_count"] == 0
    assert result["count_metrics"]["FullCountNE"] == 2


def test_more_than_100_tracks_not_truncated():
    result = score([video([gt()])], [prediction([pred(str(i), count=1) for i in range(102)])])
    assert result["population"]["predicted_tracks"] == 102
    assert result["population"]["extra_predictions"] == 101
    assert result["count_metrics"]["FullCountNE"] == 101


def test_supplied_ids_and_tie_determinism():
    gs, ps = counterexample(True)
    r1 = score(gs, ps)
    gs[0]["tracks"].reverse()
    ps[0]["tracks"].reverse()
    assert score(gs, ps) == r1
    ps = [prediction([pred(supplied_gt_id=7)])]
    result = score(gs, ps, setting="supplied-track")
    assert result["videos"][0]["raw_counts"][0]["prediction_id"] == "p"
    assert result["FullPeriodAP"]["AP50"] == 0.5


def test_boundaries_inclusive_and_verified_stride_roundtrip():
    assert inclusive_frame_span(0, 9, 10) == (0, 10)
    mapped = map_boundaries([0, 1, 2, 3], [0, 1, 2, 3], [0, 3, 6, 10], 10)
    assert mapped == [0, 3, 6, 10]
    assert map_boundaries(mapped, [0, 1, 2, 3], [0, 3, 6, 10], 10, inverse=True) == [0, 1, 2, 3]
    result = score(
        [video([gt(periods=((0, 5), (5, 10)))])],
        [prediction([pred(periods=((5, 10, 0.9), (0, 5, 0.8)))])],
    )
    assert result["FullPeriodAP"]["mAP"] == 1
    with pytest.raises(ContractError):
        map_boundaries([1], [0, 1], [0, 9], 10)


def test_tube_missing_union_and_no_support_deletion():
    assert tube_iou([(0, 0, 1, 1), None], [(0, 0, 1, 1), (0, 0, 1, 1)]) == 0.5
    p = pred()
    p["bboxes"] = [None] * 10
    p["support"] = ["missing"] * 10
    result = score([video([gt()])], [prediction([p])])
    assert result["FullPeriodAP"]["n_gt"] == 1
    assert result["FullPeriodAP"]["AP50"] == 0


def test_period_unavailable_and_no_gt_periods():
    r = score([video([gt()])], [prediction([pred(periods=None)])])
    assert r["FullPeriodAP"]["status"] == "unavailable-period-decoder"
    r = score([video([gt(periods=())])], [])
    assert r["FullPeriodAP"]["mAP"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g, p: g[0]["tracks"][0].update(count=0),
        lambda g, p: p[0]["tracks"][0].update(count=float("nan")),
        lambda g, p: p[0]["tracks"][0].update(bboxes=[]),
        lambda g, p: p[0]["tracks"][0].update(support=[]),
        lambda g, p: p[0]["tracks"][0].update(prediction_id=True),
        lambda g, p: p[0]["tracks"][0].update(config_sha="bad"),
        lambda g, p: p[0]["tracks"][0]["periods"][0].update(interval=[0, 11]),
        lambda g, p: p[0]["tracks"][0]["periods"][0].update(interval=[5, 5]),
        lambda g, p: p[0].update(video_id="unknown"),
        lambda g, p: p.append(copy.deepcopy(p[0])),
        lambda g, p: g[0]["tracks"].append(copy.deepcopy(g[0]["tracks"][0])),
        lambda g, p: p[0]["tracks"].append(copy.deepcopy(p[0]["tracks"][0])),
        lambda g, p: p[0].update(status="failure", failure_reason="decoder failed"),
    ],
)
def test_malformed_fails_whole_call(mutate):
    gs, ps = [video([gt()])], [prediction([pred()])]
    mutate(gs, ps)
    with pytest.raises(ContractError):
        score(gs, ps)


def test_explicit_failure_and_supplied_mapping_errors():
    result = score(
        [video([gt()])], [dict(video_id="v", status="failure", failure_reason="crash", tracks=[])]
    )
    assert result["videos"][0]["failure_reason"] == "crash"
    for supplied in (None, 999):
        p = pred() if supplied is None else pred(supplied_gt_id=supplied)
        with pytest.raises(ContractError):
            score([video([gt()])], [prediction([p])], setting="supplied-track")


class MiniRow(dict):
    def __getitem__(self, key):
        if isinstance(key, list):
            return SimpleNamespace(values=np.array([dict.__getitem__(self, k) for k in key]))
        return dict.__getitem__(self, key)


class MiniFrame:
    """Only pandas operations reached by the unchanged official AP kernel."""

    def __init__(self, rows):
        self.rows = [MiniRow(r) for r in rows]
        self.loc = MiniLoc(self)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, list):
            return SimpleNamespace(values=np.array([[r[k] for k in key] for r in self.rows]))
        return SimpleNamespace(values=np.array([r[key] for r in self.rows]))

    def reset_index(self, drop=False):
        return MiniFrame([dict(r) if drop else dict(r, index=i) for i, r in enumerate(self.rows)])

    def iterrows(self):
        return enumerate(self.rows)

    def groupby(self, field):
        def get_group(value):
            rows = [dict(r, index=i) for i, r in enumerate(self.rows) if r[field] == value]
            if not rows:
                raise KeyError(value)
            return IndexedMiniFrame(rows)

        return SimpleNamespace(get_group=get_group)


class IndexedMiniFrame(MiniFrame):
    def reset_index(self, drop=False):
        return MiniFrame(self.rows)


class MiniLoc:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            return MiniFrame([self.frame.rows[int(i)] for i in key])
        return self.frame.rows[int(key)]


class CompatNumpy:
    """Test-only np.float alias; official function AST is not changed."""

    def __getattr__(self, name):
        return float if name == "float" else getattr(np, name)


def official_ap_kernel():
    path = Path(
        "E:/ICASSP_TempoRAC/aris-work/metric_sources/plus/mmdet/datasets/multirep_eval_api.py"
    )
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "50e9ede7792763d4f574b56528f728c9383a9b010456d51b720fdc5b33b455ab"
    )
    tree = ast.parse(raw.decode("utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiRepEval")
    names = {"compute_average_precision_detection", "segment_iou", "interpolated_prec_rec"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert [(m.name, m.lineno, m.end_lineno) for m in methods] == [
        ("compute_average_precision_detection", 566, 648),
        ("segment_iou", 650, 677),
        ("interpolated_prec_rec", 679, 686),
    ]
    # Compile original AST nodes with their original filename and line numbers.
    namespace = {"np": CompatNumpy()}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    klass = type("OriginalAPKernel", (), {name: namespace[name] for name in names})
    return klass()


def test_exact_original_official_ap_function_counterexamples():
    kernel = official_ap_kernel()

    def fields(ident, start, end, **kw):
        return {"video-id": ident, "t-start": start, "t-end": end, **kw}

    one_gt = MiniFrame([fields(91, 0, 10)])
    two_gt = MiniFrame([fields(91, 0, 10), fields(7, 0, 10)])
    tp = fields(91, 0, 10, score=0.8)
    leading_fp = fields("unmatched", 0, 10, score=0.99)
    for ground_truth, dt, expected in [
        (one_gt, [tp], 1),
        (two_gt, [tp], 0.5),
        (two_gt, [leading_fp, tp], 0.25),
    ]:
        original = kernel.compute_average_precision_detection(ground_truth, MiniFrame(dt))
        assert original.tolist() == [expected] * 10
    for leading, expected in [(False, 0.5), (True, 0.25)]:
        ours = score(*counterexample(leading))
        assert ours["FullPeriodAP"]["AP"] == [expected] * 10
        assert ours["CONDITIONAL-v1-reference-diagnostic"]["AP"] == [1] * 10


def test_extra_zero_count_still_penalizes_identity_and_events():
    p1 = pred("a", supplied_gt_id=19)
    p2 = pred("b", count=0, supplied_gt_id=None, box=(40, 0, 2, 2))
    result = score([video([gt()])], [prediction([p1, p2])], setting="supplied-track")
    assert result["count_metrics"]["FullCountNE"] == 0
    assert result["count_metrics"]["FullIdentityOBO"] == 0.5
    assert result["population"]["extra_predictions"] == 1
    assert result["videos"][0]["extra_raw_counts"] == [dict(prediction_id="b", count=0)]
    assert result["FullPeriodAP"]["n_predictions"] == 2


def test_equal_event_scores_are_stable_by_ids():
    gs, ps = counterexample(True)
    # false sorts before p at equal event score => high confidence FP first.
    ps[0]["tracks"][1]["periods"][0]["score"] = 0.8
    r1 = score(gs, ps)
    assert r1["FullPeriodAP"]["AP50"] == 0.25
    ps[0]["tracks"].reverse()
    assert score(gs, ps) == r1


def test_duplicate_temporal_events_false_positive_and_bad_ids():
    p = pred(periods=((0, 5, 0.99), (0, 5, 0.9), (5, 10, 0.8)))
    result = score([video([gt(periods=((0, 5), (5, 10)))])], [prediction([p])])
    assert result["FullPeriodAP"]["AP50"] == pytest.approx(5 / 6)
    p["periods"][1]["event_id"] = 0
    with pytest.raises(ContractError):
        score([video([gt()])], [prediction([p])])
    with pytest.raises(ContractError):
        score(
            [video([gt()])],
            [prediction([pred("x", supplied_gt_id=19), pred("y", supplied_gt_id=19)])],
            setting="supplied-track",
        )
