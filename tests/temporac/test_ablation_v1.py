from __future__ import annotations

import numpy as np

from pams.temporac.ablation_v1 import boxcar_overlap_add, run_ablation_suite


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.zeros((7, 3), dtype=np.float64)
    probabilities[:, 0] = 0.2
    probabilities[:, 1] = np.asarray((0.1, 0.8, 0.7, 0.1, 0.9, 0.8, 0.1))
    probabilities[:, 2] = 0.4
    gates = np.asarray(((0.1, 0.8, 0.1), (0.2, 0.7, 0.1)), dtype=np.float64)
    windows = np.asarray(((0, 5), (2, 7)), dtype=np.int32)
    mask = np.ones(7, dtype=np.uint8)
    bounds = np.asarray(((0, 7),), dtype=np.int32)
    return probabilities, gates, windows, mask, bounds


def test_boxcar_overlap_add_is_normalized_and_bounded() -> None:
    probabilities, gates, windows, mask, _ = _fixture()
    response = boxcar_overlap_add(probabilities, gates, windows, mask)
    assert response.shape == (7,)
    assert np.all((0.0 <= response) & (response <= 1.0))
    assert not response.flags.writeable


def test_suite_runs_five_same_checkpoint_variants() -> None:
    probabilities, gates, windows, mask, bounds = _fixture()
    outputs = run_ablation_suite(probabilities, gates, windows, mask, bounds)
    assert set(outputs) == {
        "full_local_soft_nola",
        "without_local_tempo_uniform",
        "without_soft_routing_hard",
        "single_medium_expert",
        "without_nola_taper_boxcar",
    }
    assert all(output.decode.decoder_invocations == 1 for output in outputs.values())


def test_global_variant_is_only_added_when_exact_gates_are_supplied() -> None:
    probabilities, gates, windows, mask, bounds = _fixture()
    global_gates = np.tile(np.asarray((0.6, 0.3, 0.1)), (len(windows), 1))
    outputs = run_ablation_suite(
        probabilities,
        gates,
        windows,
        mask,
        bounds,
        global_gates=global_gates,
    )
    assert "global_tempo_soft_nola" in outputs
