"""Counting metrics.

``mae`` follows the community RepCount/UCFRep convention used by the official
TransRAC and ESCounts evaluation code: ``|round(pred) - gt| / (gt + 0.1)``.
``mae_pos`` is the literal paper formula ``|round(pred) - gt| / gt`` on gt > 0 only.
``obo``: ``|round(pred) - gt| <= 1``; ``obz``: exact match.  All videos are kept.
"""
from __future__ import annotations

import numpy as np


def count_metrics(pred, gt) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = ~np.isnan(gt)
    p, g = np.round(pred[ok]), gt[ok]
    err = np.abs(p - g)
    pos = g > 0
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(err / (g + 0.1))),
        "obo": float(np.mean(err <= 1)),
        "obz": float(np.mean(err == 0)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "mae_pos": float(np.mean(err[pos] / g[pos])) if pos.any() else float("nan"),
        "n_pos": int(pos.sum()),
        "mae_raw_cont": float(np.mean(np.abs(pred[ok] - g) / (g + 0.1))),
    }


def bootstrap_ci(values: np.ndarray, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
