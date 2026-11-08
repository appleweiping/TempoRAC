"""Feature-level datasets, density targets and tempo statistics for RAC benchmarks.

Annotation convention (RepCount-A / UCFRep): ``bounds`` is the flat L1..L2n list of
cycle start/end frame indices in the annotation's frame coordinate.  Cycle ``k`` is
``[bounds[2k], bounds[2k+1]]``.  Targets live on the base grid of cell width
``BASE`` raw frames (the rate-1 feature grid); a cell ``g`` has centre frame
``BASE * g + (BASE - 1) / 2``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BASE = 2  # raw frames per base-grid cell (rate-1 token width)


@dataclass
class Video:
    key: str
    split: str
    video_id: str
    count: float
    cycles: np.ndarray  # (K, 2) in decoded-frame coordinates
    T: int  # decoded frames
    fps: float
    action: str
    feats: dict  # rate -> np.ndarray (G_s, P, C) float16 (memory resident)


def cycles_from_bounds(bounds, scale: float = 1.0) -> np.ndarray:
    b = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if b.size < 2:
        return np.zeros((0, 2))
    if b.size % 2 == 1:
        b = b[:-1]
    c = b.reshape(-1, 2) * scale
    return c


def density_target(cycles: np.ndarray, T: int, sigma_frac: float = 0.25, kind: str = "gauss") -> np.ndarray:
    """Per-cycle unit-mass density on the base grid (sum == number of cycles)."""
    G = math.ceil(T / BASE)
    d = np.zeros(G, dtype=np.float64)
    for a, b in cycles:
        a, b = float(min(a, b)), float(max(a, b))
        # Normalise each cycle over a grid that fully contains it, then keep only the
        # cells inside [0, G): a cycle cut by a temporal crop keeps its partial mass.
        lo = int(math.floor((a - 4 * max(sigma_frac * (b - a), 1.0)) / BASE)) - 1
        hi = int(math.ceil((b + 4 * max(sigma_frac * (b - a), 1.0)) / BASE)) + 1
        cells = np.arange(lo, hi + 1)
        centres = BASE * cells + (BASE - 1) / 2.0
        if kind == "gauss":
            mu = 0.5 * (a + b)
            sd = max(sigma_frac * (b - a), 1.0)
            w = np.exp(-0.5 * ((centres - mu) / sd) ** 2)
        elif kind == "box":
            w = ((centres >= a) & (centres <= b)).astype(np.float64)
        else:
            raise ValueError(kind)
        if w.sum() <= 0:  # degenerate cycle: unit mass at the nearest cell
            w = (cells == int(round(0.5 * (a + b) / BASE))).astype(np.float64)
        w = w / w.sum()
        keep = (cells >= 0) & (cells < G)
        np.add.at(d, cells[keep], w[keep])
    return d.astype(np.float32)


def period_target(cycles: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell period length (in base cells) and periodicity mask from annotated cycles."""
    G = math.ceil(T / BASE)
    centres = BASE * np.arange(G) + (BASE - 1) / 2.0
    per = np.zeros(G, dtype=np.float32)
    on = np.zeros(G, dtype=np.float32)
    for a, b in cycles:
        a, b = float(min(a, b)), float(max(a, b))
        m = (centres >= a) & (centres <= b)
        per[m] = max(b - a, 1.0) / BASE
        on[m] = 1.0
    return per, on


def tempo_cv(cycles: np.ndarray) -> float:
    if len(cycles) < 2:
        return 0.0
    p = np.abs(cycles[:, 1] - cycles[:, 0])
    p = p[p > 0]
    return float(p.std() / p.mean()) if len(p) > 1 else 0.0


def load_videos(manifest: list[dict], feat_root: str, rates: list[int], splits: set[str],
                pool_to: int | None = None) -> list[Video]:
    root = Path(feat_root)
    vids = []
    for it in manifest:
        if it["split"] not in splits:
            continue
        fkey = f"{it.get('fsplit', it['split'])}__{it['video_id']}"  # feature file key
        key = f"{it['split']}__{it['video_id']}"  # protocol role key
        meta = json.loads((root / "meta" / f"{fkey}.json").read_text())
        T = int(meta["frames_decoded"])
        n_ann = it.get("num_frames_csv")
        # Annotations index the original frame sequence; rescale only if the decoded
        # length differs from the annotated length (recorded, see audit).
        scale = 1.0
        if n_ann and abs(n_ann - T) > 1:
            scale = T / float(n_ann)
        feats = {}
        for s in rates:
            # memory-mapped: concurrent training processes share one OS page cache
            a = np.load(root / f"r{s}" / f"{fkey}.npy", mmap_mode="r")
            if pool_to is not None and pool_to != int(round(math.sqrt(a.shape[1]))):
                p = int(round(math.sqrt(a.shape[1])))
                t = torch.from_numpy(a.astype(np.float32)).view(a.shape[0], p, p, -1).permute(0, 3, 1, 2)
                t = torch.nn.functional.adaptive_avg_pool2d(t, pool_to).permute(0, 2, 3, 1)
                a = t.reshape(a.shape[0], pool_to * pool_to, -1).numpy().astype(np.float16)
            feats[s] = a
        count = it["count"] if it["count"] is not None else float("nan")
        vids.append(Video(key=key, split=it["split"], video_id=it["video_id"], count=float(count),
                          cycles=cycles_from_bounds(it.get("bounds", []), scale), T=T,
                          fps=float(meta.get("fps") or it.get("fps_csv") or 30.0),
                          action=it.get("action", ""), feats=feats))
    return vids


def interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Piecewise-linear map with linear extrapolation beyond the end knots."""
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def apply_warp(v: Video) -> Video:
    """Materialise a tempo warp drawn by train.warp_video (runs in collate workers).

    Every rate stream is resampled at the warped cell centres by linear interpolation
    between neighbouring cells; cycle bounds go through the same time map with linear
    extrapolation, so cycles that a crop left outside the clip stay outside and keep
    being truncated by the target builder (count is preserved inside the clip).
    """
    src, out, T2 = v.warp
    feats = {}
    for s, arr in v.feats.items():
        a = np.asarray(arr, dtype=np.float32)
        G2 = math.ceil(T2 / (BASE * s))
        half = (BASE * s - 1) / 2.0
        centres = BASE * s * np.arange(G2) + half
        u = np.clip((np.interp(centres, out, src) - half) / (BASE * s), 0, a.shape[0] - 1)
        i0 = np.floor(u).astype(int)
        i1 = np.minimum(i0 + 1, a.shape[0] - 1)
        w = (u - i0)[:, None, None]
        feats[s] = (a[i0] * (1 - w) + a[i1] * w).astype(np.float16)
    cyc = interp_extrap(v.cycles.reshape(-1), src, out).reshape(-1, 2) if len(v.cycles) else v.cycles
    import copy as _copy

    nv = _copy.copy(v)
    nv.feats, nv.T, nv.cycles, nv.warp = feats, T2, cyc, None
    return nv


def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class BatchSet(torch.utils.data.Dataset):
    """Pre-formed batches collated in DataLoader workers (CPU tensors, pinned)."""

    def __init__(self, batches, rates, sigma_frac, kind):
        self.batches, self.rates, self.sigma, self.kind = batches, rates, sigma_frac, kind

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return collate(self.batches[i], self.rates, self.sigma, self.kind, None)


def batch_loader(batches, rates, sigma_frac, kind, workers: int = 6):
    return torch.utils.data.DataLoader(BatchSet(batches, rates, sigma_frac, kind), batch_size=None, shuffle=False,
                                       num_workers=workers, pin_memory=True, prefetch_factor=2 if workers else None)


def collate(batch: list[Video], rates: list[int], sigma_frac: float, kind: str, device) -> dict:
    """Pad a list of videos to a batch; features stay at their native rate grids.

    ``device=None`` returns CPU tensors (for DataLoader workers)."""
    batch = [apply_warp(v) if getattr(v, "warp", None) is not None else v for v in batch]
    B = len(batch)
    G1 = [math.ceil(v.T / BASE) for v in batch]
    Gmax = max(G1)
    out = {"feats": {}, "masks": {}}
    for s in rates:
        Gs = [v.feats[s].shape[0] for v in batch]
        Lmax = math.ceil(Gmax / s)
        P, C = batch[0].feats[s].shape[1:]
        x = torch.zeros(B, Lmax, P, C, dtype=torch.float16)
        m = torch.zeros(B, Lmax, dtype=torch.bool)
        for i, v in enumerate(batch):
            g = min(Gs[i], Lmax)
            x[i, :g] = torch.from_numpy(np.array(v.feats[s][:g]))
            m[i, :g] = True
        out["feats"][s] = x
        out["masks"][s] = m
    dens = torch.zeros(B, Gmax)
    per = torch.zeros(B, Gmax)
    on = torch.zeros(B, Gmax)
    mask = torch.zeros(B, Gmax, dtype=torch.bool)
    for i, v in enumerate(batch):
        d = density_target(v.cycles, v.T, sigma_frac, kind)
        p, o = period_target(v.cycles, v.T)
        dens[i, : len(d)] = torch.from_numpy(d)
        per[i, : len(p)] = torch.from_numpy(p)
        on[i, : len(o)] = torch.from_numpy(o)
        mask[i, : G1[i]] = True
    out.update({
        "density": dens, "period": per, "periodic": on, "mask": mask,
        "count": torch.tensor([v.count for v in batch], dtype=torch.float32),
        "keys": [v.key for v in batch],
    })
    return out if device is None else to_device(out, device)
