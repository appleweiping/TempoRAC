"""Counting heads on frozen multi-rate video tokens.

All heads output a non-negative density on the base grid (cell = 2 raw frames);
the count is the density sum.  Rate-``s`` features live on a grid whose cells are
``s`` base cells wide; ``to_base`` repeats them (and divides densities by ``s`` so
mass is preserved).

Heads
-----
* ``single``   - one rate, spatial pooling + temporal encoder (exemplar-free
                 ESCounts-style decoder baseline).
* ``concat``   - all rates upsampled and concatenated (multi-rate early fusion,
                 no routing).
* ``tsm``      - TransRAC-style multi-scale temporal self-similarity head
                 (banded TSM per rate, conv over lags, temporal encoder).
* ``repnet``   - RepNet-style head: banded TSM -> per-cell period-length
                 classification + periodicity; count = sum(periodicity / period).
* ``temporac`` - TempoRAC: per-rate tempo experts, local masked tempogram,
                 tempo-conditioned soft routing, fused density decoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ----------------------------------------------------------------------------- utils
def to_base(x: torch.Tensor, s: int, G: int, density: bool = False) -> torch.Tensor:
    """(B, L, ...) on a rate-s grid -> (B, G, ...) on the base grid."""
    if s != 1:
        x = torch.repeat_interleave(x, s, dim=1)
        if density:
            x = x / s
    if x.shape[1] < G:
        pad = [0] * (2 * (x.dim() - 2)) + [0, G - x.shape[1]]
        x = F.pad(x, pad)
    return x[:, :G]


def sinusoid(L: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)[None]
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class SpatialPool(nn.Module):
    """(B, L, P, Cin) -> (B, L, D) with attention pooling over the P spatial tokens."""

    def __init__(self, cin: int, d: int, mode: str = "attn"):
        super().__init__()
        self.norm = nn.LayerNorm(cin)
        self.proj = nn.Linear(cin, d)
        self.mode = mode
        if mode == "attn":
            self.score = nn.Sequential(nn.Linear(d, d // 4), nn.Tanh(), nn.Linear(d // 4, 1))

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(x))
        if self.mode == "attn":
            w = torch.softmax(self.score(h), dim=2)
            return (w * h).sum(2)
        return h.mean(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:  # recompute the (B, L, P, C) activations in backward to save memory
            return torch.utils.checkpoint.checkpoint(self._pool, x, use_reentrant=False)
        return self._pool(x)


class EncoderLayer(nn.Module):
    """Pre-norm block: depthwise temporal conv (local) + self-attention + MLP."""

    def __init__(self, d: int, heads: int, conv_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n0 = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, conv_k, padding=conv_k // 2, groups=d)
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        x = x + self.drop(self.dw((self.n0(x) * m).transpose(1, 2)).transpose(1, 2))
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.n2(x)))
        return x * m


class TemporalEncoder(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, conv_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d, heads, conv_k, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_scale: float = 1.0) -> torch.Tensor:
        # positions are expressed in base cells so that rates share one time axis
        L = x.shape[1]
        pe = sinusoid(int(math.ceil(L * pos_scale)) + 1, x.shape[-1], x.device)
        idx = (torch.arange(L, device=x.device).float() * pos_scale).long()
        x = x + pe[idx][None]
        for lyr in self.layers:
            x = lyr(x, mask)
        return self.norm(x)


class DensityHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.net[-1].bias, -4.0)  # softplus(-4) ~ 0.018 per cell at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x).squeeze(-1))


def _band_index(G: int, lags: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(G, device=device)[:, None] + lags[None, :]  # G x L
    inr = (idx >= 0) & (idx < G)
    return idx.clamp(0, G - 1), inr


def band_of(S: torch.Tensor, mask: torch.Tensor, lags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather S[b, t, t + lag] for every lag from a full (B, G, G) matrix."""
    B, G, _ = S.shape
    idx, inr = _band_index(G, lags, S.device)
    band = torch.gather(S, 2, idx[None].expand(B, -1, -1))
    m = mask.float()
    valid = inr[None].float() * m[:, :, None] * torch.gather(m, 1, idx.reshape(1, -1).expand(B, -1)).view(B, G, -1)
    return band * valid, valid


def banded_similarity(f: torch.Tensor, mask: torch.Tensor, max_lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity of each cell to cells at lags +-1..max_lag: (B, G, 2*max_lag).

    Computed as one batched Gram matrix plus a gather (no per-lag copies kept for
    autograd).  Invalid (padded / out-of-range) pairs are 0; validity is returned.
    """
    f = F.normalize(f.float(), dim=-1)
    S = torch.bmm(f, f.transpose(1, 2))
    lags = torch.cat([torch.arange(-max_lag, 0), torch.arange(1, max_lag + 1)]).to(f.device)
    return band_of(S, mask, lags)


# ----------------------------------------------------------------------------- tempogram
class LocalTempogram(nn.Module):
    """Masked local autocorrelation of features: A(t, tau) for tau = 1..max_lag.

    A(t, tau) = sum_u w(u - t) q(u) q(u + tau) <f_u, f_{u+tau}> / sum_u w(u - t) q(u) q(u + tau)
    with a Hann window w of half-width ``win`` cells.  Parameter-free; optionally
    detached so that routing reads tempo evidence rather than shaping it.
    """

    def __init__(self, max_lag: int = 160, win: int = 48, detach: bool = True):
        super().__init__()
        self.max_lag, self.win, self.detach = max_lag, win, detach
        w = torch.hann_window(2 * win + 3, periodic=False)[1:-1]
        self.register_buffer("w", (w / w.sum()).view(1, 1, -1), persistent=False)

    def forward(self, f: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.detach:
            f = f.detach()
        f = F.normalize(f.float(), dim=-1)
        B, G, D = f.shape
        S = torch.bmm(f, f.transpose(1, 2))
        lags = torch.arange(1, self.max_lag + 1, device=f.device)
        prod, q = band_of(S, mask, lags)  # B G L: q(u) q(u+tau) <f_u, f_{u+tau}>
        num = prod.float().transpose(1, 2).contiguous()  # B L G
        den = q.float().transpose(1, 2).contiguous()
        k = self.w.shape[-1]
        num = F.conv1d(num.reshape(B * self.max_lag, 1, G), self.w, padding=k // 2).view(B, self.max_lag, G)
        den = F.conv1d(den.reshape(B * self.max_lag, 1, G), self.w, padding=k // 2).view(B, self.max_lag, G)
        A = num / den.clamp_min(1e-4)
        support = (den > 1e-3).float()
        return (A * support).transpose(1, 2), support.transpose(1, 2)  # B G L


# ----------------------------------------------------------------------------- heads
class SingleRateHead(nn.Module):
    def __init__(self, cfg: dict, rate: int):
        super().__init__()
        d = cfg["d"]
        self.rate = rate
        self.pool = SpatialPool(768, d, cfg.get("spatial", "attn"))
        self.enc = TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
        self.head = DensityHead(d)

    def forward(self, batch: dict) -> dict:
        s = self.rate
        x = self.pool(batch["feats"][s])
        m = batch["masks"][s]
        h = self.enc(x, m, pos_scale=float(s))
        dens = self.head(h) * m.float()
        G = batch["mask"].shape[1]
        return {"density": to_base(dens, s, G, density=True) * batch["mask"].float()}


class ConcatHead(nn.Module):
    def __init__(self, cfg: dict, rates: list[int]):
        super().__init__()
        d = cfg["d"]
        self.rates = rates
        self.pools = nn.ModuleDict({str(s): SpatialPool(768, d, cfg.get("spatial", "attn")) for s in rates})
        self.fuse = nn.Linear(d * len(rates), d)
        self.enc = TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
        self.head = DensityHead(d)

    def forward(self, batch: dict) -> dict:
        G = batch["mask"].shape[1]
        xs = [to_base(self.pools[str(s)](batch["feats"][s]), s, G) for s in self.rates]
        h = self.enc(self.fuse(torch.cat(xs, -1)), batch["mask"])
        return {"density": self.head(h) * batch["mask"].float()}


class TSMHead(nn.Module):
    """TransRAC-style: multi-scale (per-rate) multi-head banded self-similarity."""

    def __init__(self, cfg: dict, rates: list[int]):
        super().__init__()
        d = cfg["d"]
        self.rates = rates
        self.max_lag = cfg.get("tsm_lag", 96)
        self.nh = cfg.get("tsm_heads", 4)
        self.pools = nn.ModuleDict({str(s): SpatialPool(768, d, cfg.get("spatial", "attn")) for s in rates})
        self.qk = nn.ModuleDict({str(s): nn.Linear(d, d) for s in rates})
        c_in = len(rates) * self.nh
        self.lag_conv = nn.Sequential(nn.Conv1d(c_in, 32, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(32, 32, 5, padding=2), nn.GELU())
        self.proj = nn.Linear(32 * (2 * self.max_lag), d)
        self.enc = TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
        self.head = DensityHead(d)

    def forward(self, batch: dict) -> dict:
        G = batch["mask"].shape[1]
        mask = batch["mask"]
        chans = []
        for s in self.rates:
            x = to_base(self.pools[str(s)](batch["feats"][s]), s, G)
            q = self.qk[str(s)](x)
            B, _, D = q.shape
            q = q.view(B, G, self.nh, D // self.nh)
            for h in range(self.nh):
                sim, _ = banded_similarity(q[:, :, h], mask, self.max_lag)
                chans.append(sim)
        t = torch.stack(chans, 2)  # B G C 2L
        B = t.shape[0]
        t = self.lag_conv(t.reshape(B * G, t.shape[2], t.shape[3]))
        h = self.proj(t.reshape(B, G, -1))
        h = self.enc(h, mask)
        return {"density": self.head(h) * mask.float()}


class RepNetHead(nn.Module):
    """RepNet-style period-length + periodicity prediction from a banded TSM."""

    def __init__(self, cfg: dict, rate: int = 1):
        super().__init__()
        d = cfg["d"]
        self.rate = rate
        self.max_lag = cfg.get("tsm_lag", 96)
        self.n_per = cfg.get("n_period_bins", 64)
        self.pool = SpatialPool(768, d, cfg.get("spatial", "attn"))
        self.lag_conv = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.GELU())
        self.proj = nn.Linear(32 * 2 * self.max_lag, d)
        self.enc = TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
        self.per_head = nn.Linear(d, self.n_per)
        self.on_head = nn.Linear(d, 1)
        # log-spaced period bins in base cells (2 .. 2*max_lag cells)
        self.register_buffer("bins", torch.exp(torch.linspace(math.log(2.0), math.log(2.0 * self.max_lag), self.n_per)),
                             persistent=False)

    def forward(self, batch: dict) -> dict:
        G = batch["mask"].shape[1]
        mask = batch["mask"]
        s = self.rate
        x = to_base(self.pool(batch["feats"][s]), s, G)
        sim, _ = banded_similarity(x, mask, self.max_lag)
        B = sim.shape[0]
        t = self.lag_conv(sim.reshape(B * G, 1, -1))
        h = self.enc(self.proj(t.reshape(B, G, -1)), mask)
        logits = self.per_head(h)
        on = torch.sigmoid(self.on_head(h).squeeze(-1))
        per = (torch.softmax(logits, -1) * self.bins).sum(-1)
        dens = on / per.clamp_min(1.0) * mask.float()
        return {"density": dens, "period_logits": logits, "periodic_prob": on}


class TempoRACHead(nn.Module):
    """TempoRAC: tempo-routed mixture of multi-rate temporal experts.

    (A) LocalTempogram: masked local autocorrelation of features over lags.
    (B) Rate experts: one temporal encoder per sampling rate, each operating on
        its own grid so a fixed receptive field covers different absolute periods.
    (C) Tempo-conditioned soft routing: per-cell simplex weights over experts from
        the tempogram; mixed response decoded once into a density.
    Ablation switches: router in {soft, hard, uniform, feat}; tempogram on/off.
    """

    def __init__(self, cfg: dict, rates: list[int]):
        super().__init__()
        d = cfg["d"]
        self.rates = rates
        self.router_mode = cfg.get("router", "soft")
        self.use_tempo = cfg.get("use_tempogram", True)
        self.tau = cfg.get("router_temp", 1.0)
        self.pools = nn.ModuleDict({str(s): SpatialPool(768, d, cfg.get("spatial", "attn")) for s in rates})
        share = cfg.get("share_experts", False)
        if share:
            enc = TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
            self.experts = nn.ModuleDict({str(s): enc for s in rates})
        else:
            self.experts = nn.ModuleDict({str(s): TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5),
                                                                  cfg["dropout"]) for s in rates})
        self.tempo = LocalTempogram(cfg.get("max_lag", 160), cfg.get("tempo_win", 48), cfg.get("detach_tempo", True))
        L = cfg.get("max_lag", 160)
        self.tempo_proj = nn.Sequential(nn.Linear(2 * L, d), nn.GELU(), nn.Linear(d, d))
        r_in = d * (2 if self.use_tempo else 1)
        self.router = nn.Sequential(nn.LayerNorm(r_in), nn.Linear(r_in, d), nn.GELU(), nn.Linear(d, len(rates)))
        self.fuse = TemporalEncoder(d, cfg.get("fuse_layers", 1), cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
        self.head = DensityHead(d)

    def forward(self, batch: dict) -> dict:
        G = batch["mask"].shape[1]
        mask = batch["mask"]
        ys, xs = [], []
        for s in self.rates:
            x = self.pools[str(s)](batch["feats"][s])
            m = batch["masks"][s]
            y = self.experts[str(s)](x, m, pos_scale=float(s))
            ys.append(to_base(y, s, G))
            xs.append(to_base(x, s, G))
        Y = torch.stack(ys, 2)  # B G S D
        cue_feat = xs[0]  # finest-rate features carry the local tempo evidence
        A, sup = self.tempo(cue_feat, mask)
        tfeat = self.tempo_proj(torch.cat([A, sup], -1))
        ctx = Y.mean(2)
        r_in = torch.cat([ctx, tfeat], -1) if self.use_tempo else ctx
        logits = self.router(r_in)
        if self.router_mode == "uniform":
            g = torch.full_like(logits, 1.0 / len(self.rates))
        elif self.router_mode == "hard":
            soft = torch.softmax(logits / self.tau, -1)
            hard = F.one_hot(soft.argmax(-1), len(self.rates)).float()
            g = hard + soft - soft.detach()
        else:  # "soft" and "feat" (feat = soft routing without tempogram, set use_tempogram False)
            g = torch.softmax(logits / self.tau, -1)
        z = (g.unsqueeze(-1) * Y).sum(2)
        h = self.fuse(z, mask)
        dens = self.head(h) * mask.float()
        return {"density": dens, "route": g, "tempogram": A}


def build_model(cfg: dict) -> nn.Module:
    kind = cfg["head"]
    rates = cfg["rates"]
    if kind == "single":
        return SingleRateHead(cfg, rates[0])
    if kind == "concat":
        return ConcatHead(cfg, rates)
    if kind == "tsm":
        return TSMHead(cfg, rates)
    if kind == "repnet":
        return RepNetHead(cfg, rates[0])
    if kind == "temporac":
        return TempoRACHead(cfg, rates)
    raise ValueError(kind)
