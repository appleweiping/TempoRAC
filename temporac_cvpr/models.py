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

    def __init__(self, cin: int, d: int, mode: str = "attn", periodic: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(cin)
        self.proj = nn.Linear(cin, d)
        self.mode = mode
        if mode == "attn":
            self.score = nn.Sequential(nn.Linear(d, d // 4), nn.Tanh(), nn.Linear(d // 4, 1))
        if periodic:  # strength of the periodicity saliency; 0 = plain attention pooling
            self.beta = nn.Parameter(torch.zeros(()))

    def _pool(self, x: torch.Tensor, sal: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(self.norm(x))
        if self.mode == "attn":
            logit = self.score(h)
            if sal is not None:  # periodicity saliency (B, L, P) with learnable strength beta
                logit = logit + self.beta * sal.unsqueeze(-1).to(logit.dtype)
            w = torch.softmax(logit, dim=2)
            return (w * h).sum(2)
        return h.mean(2)

    def forward(self, x: torch.Tensor, sal: torch.Tensor | None = None) -> torch.Tensor:
        if self.training:  # recompute the (B, L, P, C) activations in backward to save memory
            return torch.utils.checkpoint.checkpoint(self._pool, x, sal, use_reentrant=False)
        return self._pool(x, sal)


def periodic_saliency(x: torch.Tensor, mask: torch.Tensor, max_lag: int, win: int, min_lag: int = 2) -> torch.Tensor:
    """Per-spatial-token local periodicity prominence on one rate grid (parameter-free).

    x: (B, L, P, C) frozen tokens; returns (B, L, P): for each position, the peak of
    its local masked autocorrelation over lags [min_lag, max_lag] minus its mean over
    lags, centred across positions (relative saliency).  Gradient-free.
    """
    with torch.no_grad():
        B, L, P, C = x.shape
        z = F.normalize(x.float(), dim=-1).permute(0, 2, 1, 3).reshape(B * P, L, C)
        m = mask.unsqueeze(1).expand(B, P, L).reshape(B * P, L)
        S = torch.bmm(z, z.transpose(1, 2))
        lags = torch.arange(1, max_lag + 1, device=x.device)
        prod, q = band_of(S, m, lags)  # (BP, L, K)
        k = 2 * win + 1
        wdw = torch.hann_window(k + 2, periodic=False, device=x.device)[1:-1]
        wdw = (wdw / wdw.sum()).view(1, 1, -1)
        K = prod.shape[-1]
        num = F.conv1d(prod.transpose(1, 2).reshape(-1, 1, L), wdw, padding=win).view(B * P, K, L)
        den = F.conv1d(q.transpose(1, 2).reshape(-1, 1, L), wdw, padding=win).view(B * P, K, L)
        A = (num / den.clamp_min(1e-4)) * (den > 1e-3)
        A = A[:, min_lag - 1:]  # ignore the trivially high smallest lags
        prom = A.max(1).values - A.mean(1)  # (BP, L)
        prom = prom.view(B, P, L).permute(0, 2, 1)  # B L P
        prom = prom - prom.mean(-1, keepdim=True)
        return prom * mask.unsqueeze(-1).float()


class EncoderLayer(nn.Module):
    """Pre-norm block: depthwise temporal conv (local) + self-attention + MLP."""

    def __init__(self, d: int, heads: int, conv_k: int = 5, dropout: float = 0.1, tg_bias: bool = False):
        super().__init__()
        self.n0 = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, conv_k, padding=conv_k // 2, groups=d)
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.heads = heads
        if tg_bias:  # per-head strength of the tempogram attention bias (0 = plain attention)
            self.alpha = nn.Parameter(torch.zeros(heads))

    def _biased_attn(self, h: torch.Tensor, mask: torch.Tensor, tbias: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        H = self.heads
        q, k, v = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, -1)
        q, k, v = (t.reshape(B, L, H, D // H).transpose(1, 2) for t in (q, k, v))
        if tbias.dim() == 4:  # already per-head (learned relative-position table control)
            add = tbias.expand(B, -1, -1, -1)
        else:
            add = self.alpha.view(1, H, 1, 1) * tbias.unsqueeze(1)  # B H L L
        add = add.masked_fill(~mask.view(B, 1, 1, L), -1e4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=add.to(q.dtype),
                                           dropout_p=self.attn.dropout if self.training else 0.0)
        return self.attn.out_proj(o.transpose(1, 2).reshape(B, L, D))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, tbias: torch.Tensor | None = None) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        x = x + self.drop(self.dw((self.n0(x) * m).transpose(1, 2)).transpose(1, 2))
        h = self.n1(x)
        if tbias is not None and hasattr(self, "alpha"):
            a = self._biased_attn(h, mask, tbias)
        else:
            a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.n2(x)))
        return x * m


class TemporalEncoder(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, conv_k: int = 5, dropout: float = 0.1,
                 tg_bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d, heads, conv_k, dropout, tg_bias) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_scale: float = 1.0,
                tbias: torch.Tensor | None = None) -> torch.Tensor:
        # positions are expressed in base cells so that rates share one time axis
        L = x.shape[1]
        pe = sinusoid(int(math.ceil(L * pos_scale)) + 1, x.shape[-1], x.device)
        idx = (torch.arange(L, device=x.device).float() * pos_scale).long()
        x = x + pe[idx][None]
        for lyr in self.layers:
            x = lyr(x, mask, tbias)
        return self.norm(x)


def similarity_bias(f: torch.Tensor, mask: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Control for the tempogram bias: raw (un-windowed) cosine self-similarity of the
    finest stream, z-scored per row over lags 1..max_lag, 0 beyond max_lag."""
    with torch.no_grad():
        f = F.normalize(f.detach().float(), dim=-1)
        B, G, _ = f.shape
        S = torch.bmm(f, f.transpose(1, 2))
        pos = torch.arange(G, device=f.device)
        lag = (pos[None, :] - pos[:, None]).abs()
        band = ((lag >= 1) & (lag <= max_lag)).float()[None] * (mask[:, :, None] & mask[:, None, :]).float()
        n = band.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (S * band).sum(-1, keepdim=True) / n
        sd = (((S - mu) ** 2) * band).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-3)
        Z = (S - mu) / sd * band
        return 0.5 * (Z + Z.transpose(1, 2))


def tempogram_bias(A: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """Turn a local tempogram A(t, tau), tau = 1..L, into a (B, G, G) attention bias.

    Each row is z-scored over its supported lags so that the bias highlights lags at
    which the signal repeats (period peaks) rather than generic temporal smoothness;
    bias(t, t') = 0.5 [Az(t, |t'-t|) + Az(t', |t'-t|)], and 0 for |t'-t| = 0 or > L.
    """
    B, G, L = A.shape
    w = sup.float()
    n = w.sum(-1, keepdim=True).clamp_min(1.0)
    mu = (A * w).sum(-1, keepdim=True) / n
    sd = (((A - mu) ** 2) * w).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-3)
    Az = (A - mu) / sd * w
    Apad = torch.cat([A.new_zeros(B, G, 1), Az, A.new_zeros(B, G, 1)], -1)  # lag 0, 1..L, >L
    pos = torch.arange(G, device=A.device)
    lag = (pos[None, :] - pos[:, None]).abs().clamp(max=L + 1)  # G x G
    bias = torch.gather(Apad, 2, lag[None].expand(B, -1, -1))
    return 0.5 * (bias + bias.transpose(1, 2))


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


class TempoGate(nn.Module):
    """Tempo-gated pyramid fusion.

    From the local tempogram it derives a soft dominant period (softmax over z-scored
    lags >= 2, averaged in log-lag space), the peak prominence and the lag entropy,
    and maps these three tempo scalars -- and nothing else -- to one gate per rate,
    g_s(t) in (0, 2) (initialised near 1).  Because the gate sees only tempo, it can
    express "fine rates for fast moments, coarse rates for slow ones" but cannot
    memorise content.
    """

    def __init__(self, n_rates: int, max_lag: int, temp: float = 0.5, hidden: int = 16):
        super().__init__()
        self.temp = temp
        self.register_buffer("loglag", torch.log(torch.arange(1, max_lag + 1).float()), persistent=False)
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, n_rates))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, A: torch.Tensor, sup: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = sup.float()
        n = w.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (A * w).sum(-1, keepdim=True) / n
        sd = (((A - mu) ** 2) * w).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-3)
        Az = (A - mu) / sd
        Az = Az.masked_fill(w <= 0, -1e4)
        Az[..., 0] = -1e4  # lag 1 is dominated by temporal smoothness
        pr = torch.softmax(Az / self.temp, -1)
        logp = (pr * self.loglag).sum(-1)
        prom = (Az.clamp_min(-10).max(-1).values).clamp(-5, 10) / 5.0
        ent = -(pr * torch.log(pr.clamp_min(1e-8))).sum(-1) / math.log(A.shape[-1])
        f = torch.stack([logp / self.loglag[-1], prom, ent], -1).detach()
        g = 2.0 * torch.sigmoid(self.net(f))  # == 1 at initialisation
        return g, torch.exp(logp)


class PeriodBranch(nn.Module):
    """Lightweight supervised tempo estimator on the pyramid (before fusion).

    Predicts, per base cell, a distribution over log-spaced period bins (cells) from
    the concatenated pooled streams with a few depthwise-separable temporal convs.
    Supervised by annotated cycle lengths (cross-entropy on cells inside cycles).
    """

    def __init__(self, d_in: int, d: int, n_bins: int, p_min: float, p_max: float, k: int = 9, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            dil = 2 ** i
            self.blocks.append(nn.ModuleDict({
                "n": nn.LayerNorm(d),
                "dw": nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil, groups=d),
                "pw": nn.Linear(d, d)}))
        self.out = nn.Linear(d, n_bins)
        self.register_buffer("log_bins", torch.linspace(math.log(p_min), math.log(p_max), n_bins), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        h = self.inp(x) * m
        for b in self.blocks:
            y = b["dw"](b["n"](h).transpose(1, 2)).transpose(1, 2)
            h = (h + b["pw"](F.gelu(y))) * m
        return self.out(h)  # B G n_bins

    def period(self, logits: torch.Tensor) -> torch.Tensor:
        pr = torch.softmax(logits.float(), -1)
        return torch.exp((pr * self.log_bins).sum(-1))  # soft period estimate (cells)


class TempoConsistentDecoder(nn.Module):
    """(C) Tempo-consistent decoding: correct harmonic over/under-counting.

    The event density d implies a local candidate period q(t) = 1 / mean local density
    (in base cells).  If the local tempogram shows stronger repetition evidence at 2q
    than at q, the counter is splitting each cycle into two events (double counting);
    stronger evidence at q/2 indicates merged cycles.  A small MLP reads the z-scored
    tempogram at q/2, q, 2q, 3q (linear interpolation along the lag axis), their lag
    support, log q and the local density, and outputs a bounded per-cell factor
    f(t) = 2**tanh(s(t)) in [1/2, 2] applied to d.  The last layer starts at zero
    (identity decoding); q is stop-gradient, so gradients reach d only through the
    multiplication.  Trained by the same counting loss -- no extra loss term.
    """

    def __init__(self, max_lag: int, win: int = 24, hidden: int = 32, use_tempo: bool = True):
        super().__init__()
        self.L = max_lag
        self.use_tempo = use_tempo  # ablation: False reads no tempogram evidence (density statistics only)
        w = torch.hann_window(2 * win + 3, periodic=False)[1:-1]
        self.register_buffer("w", (w / w.sum()).view(1, 1, -1), persistent=False)
        self.net = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _read(Az: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Az: (B, G, L) for lags 1..L; x: (B, G) fractional lag. Returns value and in-range flag."""
        L = Az.shape[-1]
        inr = ((x >= 1) & (x <= L)).float()
        xi = x.clamp(1, L) - 1
        i0 = xi.floor().long().clamp(0, L - 1)
        i1 = (i0 + 1).clamp(max=L - 1)
        a = xi - i0.float()
        v = torch.gather(Az, 2, i0.unsqueeze(-1)).squeeze(-1) * (1 - a) + \
            torch.gather(Az, 2, i1.unsqueeze(-1)).squeeze(-1) * a
        return v * inr, inr

    def forward(self, d: torch.Tensor, A: torch.Tensor, sup: torch.Tensor, mask: torch.Tensor):
        m = mask.float()
        k = self.w.shape[-1]
        num = F.conv1d((d.detach().float() * m).unsqueeze(1), self.w, padding=k // 2).squeeze(1)
        den = F.conv1d(m.unsqueeze(1), self.w, padding=k // 2).squeeze(1).clamp_min(1e-4)
        dbar = (num / den).clamp_min(1.0 / (4 * self.L))
        q = (1.0 / dbar).clamp(1.0, 4.0 * self.L)  # candidate period (cells), stop-gradient
        w = sup.float()
        n = w.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (A * w).sum(-1, keepdim=True) / n
        sd = (((A - mu) ** 2) * w).sum(-1, keepdim=True).div(n).sqrt().clamp_min(1e-3)
        Az = ((A - mu) / sd * w).float()
        feats = []
        for mult in (0.5, 1.0, 2.0, 3.0):
            v, inr = self._read(Az, q * mult)
            if not self.use_tempo:
                v = torch.zeros_like(v)
            feats += [v, inr]
        feats += [torch.log(q / self.L), torch.log(dbar * self.L)]
        s = self.net(torch.stack(feats, -1)).squeeze(-1)
        f = torch.pow(2.0, torch.tanh(s))
        return d * f * m, f


# ----------------------------------------------------------------------------- heads
class SingleRateHead(nn.Module):
    def __init__(self, cfg: dict, rate: int):
        super().__init__()
        d = cfg["d"]
        self.rate = rate
        self.pool = SpatialPool(cfg.get("in_dim", 768), d, cfg.get("spatial", "attn"))
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
    """Multi-rate token pyramid fused per cell, optionally with tempogram components.

    ``tg_bias``  : local tempogram of the finest stream as a per-head attention bias.
    ``tg_input`` : tempogram descriptor appended to the fused input (alternative use).
    """

    def __init__(self, cfg: dict, rates: list[int]):
        super().__init__()
        d = cfg["d"]
        self.rates = rates
        self.tg_bias = bool(cfg.get("tg_bias", False))
        self.tg_input = bool(cfg.get("tg_input", False))
        self.tc_decode = bool(cfg.get("tc_decode", False))
        self.bias_kind = cfg.get("bias_kind", "tempogram")  # tempogram | sim | relpos (controls)
        self.tempo_src = cfg.get("tempo_src", "finest")  # finest | fused
        self.bias_lag = int(cfg.get("max_lag", 160))
        if self.tg_bias and self.bias_kind == "relpos":
            self.relpos = nn.Parameter(torch.zeros(cfg["heads"], self.bias_lag + 2))
        self.periodic_pool = bool(cfg.get("periodic_pool", False))
        self.tempo_gate = bool(cfg.get("tempo_gate", False))
        # supervised tempo estimator and explicit tokens-per-cycle rate gating
        self.period_head = bool(cfg.get("period_head", False))
        self.matched_gate = bool(cfg.get("matched_gate", False))
        self.K_star = float(cfg.get("tokens_per_cycle", 8.0))
        self.gate_sigma = float(cfg.get("gate_sigma", 1.0))  # in log2 units
        self.pp_lag, self.pp_win = int(cfg.get("pp_max_lag", 80)), int(cfg.get("pp_win", 24))
        self.pools = nn.ModuleDict({str(s): SpatialPool(cfg.get("in_dim", 768), d, cfg.get("spatial", "attn"), self.periodic_pool)
                                    for s in rates})
        # frozen video-MLLM conditioning: a time-aligned semantic stream fused like another rate, and a
        # global summary (instruction/answer hidden states) that modulates every fused cell (FiLM)
        self.mllm_stream = bool(cfg.get("mllm_stream", False))
        self.mllm_film = bool(cfg.get("mllm_film", False))
        if self.mllm_stream:
            self.m_vis = nn.Sequential(nn.LayerNorm(cfg["mllm_vis_dim"]), nn.Linear(cfg["mllm_vis_dim"], d))
        if self.mllm_film:
            Dg = cfg["mllm_glob_dim"]
            self.m_glob = nn.Sequential(nn.LayerNorm(Dg), nn.Linear(Dg, d), nn.GELU(), nn.Linear(d, 2 * d))
            nn.init.zeros_(self.m_glob[-1].weight)  # identity modulation at initialisation
            nn.init.zeros_(self.m_glob[-1].bias)
        # condition dropout: with this probability a training video is decoded without semantic
        # modulation, so the decoder cannot lean on the conditioning vector to memorise videos
        self.mllm_drop = float(cfg.get("mllm_drop", 0.0))
        self.fuse = nn.Linear(d * (len(rates) + int(self.tg_input) + int(self.mllm_stream)), d)
        # optional cross-rate attention: per cell, the rate tokens attend to each other before fusion
        self.rate_fusion = cfg.get("rate_fusion", "concat")
        if self.rate_fusion == "attn":
            self.rate_emb = nn.Parameter(torch.zeros(len(rates), d))
            self.rate_attn = nn.TransformerEncoderLayer(d, 4, 2 * d, cfg["dropout"], batch_first=True, norm_first=True)
        self.rate_dropout = float(cfg.get("rate_dropout", 0.0))
        self.enc = TemporalEncoder(d, cfg["layers"], cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"],
                                   tg_bias=self.tg_bias)
        self.head = DensityHead(d)
        # dataset token: one learned vector per source dataset added to every fused cell, so that
        # jointly trained datasets share representation but keep their own counting conventions
        self.n_domains = int(cfg.get("n_domains", 0))
        if self.n_domains:
            self.dom = nn.Embedding(self.n_domains, d)
            nn.init.zeros_(self.dom.weight)  # starts as the shared model
        # optional per-dataset density heads on the shared temporal encoder
        self.dom_heads = bool(cfg.get("dom_heads", False)) and self.n_domains > 0
        if self.dom_heads:
            self.heads_d = nn.ModuleList([DensityHead(d) for _ in range(self.n_domains)])
        if self.tg_bias or self.tg_input or self.tc_decode or self.tempo_gate:
            self.tempo = LocalTempogram(cfg.get("max_lag", 160), cfg.get("tempo_win", 48),
                                        cfg.get("detach_tempo", True))
        if self.tempo_gate:
            self.gate = TempoGate(len(rates), cfg.get("max_lag", 160), cfg.get("gate_temp", 0.5))
        if self.period_head or self.matched_gate:
            self.pbranch = PeriodBranch(d * len(rates), d // 2, cfg.get("n_period_bins", 64), 2.0,
                                        2.0 * cfg.get("tsm_lag", 96))
        if self.tg_input:
            L = cfg.get("max_lag", 160)
            self.tempo_proj = nn.Sequential(nn.Linear(2 * L, d), nn.GELU(), nn.Linear(d, d))
        if self.tc_decode:
            self.tcd = TempoConsistentDecoder(cfg.get("max_lag", 160), cfg.get("tc_win", 24),
                                              use_tempo=cfg.get("tc_use_tempo", True))

    def decode(self, h: torch.Tensor, batch: dict) -> torch.Tensor:
        """Density from base-grid features (shared head, or the head of each video's dataset)."""
        if self.dom_heads:
            allh = torch.stack([hd(h) for hd in self.heads_d], 0)  # n_domains B G
            return allh.gather(0, batch["domain"].view(1, -1, 1).expand(1, -1, allh.shape[-1]))[0]
        return self.head(h)

    def forward(self, batch: dict) -> dict:
        G = batch["mask"].shape[1]
        mask = batch["mask"]
        sal = None
        if self.periodic_pool:  # periodicity saliency on the finest stream, shared by all rates
            r0 = self.rates[0]
            sal0 = periodic_saliency(batch["feats"][r0], batch["masks"][r0], self.pp_lag, self.pp_win)
            sal = {}
            for s in self.rates:
                f = s // r0
                if f == 1:
                    sal[s] = sal0
                else:
                    t = F.avg_pool1d(sal0.transpose(1, 2), f, f, ceil_mode=True).transpose(1, 2)
                    Ls = batch["feats"][s].shape[1]
                    t = F.pad(t, (0, 0, 0, max(0, Ls - t.shape[1])))[:, :Ls]
                    sal[s] = t * batch["masks"][s].unsqueeze(-1).float()
        xs = [to_base(self.pools[str(s)](batch["feats"][s], None if sal is None else sal[s]), s, G)
              for s in self.rates]
        tbias, out = None, {}
        A = sup = None
        if self.tg_bias or self.tg_input or self.tc_decode or self.tempo_gate:
            if self.tempo_src == "fused":  # periodic evidence from all pyramid levels (fused, gradient-stopped)
                src = self.fuse(torch.cat(xs + ([] if not self.tg_input else [torch.zeros_like(xs[0])]), -1))
            else:  # finest stream
                src = xs[0]
            A, sup = self.tempo(src, mask)
            out["tempogram"] = A
            if self.tempo_gate:
                g, out["tempo_period"] = self.gate(A, sup)  # (B, G, S)
                out["gate"] = g
                xs = [x * g[..., j:j + 1].to(x.dtype) for j, x in enumerate(xs)]
        if self.period_head or self.matched_gate:
            plog = self.pbranch(torch.cat(xs, -1), mask)
            out["period_logits"] = plog
            if self.matched_gate:
                # explicit tempo-matched sampling: stream s sees p/s tokens per cycle; weight streams by
                # a Gaussian in log2(tokens per cycle) around K*, normalised so the best stream has weight 1.
                p = self.pbranch.period(plog).detach()  # (B, G) cells of the base grid
                n = torch.stack([p / s for s in self.rates], -1)
                lg = -0.5 * ((torch.log2(n.clamp_min(1e-3)) - math.log2(self.K_star)) / self.gate_sigma) ** 2
                g = torch.exp(lg - lg.max(-1, keepdim=True).values)
                out["gate"] = g
                xs = [x * g[..., j:j + 1].to(x.dtype) for j, x in enumerate(xs)]
        if self.tg_input:
            xs = xs + [self.tempo_proj(torch.cat([A, sup], -1))]
        if self.tg_bias:
            if self.bias_kind == "tempogram":
                tbias = tempogram_bias(A, sup)
            elif self.bias_kind == "sim":
                tbias = similarity_bias(xs[0], mask, self.bias_lag)
            else:  # "relpos": learned per-head table over |lag| (content-independent control)
                pos = torch.arange(G, device=mask.device)
                lag = (pos[None, :] - pos[:, None]).abs().clamp(max=self.bias_lag + 1)
                tbias = self.relpos[:, lag].unsqueeze(0)  # 1 H G G
        if "knockout" in batch:  # analysis only: zero the pooled tokens of the listed rates at inference
            xs = [torch.zeros_like(x) if (j < len(self.rates) and self.rates[j] in batch["knockout"]) else x
                  for j, x in enumerate(xs)]
        if self.training and self.rate_dropout > 0 and len(self.rates) > 1:
            # drop one whole rate stream per video with probability rate_dropout (rescale the rest)
            B = xs[0].shape[0]
            drop = (torch.rand(B, device=mask.device) < self.rate_dropout)
            which = torch.randint(0, len(self.rates), (B,), device=mask.device)
            S = len(self.rates)
            keep = [1.0 - (drop & (which == j)).float() for j in range(S)]
            scale = S / torch.stack(keep, 0).sum(0)
            xs = [x * (keep[j] * scale).view(B, 1, 1).to(x.dtype) for j, x in enumerate(xs[:S])] + xs[S:]
        if self.rate_fusion == "attn":
            S = len(self.rates)
            t = torch.stack(xs[:S], 2) + self.rate_emb.to(xs[0].dtype)  # B G S d
            B_, G_, _, D_ = t.shape
            t = self.rate_attn(t.reshape(B_ * G_, S, D_)).reshape(B_, G_, S, D_)
            xs = [t[:, :, j] for j in range(S)] + xs[S:]
        if self.mllm_stream:
            sv = self.m_vis(batch["mllm_vis"].to(xs[0].dtype))  # B Gg d
            sg = torch.gather(sv, 1, batch["mllm_idx"][:, :G].unsqueeze(-1).expand(-1, -1, sv.shape[-1]))
            xs = xs + [sg.to(xs[0].dtype)]
        z = self.fuse(torch.cat(xs, -1))
        if self.n_domains:
            z = z + self.dom(batch["domain"]).unsqueeze(1).to(z.dtype)
        if self.mllm_film:
            gam, bet = self.m_glob(batch["mllm_glob"].to(z.dtype)).chunk(2, -1)
            if self.training and self.mllm_drop > 0:
                keep = (torch.rand(gam.shape[0], 1, device=gam.device) >= self.mllm_drop).to(gam.dtype)
                gam, bet = gam * keep, bet * keep
            z = z * (1 + gam.unsqueeze(1)) + bet.unsqueeze(1)
        h = self.enc(z, mask, tbias=tbias)
        out["hidden"] = h  # temporal-encoder output on the base grid (used by the LLM head)
        dens = self.decode(h, batch) * mask.float()
        if self.tc_decode:
            out["density_raw"] = dens
            dens, out["tc_factor"] = self.tcd(dens, A, sup, mask)
        out["density"] = dens
        return out


class TSMHead(nn.Module):
    """TransRAC-style: multi-scale (per-rate) multi-head banded self-similarity."""

    def __init__(self, cfg: dict, rates: list[int]):
        super().__init__()
        d = cfg["d"]
        self.rates = rates
        self.max_lag = cfg.get("tsm_lag", 96)
        self.nh = cfg.get("tsm_heads", 4)
        self.pools = nn.ModuleDict({str(s): SpatialPool(cfg.get("in_dim", 768), d, cfg.get("spatial", "attn")) for s in rates})
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
        self.pool = SpatialPool(cfg.get("in_dim", 768), d, cfg.get("spatial", "attn"))
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
        self.pools = nn.ModuleDict({str(s): SpatialPool(cfg.get("in_dim", 768), d, cfg.get("spatial", "attn")) for s in rates})
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
        # router input: "both" = expert context + tempogram, "tempo" = tempogram only, "feat" = context only
        self.router_input = cfg.get("router_input", "both" if self.use_tempo else "feat")
        r_in = d * (2 if self.router_input == "both" else 1)
        self.router = nn.Sequential(nn.LayerNorm(r_in), nn.Linear(r_in, d), nn.GELU(), nn.Linear(d, len(rates)))
        self.router_smooth = int(cfg.get("router_smooth", 0))
        # mix at the "feature" level (shared fuse encoder + one head) or the "density" level
        # (one density head per rate expert; routed mixture of densities, decoded once by summation)
        self.mix_level = cfg.get("mix_level", "feature")
        if self.mix_level == "feature":
            self.fuse = TemporalEncoder(d, cfg.get("fuse_layers", 1), cfg["heads"], cfg.get("conv_k", 5), cfg["dropout"])
            self.head = DensityHead(d)
        else:
            self.heads_s = nn.ModuleDict({str(s): DensityHead(d) for s in rates})

    def forward(self, batch: dict) -> dict:
        G = batch["mask"].shape[1]
        mask = batch["mask"]
        ys, xs, ds = [], [], []
        for s in self.rates:
            x = self.pools[str(s)](batch["feats"][s])
            m = batch["masks"][s]
            y = self.experts[str(s)](x, m, pos_scale=float(s))
            if self.mix_level == "density":
                ds.append(to_base(self.heads_s[str(s)](y) * m.float(), s, G, density=True))
            ys.append(to_base(y, s, G))
            xs.append(to_base(x, s, G))
        Y = torch.stack(ys, 2)  # B G S D
        cue_feat = xs[0]  # finest-rate features carry the local tempo evidence
        A, sup = self.tempo(cue_feat, mask)
        tfeat = self.tempo_proj(torch.cat([A, sup], -1))
        ctx = Y.mean(2)
        if self.router_input == "both":
            r_in = torch.cat([ctx, tfeat], -1)
        elif self.router_input == "tempo":
            r_in = tfeat
        else:
            r_in = ctx
        logits = self.router(r_in)
        if self.router_smooth > 1:  # temporal smoothing of routing logits (masked box filter)
            k = self.router_smooth
            mf = mask.float().unsqueeze(1)
            num = F.avg_pool1d((logits.transpose(1, 2) * mf), k, 1, k // 2, count_include_pad=False)
            den = F.avg_pool1d(mf, k, 1, k // 2, count_include_pad=False).clamp_min(1e-6)
            logits = (num / den).transpose(1, 2)[:, :G]
        if self.router_mode == "uniform":
            g = torch.full_like(logits, 1.0 / len(self.rates))
        elif self.router_mode == "hard":
            soft = torch.softmax(logits / self.tau, -1)
            hard = F.one_hot(soft.argmax(-1), len(self.rates)).float()
            g = hard + soft - soft.detach()
        else:  # "soft" and "feat" (feat = soft routing without tempogram, set use_tempogram False)
            g = torch.softmax(logits / self.tau, -1)
        if self.mix_level == "density":
            D = torch.stack(ds, 2)  # B G S
            dens = (g * D).sum(2) * mask.float()
            return {"density": dens, "route": g, "tempogram": A, "expert_density": D}
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
    if kind == "llm":  # LoRA-tuned LLM over the temporal video tokens (llm_head.py)
        from temporac_cvpr.llm_head import TempoRACLLM

        return TempoRACLLM(cfg, rates)
    raise ValueError(kind)
