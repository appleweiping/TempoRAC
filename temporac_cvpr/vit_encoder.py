"""Frozen ST-MAE ViT-B/16 (16x4, Kinetics-400 self-supervised) video encoder.

This re-implements the pySlowFast ``MaskMViT`` ViT-B encoder that produced the
``VIT_B_16x4_MAE_PT.pyth`` checkpoint (the same encoder used by ESCounts), so
that features can be extracted without the pySlowFast/detectron2 stack.

Checkpoint config (read from the stored cfg): CLS_EMBED_ON=True,
SEP_POS_EMBED=True, USE_ABS_POS=True, PATCH_KERNEL=[2,16,16], DEPTH=12,
EMBED_DIM=768, NUM_HEADS=12, MLP_RATIO=4, QKV_BIAS=True, no pooling,
no relative positions, no layer scale, LayerNorm(eps=1e-6).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.heads
        q = self.q(x).reshape(b, n, h, c // h).transpose(1, 2)
        k = self.k(x).reshape(b, n, h, c // h).transpose(1, 2)
        v = self.v(x).reshape(b, n, h, c // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class STMAEViTB(nn.Module):
    """Input ``(B, 3, 16, 224, 224)`` normalised clips; output ``(B, 8, 14, 14, 768)``."""

    def __init__(self, depth: int = 12, dim: int = 768, heads: int = 12):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 196, dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 8, dim))
        self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # B C T H W
        b, c, t, h, w = x.shape
        assert (t, h, w) == (8, 14, 14), (t, h, w)
        x = x.flatten(2).transpose(1, 2)  # B (T H W) C, temporal-major order
        pos = self.pos_embed_spatial.repeat(1, t, 1) + torch.repeat_interleave(
            self.pos_embed_temporal, h * w, dim=1
        )
        pos = torch.cat([self.pos_embed_class, pos], dim=1)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1) + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:].reshape(b, t, h, w, c)


def load_stmae_vitb(path: str) -> tuple[STMAEViTB, dict]:
    # Official pySlowFast checkpoint (SHA-256 verified at download time).
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model_state"]
    model = STMAEViTB()
    wanted = {k: v for k, v in state.items() if k in model.state_dict()}
    model.load_state_dict(wanted, strict=True)
    report = {
        "checkpoint_keys": len(state),
        "loaded_keys": len(wanted),
        "model_keys": len(model.state_dict()),
        "ignored_keys": sorted(k for k in state if k not in wanted)[:8],
        "ignored_count": sum(1 for k in state if k not in wanted),
    }
    assert report["loaded_keys"] == report["model_keys"], report
    return model.eval(), report
