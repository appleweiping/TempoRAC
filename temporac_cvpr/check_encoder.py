"""Server-side check: our ViT block equals pySlowFast MultiScaleBlock on real weights.

Also checks that the full encoder output is finite and records token statistics.
"""
from __future__ import annotations

import json
import sys

import torch

from temporac_cvpr.vit_encoder import load_stmae_vitb


def main(ckpt: str, escounts_root: str) -> None:
    torch.manual_seed(0)
    model, rep = load_stmae_vitb(ckpt)
    out = {"load": rep}
    try:
        # Load the vendored pySlowFast attention/common modules by path, bypassing the
        # package __init__ (which pulls detectron2/pytorchvideo dependencies).
        import importlib.util
        import types

        for pkg in ("slowfast", "slowfast.models"):
            sys.modules.setdefault(pkg, types.ModuleType(pkg))
        for name in ("common", "attention"):
            spec = importlib.util.spec_from_file_location(
                f"slowfast.models.{name}", f"{escounts_root}/slowfast/models/{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"slowfast.models.{name}"] = mod
            spec.loader.exec_module(mod)
        MultiScaleBlock = sys.modules["slowfast.models.attention"].MultiScaleBlock

        ref = MultiScaleBlock(dim=768, dim_out=768, num_heads=12, input_size=[8, 14, 14], mlp_ratio=4.0,
                              qkv_bias=True, drop_path=0.0, norm_layer=lambda d: torch.nn.LayerNorm(d, eps=1e-6),
                              kernel_q=(), kernel_kv=(), stride_q=(), stride_kv=(), mode="conv",
                              has_cls_embed=True, pool_first=False, rel_pos_spatial=False,
                              rel_pos_temporal=False, rel_pos_zero_init=False, residual_pooling=False,
                              dim_mul_in_att=False, separate_qkv=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state"]
        sub = {k[len("blocks.0."):]: v for k, v in state.items() if k.startswith("blocks.0.")}
        missing = ref.load_state_dict(sub, strict=False)
        x = torch.randn(2, 1 + 8 * 196, 768)
        with torch.no_grad():
            y_ref, _ = ref.eval()(x, [8, 14, 14])
            y_our = model.blocks[0](x)
        out["block_max_abs_diff"] = float((y_ref - y_our).abs().max())
        out["ref_missing"] = [list(missing.missing_keys), list(missing.unexpected_keys)]
    except Exception as e:  # pragma: no cover - record, do not hide
        out["block_check_error"] = repr(e)
    clip = torch.randn(2, 3, 16, 224, 224)
    with torch.no_grad():
        tok = model(clip)
    out["tokens_shape"] = list(tok.shape)
    out["tokens_finite"] = bool(torch.isfinite(tok).all())
    out["tokens_std"] = float(tok.std())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
