"""Real checkpoint tensor mapping parity; does NOT instantiate the full encoder.

Target names/shapes here are derived from the checkpoint and fused-QKV layout.
Consequently a pass validates mapping values, not actual-model state coverage.
"""
import argparse
import ast
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from escounts_valid131_common import ENCODER_SHA, restore_encoder, verify_sha, write_json


class TargetState:
    cls_embed_on = True
    use_fixed_sincos_pos = False
    use_abs_pos = True
    sep_pos_embed = True

    def __init__(self, shapes):
        self.state = {name: torch.zeros_like(value) for name, value in shapes.items()}

    def state_dict(self):
        return self.state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    verify_sha(args.checkpoint, ENCODER_SHA)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = checkpoint["model_state"]
    shapes = {}
    for name, value in weights.items():
        if not name.startswith(("patch_embed.", "blocks.", "norm.", "pos_embed")) and name != "cls_token":
            continue
        if any(f".attn.{k}." in name for k in ("q", "k", "v")):
            if ".attn.q." in name:
                fused = name.replace(".attn.q.", ".attn.qkv.")
                shapes[fused] = torch.cat([weights[name.replace(".attn.q.", f".attn.{k}.")] for k in ("q", "k", "v")], dim=0)
        else:
            shapes[name] = value
    actual, reference = TargetState(shapes), TargetState(shapes)
    audit = restore_encoder(actual, weights)
    tree = ast.parse((Path(args.source) / "demo.py").read_text(encoding="utf-8"))
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main_fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "name")
    with torch.no_grad():
        exec(compile(ast.Module([loop], []), "<unaltered official demo mapping loop>", "exec"), dict(torch=torch, encoder=reference, state_dict=weights))
    differences = [name for name in actual.state if not torch.equal(actual.state[name], reference.state[name])]
    if audit["unmapped"] or differences:
        raise AssertionError({"unmapped": audit["unmapped"], "differences": differences})
    # Explicit negative coverage check: no missing reachable key may be silently used.
    reduced = dict(weights)
    reduced.pop("norm.weight")
    rejected_audit = restore_encoder(TargetState(shapes), reduced)
    if "norm.weight" not in rejected_audit["unmapped"]:
        raise AssertionError("Missing reachable parameter was not detected")
    write_json(args.output, {"scope": "actual checkpoint tensor mapping / checkpoint-derived fused target state; not instantiated model",
        "torch": torch.__version__, "checkpoint_sha256": ENCODER_SHA, "target_count": len(shapes),
        "concat_count": sum(x["mode"] == "concat_qkv" for x in audit["parameters"]),
        "all_values_equal_to_official_demo": True, "unmapped_rejection_test": True,
        "full_model_state_coverage": "UNVERIFIED: missing official runtime dependencies", "audit": audit})
    print(f"PASS: {len(shapes)} target tensors equal official demo; full model coverage NOT VERIFIED")


if __name__ == "__main__":
    main()
