"""VALID131-only ESCounts interfaces; no training or model reimplementation.

Numerical preprocessing excerpts derive from Saptarshi Sinha's MIT-licensed
EveryShotCounts f18fcf1933abb3d4f199fd12ee3cafc53581576a. Keep upstream LICENSE.
The separate contract records the three permitted entry adaptations.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "f18fcf1933abb3d4f199fd12ee3cafc53581576a"
CSV_SHA = "49695468686d894c06beb27fc4182675c762960175126a6eb8372b6148f2d148"
ENCODER_SHA = "b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8"
DECODER_SHA = "297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559"
CONTRACT = "escounts-valid131-two-offset-v1"
PARAMETERS = dict(evaluation_window_mode="two_offset", token_pool_ratio=0.4,
    window_size=[4, 7, 7], shot_num=0, density_peak_width=0.5, scale_counts=100,
    seed=0, batch_size=1, num_gpus=1, shuffle=False, drop_last=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA mismatch: {path}: {actual}")
    return actual


def data_path(path):
    p = storage_path(path)
    if any(part.lower() in {"train", "test"} for part in p.parts):
        raise ValueError(f"Forbidden train/test data path: {p}")
    return p


def storage_path(path):
    """Constrain all generated state to the authorized local/remote project disks."""
    p = Path(path).resolve()
    if os.name == "nt":
        if p.drive.casefold() != "e:":
            raise ValueError(f"Generated/read data must stay on E: {p}")
    else:
        root = Path("/mnt/4tssd/lilei/TempoRAC")
        if p != root and root not in p.parents:
            raise ValueError(f"Outside authorized project storage: {p}")
    return p


def source_identity(source):
    source = Path(source).resolve()
    opts = dict(text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], **opts).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], **opts).strip()
    if head != COMMIT or status:
        raise ValueError(f"Expected clean official commit {COMMIT}; got {head}, status={status!r}")
    files = ("video_mae_cross_full_attention.py", "Repcount_multishot_loader.py",
             "Rep_count.py", "save_swim_features.py", "demo.py", "configs/pretrain_config.yaml")
    adapter_files = ("escounts_valid131_common.py", "escounts_valid131_extract.py", "escounts_valid131_eval.py")
    return {"commit": head, "source": str(source), "source_file_sha256": {n: sha256(source / n) for n in files},
            "adapter_file_sha256": {n: sha256(Path(__file__).parent / n) for n in adapter_files},
            "valid_csv_sha256": CSV_SHA, "parameters": PARAMETERS}


def valid_rows(source):
    path = Path(source) / "datasets/repcount/valid_with_fps.csv"
    verify_sha(path, CSV_SHA)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    if len(rows) != 131 or len(set(names)) != 131:
        raise ValueError("VALID131 cardinality/duplicate mismatch")
    if any(Path(n).name != n or not n.endswith(".mp4") for n in names):
        raise ValueError("Unexpected video basename")
    if "stu9_67.mp4" not in names:
        raise ValueError("Zero-count video missing")
    return rows


def write_json(path, value):
    path = storage_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path):
    with storage_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def verify_fields(record, expected):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Resume lineage mismatch: {key}")


def save_or_verify(path, value):
    path = storage_path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing result differs: {path}")
    else:
        write_json(path, value)


def reusable_token(path, expected):
    path = data_path(path)
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() != receipt.exists():
        raise ValueError(f"Orphan token/receipt retained for inspection: {path}")
    if not path.exists():
        return None
    record = read_json(receipt)
    verify_fields(record, expected)
    verify_sha(path, record["sha256"])
    return record


def configure_runtime(output):
    root = storage_path(output).parent / "runtime-cache"
    root.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        os.environ[key] = str(root)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def video_manifest(source, video_root):
    root = data_path(video_root)
    entries = []
    for row in valid_rows(source):
        p = data_path(root / row["name"])
        if not p.is_file():
            raise FileNotFoundError(p)
        entries.append({"name": row["name"], "path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
    return entries


def cycle_points(row):
    return [int(float(value)) for key, value in row.items()
            if "L" in key and value.strip() and not math.isnan(float(value))]


def density_target(row):
    """Same grid, assignment, Gaussian and half-before-sum as official loader."""
    import numpy as np
    import torch
    from scipy import ndimage
    points = cycle_points(row)
    if len(points) % 2:
        raise ValueError(f"Odd cycle endpoints: {row['name']}")
    n = int(float(row["num_frames"]))
    start, end = float(row["segment_start"]), float(row["segment_end"])
    ids = np.arange(n)[int((start // 8) * 8):int(math.ceil(end / 8) * 8)][0::8]
    if not len(ids):
        raise ValueError(f"Empty target grid: {row['name']}")
    density = np.zeros(len(ids))
    for i in range(0, len(points), 2):
        if points[i] == points[i + 1]:
            continue
        a = (points[i] // 8) * 8
        b = min(np.ceil(points[i + 1] / 8) * 8, ids[-1])
        if a in ids and b in ids:
            ai, bi = np.where(ids == a)[0][0], np.where(ids == b)[0][0]
            density[(ai + bi) // 2] = 1
    target = torch.from_numpy(ndimage.gaussian_filter1d(density, sigma=0.5, order=0)).half()
    return target, target.sum()


def two_offset_tokens(tokens, row):
    import numpy as np
    import torch
    if tokens.ndim != 5 or tokens.shape[1] != 768:
        raise ValueError(f"Expected Sx768xTxHxW tokens: {row['name']}, {tokens.shape}")
    low = int(float(row["segment_start"]) // 8)
    up = math.ceil(float(row["segment_end"]) / 8)
    windows = []
    for offset, a, b in ((0, low, up), (2, max(low - 4, 0), max(up - 4, 0))):
        selected = tokens[offset::4]
        s, c, t, h, w = selected.shape
        arranged = selected.transpose(1, 0, 2, 3, 4).reshape(c, s * t, h, w)
        x = torch.from_numpy(np.ascontiguousarray(arranged[:, a:b]))
        if x.shape[1] == 0:
            raise ValueError(f"Empty offset {offset}: {row['name']}")
        factor = math.ceil(tokens.shape[-1] * 0.4)
        x = torch.nn.functional.adaptive_avg_pool3d(x, (x.shape[-3], factor, factor))
        windows.append(x)
    return tuple(windows)


def batch_window(x, device):
    # Official compact collate rearranges C T H W -> B (T H W) C.
    return x.permute(1, 2, 3, 0).reshape(1, -1, x.shape[0]).to(device=device).float()


def merge_predictions(p1, p2):
    import torch
    if p1.shape != p2.shape:
        p2 = torch.cat([torch.zeros(1, 4, dtype=p2.dtype, device=p2.device), p2], 1)
    if p1.shape != p2.shape:
        raise ValueError(f"Official four-zero padding still mismatches {tuple(p1.shape)} / {tuple(p2.shape)}")
    return (p1 + p2) / 2


def score_records(records):
    import numpy as np
    q = np.array([x["rounded_count"] for x in records], dtype=np.float32)
    g = np.array([x["native_target"] for x in records], dtype=np.float32)
    n = np.array([x["csv_count"] for x in records])
    difference = np.abs(q - g)
    positive = n > 0
    return {"native_VALID131": {"n": len(records),
        "MAE_native": float(np.mean(difference / (g + .1))),
        "OBO_native": float(np.mean(difference <= 1)),
        "OBZ_native": float(np.mean(np.abs(np.round(q) - np.round(g)) == 0)),
        "RMSE_count_native": float(np.sqrt(np.mean(difference ** 2)))},
        "paperform_CSV_positive130": {"n": int(positive.sum()),
        "MAE_paperform_CSV_positive130": float(np.mean(np.abs(q[positive] - n[positive]) / n[positive]))},
        "zero_count_rows": [{**x, "absolute_error_csv": abs(x["rounded_count"]),
            "false_positive": x["rounded_count"] > 0} for x in records if x["csv_count"] == 0]}


def load_official(source):
    """Import the unchanged official model only after runtime paths are configured."""
    import importlib
    sys.path.insert(0, str(Path(source).resolve()))
    model_module = importlib.import_module("video_mae_cross_full_attention")
    parser_module = importlib.import_module("slowfast.utils.parser")
    args = argparse.Namespace(opts=None, output_dir=os.environ["TMP"])
    cfg = parser_module.load_config(args, path_to_config=str(Path(source) / "configs/pretrain_config.yaml"))
    return model_module.SupervisedMAE, cfg


def selected_video_loader(source, video_root):
    """Compile original RGB loader with ONLY valid CSV and zero-count retention."""
    import types
    video_root = data_path(video_root)
    if video_root.name != "valid":
        raise ValueError("video-root must be the original valid/ directory")
    path = Path(source) / "Rep_count.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Rep_count")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    for node in init.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "csv_path" for t in node.targets):
            node.value = ast.Constant(str(Path(source) / "datasets/repcount/valid_with_fps.csv"))
    # Remove exactly the original positive-count filter; keep other verified filters.
    filtered = [n for n in init.body if isinstance(n, ast.Assign)
        and "self.df['count'] > 0" in ast.unparse(n)]
    if len(filtered) != 1:
        raise ValueError("Official zero-count filter changed")
    init.body.remove(filtered[0])
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    ast.fix_missing_locations(tree)
    module = types.ModuleType("escounts_selected_valid_video")
    exec(compile(tree, str(path) + " [valid-only positive filter removed]", "exec"), module.__dict__)
    ds = module.Rep_count(split="valid", sampling_interval=1, encode_only=True, data_dir=str(Path(video_root).parent))
    if list(ds.df["name"]) != [r["name"] for r in valid_rows(source)]:
        raise ValueError("Official video loader changed VALID131 order/population")
    return ds


def restore_encoder(encoder, state_dict):
    """Complete official-demo direct/q-k-v mapping, with reachable-state coverage.

    Approved addendum replaces the premature outer break in save_swim_features.
    This helper changes loading only, never the encoder architecture.
    """
    import torch
    target = encoder.state_dict()
    originals = {}
    for key in state_dict:
        normalized = key.removeprefix("module.").removeprefix("encoder.")
        if normalized in originals:
            raise ValueError(f"Duplicate normalized source key: {normalized}")
        originals[normalized] = key
    def reachable(name):
        if name.startswith(("patch_embed.", "blocks.", "norm.")):
            return True
        if name == "cls_token":
            return encoder.cls_embed_on
        if name.startswith("pos_embed"):
            if name == "pos_embed":
                return encoder.use_fixed_sincos_pos or (encoder.use_abs_pos and not encoder.sep_pos_embed)
            return encoder.use_abs_pos and encoder.sep_pos_embed
        return False
    required = [name for name in target if reachable(name.removeprefix("module."))]
    if {int(n.removeprefix("module.").split(".")[1]) for n in required if n.removeprefix("module.").startswith("blocks.")} != set(range(12)):
        raise ValueError("Expected 12 official encoder blocks")
    audit = {"mapping": "complete official demo direct/q-k-v order", "parameters": [], "unmapped": [],
             "non_encoder_target_keys": [k for k in target if k not in required]}
    used = set()
    with torch.no_grad():
        for name in required:
            normalized = name.removeprefix("module.")
            if normalized in originals:
                keys = [originals[normalized]]
                value, mode = state_dict[keys[0]], "direct"
            elif normalized.startswith("blocks.") and ".attn.qkv." in normalized:
                needed = [normalized.replace(".qkv.", f".{part}.") for part in ("q", "k", "v")]
                if any(k not in originals for k in needed):
                    audit["unmapped"].append(name)
                    continue
                keys = [originals[k] for k in needed]
                value, mode = torch.cat([state_dict[k] for k in keys], dim=0), "concat_qkv"
            else:
                audit["unmapped"].append(name)
                continue
            if target[name].shape != value.shape or target[name].dtype != value.dtype:
                raise ValueError(f"Encoder shape/dtype mismatch: {name}")
            target[name].copy_(value)
            same = torch.equal(target[name].cpu(), value.cpu())
            audit["parameters"].append({"target": name, "shape": list(value.shape), "dtype": str(value.dtype),
                "sources": keys, "mode": mode, "equal_after_copy": same})
            if not same:
                raise ValueError(f"Encoder copy mismatch: {name}")
            used.update(keys)
    audit["unused_source_keys"] = sorted(set(state_dict) - used)
    audit["complete"] = not audit["unmapped"]
    return audit


def main():
    p = argparse.ArgumentParser(description="Validate all 131 official validation video files; no model import")
    p.add_argument("--source", required=True)
    p.add_argument("--video-root", help="Original valid/ directory")
    p.add_argument("--metadata-only", action="store_true", help="Record all 131 CSV identities only; does not certify video readiness")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    configure_runtime(args.output)
    rows = valid_rows(args.source)
    if not args.metadata_only and not args.video_root:
        p.error("--video-root is required unless --metadata-only")
    write_json(args.output, {"contract": CONTRACT, **source_identity(args.source),
        "readiness": "metadata_only_not_video_verified" if args.metadata_only else "video_bytes_verified_not_decoded",
        "names": [r["name"] for r in rows],
        "videos": None if args.metadata_only else video_manifest(args.source, args.video_root)})


if __name__ == "__main__":
    main()
