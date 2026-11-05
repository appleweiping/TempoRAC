"""Official ESCounts video encoding on full VALID131 only. No exemplars.

Window/encoder operations derive from EveryShotCounts (MIT, S. Sinha, 2024).
"""
import argparse
import math
from pathlib import Path

from escounts_valid131_common import (
    CONTRACT, ENCODER_SHA, configure_runtime, data_path, load_official,
    restore_encoder, selected_video_loader, sha256,
    source_identity, valid_rows, verify_sha, video_manifest, write_json,
    reusable_token, save_or_verify, storage_path,
)


def encode_video(video, model, device):
    import numpy as np
    import torch
    # Official save_tokens: append 64 zero frames, stride16, sample16/64.
    c, t, h, w = video.shape
    video = torch.cat([video, torch.zeros([c, 64, h, w])], 1)
    clips = [video[:, np.linspace(j, j + 64, 17)[:16].astype(int)] for j in range(0, t, 16)]
    data = torch.stack(clips).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
            encoded, thw = model(data)
            encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2])
    return encoded.cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--video-root", required=True, help="Original VALID RGB directory ending in valid/")
    p.add_argument("--encoder", required=True)
    p.add_argument("--token-dir", required=True)
    p.add_argument("--output", required=True, help="Exclusive output JSON manifest")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    configure_runtime(args.output)
    identity = source_identity(args.source)
    rows = valid_rows(args.source)
    videos = video_manifest(args.source, args.video_root)
    verify_sha(args.encoder, ENCODER_SHA)
    output_dir = data_path(args.token_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.manual_seed(0)
    lineage = {"contract": CONTRACT, "source_commit": identity["commit"],
        "source_file_sha256": identity["source_file_sha256"], "parameters": identity["parameters"],
        "adapter_file_sha256": identity["adapter_file_sha256"],
        "encoder_sha256": ENCODER_SHA, "torch": torch.__version__, "device": args.device}
    results, pending = [], []
    for index, (row, video_record) in enumerate(zip(rows, videos)):
        target = output_dir / Path(row["name"]).with_suffix(".npz")
        expected = {**lineage, "name": row["name"], "video_sha256": video_record["sha256"]}
        result = reusable_token(target, expected)
        results.append(result)
        if result is None:
            pending.append((index, target, expected))
    if not pending:
        save_or_verify(args.output, {"contract": CONTRACT, **identity, "torch": torch.__version__,
            "device": args.device, "encoder_sha256": ENCODER_SHA, "videos": videos, "tokens": results,
            "exemplar_features": "not used or generated", "complete": True})
        print("Reused all 131 verified token files; no encoder import/forward")
        return
    Model, cfg = load_official(args.source)
    encoder = Model(cfg=cfg, just_encode=True, use_precomputed=False, encodings="mae").to(args.device)
    payload = torch.load(args.encoder, map_location="cpu", weights_only=False)
    weights = payload["model_state"] if "model_state" in payload else payload["model"]
    audit = restore_encoder(encoder, weights)
    save_or_verify(str(args.output) + ".encoder-audit.json", audit)
    if audit["unmapped"]:
        raise RuntimeError("Encoder mapping left reachable tensors unmapped; no feature execution permitted")
    device = torch.device(args.device)
    model = torch.nn.DataParallel(encoder, device_ids=[device.index or 0] if device.type == "cuda" else [])
    model.eval()
    ds = selected_video_loader(args.source, args.video_root)
    if len(ds) != 131:
        raise ValueError("Expected full VALID131")
    for index, target, expected in pending:
        row = rows[index]
        video, starts, ends, name = ds[index]
        if name != Path(row["name"]).stem:
            raise ValueError("Official decoder changed sample order")
        encoded = encode_video(video, model, args.device)
        import numpy as np
        with target.open("xb") as f:
            np.savez(f, encoded)
        result = {**expected, "path": str(target), "shape": list(encoded.shape), "dtype": str(encoded.dtype), "sha256": sha256(target)}
        write_json(target.with_suffix(target.suffix + ".receipt.json"), result)
        results[index] = result
    save_or_verify(args.output, {"contract": CONTRACT, **identity, "torch": torch.__version__,
        "device": args.device, "encoder_sha256": ENCODER_SHA, "videos": videos, "tokens": results,
        "exemplar_features": "not used or generated", "complete": len(results) == 131})


if __name__ == "__main__":
    main()
