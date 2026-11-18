"""Official-code TransRAC (trained by us on RepCount-A, baselines/transrac_native_v1) on the Countix test split.

Same pieces as that run's final evaluation, imported unchanged: ``native_run.build_model`` (official TransferModel,
Swin-T backbone, 64 frames, scales [1,4,8]), the selected ``checkpoints/best.pt``, ``native_inputs.decode_rgb``
(official raw 64-frame sampler inside the inclusive frame_range crop, 224x224 linear resize), batch 1, no outer
autocast, continuous count = sum of the 64-cell density. Only the population is new (Countix cx_test, 2403
videos, repetition-span crops from the manifest). Per-video records are appended atomically; reruns resume.
Usage (server): envs/transrac-native310/bin/python transrac_countix.py --gpu 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/TempoRAC")
T = ROOT / "baselines/transrac_native_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(T))
    import torch
    from native_run import build_model
    from native_inputs import decode_rgb

    contract = json.loads((T / "RUN_CONTRACT.json").read_text())
    model = build_model(contract)
    # the checkpoint stores torch.__version__ (a TorchVersion); allow-list only that class and keep weights_only
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    ck = torch.load(T / "checkpoints/best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"], strict=True)
    model = model.cuda().eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / "records.jsonl"
    done = {json.loads(l)["video_id"] for l in rec.read_text().splitlines() if l.strip()} if rec.exists() else set()
    rows = [r for r in json.loads((ROOT / "data/UNION/manifest_dev.json").read_text())
            if r.get("dataset") == "Countix" and r["split"] == "cx_test"]
    assert len(rows) == 2403, len(rows)
    f = rec.open("a")
    t0 = time.time()
    with torch.no_grad():
        for n, r in enumerate(rows):
            if r["video_id"] in done:
                continue
            row = dict(r, population="cx_test")
            arr, info = decode_rgb(row)
            x = torch.from_numpy(arr).float().div_(255).transpose(0, 1).unsqueeze(0)  # 1 x 3 x 64 x 224 x 224
            output, _ = model(x.float().cuda())
            assert output.shape == (1, 64) and bool(torch.isfinite(output).all())
            cont = float(output.sum(dim=1)[0])
            f.write(json.dumps({"video_id": r["video_id"], "gt": float(r["count"]), "continuous_count": cont,
                                "roi_frames": info["roi_decoded_frames"], "frame_range": r.get("frame_range")}) + "\n")
            f.flush()
            if n % 100 == 0:
                print(f"{n + 1}/{len(rows)} pred={cont:.2f} gt={r['count']} {(time.time() - t0) / 60:.1f}min", flush=True)
    sys.path.insert(0, str(ROOT / "code"))
    from temporac_cvpr.metrics import count_metrics
    recs = [json.loads(l) for l in rec.read_text().splitlines() if l.strip()]
    res = count_metrics([x["continuous_count"] for x in recs], [x["gt"] for x in recs])
    (out / "result.json").write_text(json.dumps({"cx_test": res, "n": len(recs), "checkpoint": str(T / "checkpoints/best.pt")}, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
