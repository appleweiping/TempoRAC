"""Evaluate a TempoRAC-LLM run with every read-out: density (original / 2x view / two-view average),
the LLM's textual count (CountLLM's read-out), and their combinations.

The 8B language model is loaded once; the per-target checkpoints (best_<val split>.pt) only replace
the decoder, projectors and LoRA factors.  Writes eval_llm_<split>.json per target.
Usage: python -m temporac_cvpr.eval_llm RUN_DIR --splits rc_valid,cx_val,ucf_dev
       [--test rc_test,cx_test,ucf_test]  (test only for frozen final configurations)
"""
from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np
import torch

from temporac_cvpr.data import attach_desc, load_videos, to_device
from temporac_cvpr.models import build_model
from temporac_cvpr.train import EvalLoader, load_rates, predict, score, shift_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--splits", default="rc_valid,cx_val,ucf_dev")
    ap.add_argument("--test", default="", help="test splits aligned with --splits (final configurations only)")
    ap.add_argument("--manifest", default="/root/autodl-tmp/TempoRAC/data/UNION/manifest_dev.json")
    ap.add_argument("--features", default=None, help="feature root override (e.g. QUVA, not in the union)")
    ap.add_argument("--aux", default=None, help="auxiliary (second-depth) feature root override")
    ap.add_argument("--desc-roots", default=None, help="comma-separated description dirs override")
    ap.add_argument("--domain", default=None, help="dataset token to use (e.g. RepCountA for QUVA zero-shot)")
    ap.add_argument("--tag", default=None, help="output tag (default: split name)")
    ap.add_argument("--skip-val", action="store_true", help="score only the --test splits (e.g. QUVA zero-shot)")
    args = ap.parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 16) if hard > 0 else 1 << 16, hard))
    run = Path(args.run)
    conf = json.loads((run / "config.json").read_text())
    cfg, features = conf["cfg"], args.features or conf["args"]["features"]
    aux = (args.aux,) if args.aux else tuple(cfg.get("aux_features", ()))
    desc_roots = args.desc_roots.split(",") if args.desc_roots else cfg["llm_desc_roots"]
    manifest = json.loads(Path(args.manifest).read_text())
    device = torch.device("cuda")
    model = build_model(cfg).to(device).eval()
    vals = args.splits.split(",")
    tests = args.test.split(",") if args.test else [None] * len(vals)
    for vs, ts in zip(vals, tests):
        ck = torch.load(run / f"best_{vs}.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        assert not unexpected and all(".base." not in k for k in missing), (missing[:5], unexpected[:5])
        for split in ([] if args.skip_val else [vs]) + ([ts] if ts else []):
            vids = load_videos(manifest, features, load_rates(cfg), {split}, cfg["pool_to"], aux)
            attach_desc(vids, desc_roots)
            for v in vids:
                v.domain = cfg["domains"][args.domain or v.dataset]
            p0 = predict(model, vids, cfg, device)
            sv = [shift_video(v, cfg["rates"]) for v in vids]
            p1 = predict(model, sv, cfg, device)
            loader = EvalLoader(vids, cfg, 2)
            ptxt = {}
            for b, batch in zip(loader.batches, loader.loader):
                batch = to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cs = model.text_counts(batch)
                for v, c in zip(b, cs):
                    ptxt[v.key] = float(c)
            views = {
                "density": p0, "density_shift": p1,
                "density_tta": {k: 0.5 * (p0[k] + p1[k]) for k in p0},
                "text": ptxt,
                "text_density_avg": {k: 0.5 * (p0[k] + ptxt[k]) for k in p0},
                "text_tta_avg": {k: (p0[k] + p1[k] + ptxt[k]) / 3 for k in p0},
            }
            res = {"split": split, "ckpt_epoch": ck.get("epoch"), "metrics": {n: score(p, vids) for n, p in views.items()},
                   "predictions": views}
            name = args.tag if (args.tag and split != vs) else split
            (run / f"eval_llm_{name}.json").write_text(json.dumps(res, indent=1))
            print(split, {n: (round(m["mae_p"], 4), round(m["obo_p"], 3)) for n, m in res["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
