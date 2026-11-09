"""Evaluate a saved run (best.pt) on another split, e.g. the tempo-warp stress test.

Writes RUN/eval_<tag>.json with per-video predictions and metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from temporac_cvpr.data import load_videos
from temporac_cvpr.models import build_model
from temporac_cvpr.train import predict, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint file in each run dir (e.g. best_rc_valid.pt)")
    ap.add_argument("--mllm-root", default=None, help="MLLM feature dir for this dataset (overrides the training cfg)")
    ap.add_argument("--domain", default=None,
                    help="dataset token to use (e.g. RepCountA for QUVA zero-shot); default: each video's own")
    ap.add_argument("--aux", nargs="*", default=None,
                    help="feature roots of the extra encoders for this dataset (overrides the training cfg)")
    args = ap.parse_args()
    import resource  # one memory-mapped file per video and rate

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 16) if hard > 0 else 1 << 16, hard))
    device = torch.device("cuda")
    manifest = []
    for m in args.manifest:
        manifest += json.loads(Path(m).read_text())
    cache = {}
    for run in args.run:
        ck = torch.load(Path(run) / args.ckpt, map_location="cpu", weights_only=True)
        cfg = ck["cfg"]
        from temporac_cvpr.train import load_rates

        aux = tuple(cfg.get("aux_features", ()) if args.aux is None else args.aux)
        key = (tuple(load_rates(cfg)), cfg["pool_to"], aux)
        if key not in cache:
            cache[key] = load_videos(manifest, args.features, load_rates(cfg), {args.split}, cfg["pool_to"], aux)
        vids = cache[key]
        if cfg.get("domains"):
            for v in vids:
                v.domain = cfg["domains"][args.domain or v.dataset]
        if cfg.get("mllm_root"):
            from temporac_cvpr.data import attach_mllm

            attach_mllm(vids, args.mllm_root or cfg["mllm_root"], cfg.get("mllm_glob", ["prompt_h:2", "answer_h"]),
                        int(cfg.get("mllm_vis_layer", 1)), load_vis=bool(cfg.get("mllm_stream")))
        if cfg.get("tempo_norm", 0) > 0:
            from temporac_cvpr.train import attach_tempo, tempo_normalize

            attach_tempo(vids, cfg["rates"][0], Path(args.features) / f"tempo_global_r{cfg['rates'][0]}.json")
            vids = [tempo_normalize(v, cfg["tempo_norm"], cfg["tn_fmax"]) for v in vids]
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["state"])
        preds = predict(model, vids, cfg, device)
        res = {"split": args.split, "n": len(vids), "metrics": score(preds, vids), "predictions": preds,
               "best_epoch": ck.get("epoch"), "ckpt": args.ckpt}
        if cfg.get("shift_eval"):  # 2x-tempo view and the two-view average, as in training-time evaluation
            from temporac_cvpr.train import shift_video

            sp = predict(model, [shift_video(v, cfg["rates"]) for v in vids], cfg, device)
            res["predictions_shift"] = sp
            res["metrics_shift"] = score(sp, vids)
            res["metrics_tta"] = score({k: 0.5 * (preds[k] + sp[k]) for k in sp}, vids)
        tag = args.tag or args.split
        (Path(run) / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
        print(Path(run).name, tag, json.dumps(res["metrics"]), flush=True)


if __name__ == "__main__":
    main()
