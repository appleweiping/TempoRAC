"""Train/evaluate a counting head on frozen multi-rate tokens.

Model selection uses the validation split only; the test split is scored once with
the selected checkpoint and all per-video predictions are written to disk.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from temporac_cvpr.data import BASE, Video, batch_loader, load_videos, to_device
from temporac_cvpr.metrics import count_metrics
from temporac_cvpr.models import build_model

DEFAULT = {
    "head": "temporac", "rates": [1, 2, 4], "d": 256, "layers": 2, "heads": 8, "conv_k": 5, "dropout": 0.1,
    "spatial": "attn", "pool_to": 7, "router": "soft", "use_tempogram": True, "router_temp": 1.0,
    "share_experts": False, "max_lag": 160, "tempo_win": 48, "detach_tempo": True, "fuse_layers": 1,
    "tsm_lag": 96, "tsm_heads": 4, "n_period_bins": 64,
    "epochs": 80, "batch": 8, "lr": 2e-4, "wd": 0.05, "warmup": 3, "grad_clip": 1.0,
    "w_density": 100.0, "w_count": 1.0, "w_period": 0.5, "sigma_frac": 0.25, "density_kind": "gauss",
    "crop_prob": 0.5, "crop_min": 0.6, "eval_every": 1,
}


def crop_video(v: Video, rates: list[int], frac_min: float, rng: random.Random) -> Video:
    unit = BASE * max(rates)  # crop boundaries on a grid shared by all rates
    n_units = v.T // unit
    if n_units < 4:
        return v
    length = max(2, int(rng.uniform(frac_min, 1.0) * n_units))
    start = rng.randint(0, n_units - length)
    f0, f1 = start * unit, (start + length) * unit
    feats = {}
    for s in rates:
        c0, c1 = f0 // (BASE * s), f1 // (BASE * s)
        feats[s] = v.feats[s][c0:c1]
    cyc = v.cycles.copy()
    if len(cyc):
        cyc = cyc - f0  # density outside [0, f1-f0) is truncated by the target grid
    nv = copy.copy(v)
    nv.feats, nv.T, nv.cycles = feats, f1 - f0, cyc
    nv.count = float("nan")
    return nv


def make_batches(vids: list[Video], bs: int, rng: random.Random, shuffle: bool) -> list[list[Video]]:
    order = sorted(range(len(vids)), key=lambda i: vids[i].T)
    if shuffle:  # length bucketing with jitter
        order = sorted(order, key=lambda i: vids[i].T * rng.uniform(0.85, 1.15))
    batches = [[vids[i] for i in order[k:k + bs]] for k in range(0, len(order), bs)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def compute_loss(out: dict, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    mask = batch["mask"].float()
    pred, gt = out["density"], batch["density"]
    l_den = (((pred - gt) ** 2) * mask).sum() / mask.sum()
    cp, cg = pred.sum(1), gt.sum(1)
    l_cnt = (torch.abs(cp - cg) / (cg + 1.0)).mean()
    loss = cfg["w_density"] * l_den + cfg["w_count"] * l_cnt
    logs = {"l_den": float(l_den), "l_cnt": float(l_cnt)}
    if "period_logits" in out:  # RepNet-style baseline needs its own period supervision
        bins = None
        logits = out["period_logits"]
        nb = logits.shape[-1]
        lo, hi = math.log(2.0), math.log(2.0 * cfg["tsm_lag"])
        tgt = ((torch.log(batch["period"].clamp_min(2.0)) - lo) / (hi - lo) * (nb - 1)).round().long().clamp(0, nb - 1)
        on = batch["periodic"] * mask
        l_per = (F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none") * on).sum() / on.sum().clamp_min(1)
        l_on = (F.binary_cross_entropy(out["periodic_prob"].clamp(1e-4, 1 - 1e-4), batch["periodic"], reduction="none")
                * mask).sum() / mask.sum()
        loss = loss + cfg["w_period"] * (l_per + l_on)
        logs.update({"l_per": float(l_per), "l_on": float(l_on)})
    return loss, logs


@torch.no_grad()
def predict(model, vids: list[Video], cfg: dict, device) -> dict:
    model.eval()
    preds = {}
    batches = make_batches(vids, 4, random.Random(0), shuffle=False)
    for b, batch in zip(batches, batch_loader(batches, cfg["rates"], cfg["sigma_frac"], cfg["density_kind"],
                                              cfg.get("workers", 6))):
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        cnt = out["density"].float().sum(1).cpu().numpy()
        for v, c in zip(b, cnt):
            preds[v.key] = float(c)
    return preds


def score(preds: dict, vids: list[Video]) -> dict:
    p = [preds[v.key] for v in vids]
    g = [v.count for v in vids]
    return count_metrics(p, g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="{}", help="JSON overrides")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--with-test", action="store_true",
                    help="score the held-out test split (only for frozen final configurations)")
    args = ap.parse_args()
    args.no_test = not args.with_test

    import resource  # one memory-mapped file per video and rate

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 16) if hard > 0 else 1 << 16, hard))
    cfg = dict(DEFAULT)
    cfg.update(json.loads(args.cfg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")

    manifest = []
    for m in args.manifest:
        manifest += json.loads(Path(m).read_text())
    splits = {args.train_split, args.val_split} | (set() if args.no_test else {args.test_split})
    t0 = time.time()
    vids = load_videos(manifest, args.features, cfg["rates"], splits, cfg["pool_to"])
    train = [v for v in vids if v.split == args.train_split]
    val = [v for v in vids if v.split == args.val_split]
    test = [v for v in vids if v.split == args.test_split]
    print(f"loaded train {len(train)} val {len(val)} test {len(test)} in {time.time() - t0:.0f}s", flush=True)

    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per_ep = math.ceil(len(train) / cfg["batch"])
    total = cfg["epochs"] * steps_per_ep
    warm = cfg["warmup"] * steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / total))))
    (out / "config.json").write_text(json.dumps({"cfg": cfg, "args": vars(args), "params": n_par}, indent=1))
    log = open(out / "log.jsonl", "a")
    best, best_state, best_ep = None, None, -1
    for ep in range(cfg["epochs"]):
        model.train()
        t1 = time.time()
        agg = {}
        batches = [[crop_video(v, cfg["rates"], cfg["crop_min"], rng) if rng.random() < cfg["crop_prob"] else v
                    for v in b] for b in make_batches(train, cfg["batch"], rng, shuffle=True)]
        for batch in batch_loader(batches, cfg["rates"], cfg["sigma_frac"], cfg["density_kind"], cfg.get("workers", 6)):
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(batch)
            o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}
            loss, lg = compute_loss(o, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            for k, v in lg.items():
                agg[k] = agg.get(k, 0.0) + v / steps_per_ep
        rec = {"epoch": ep, "time": round(time.time() - t1, 1), **{k: round(v, 5) for k, v in agg.items()}}
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            vm = score(predict(model, val, cfg, device), val)
            rec["val"] = vm
            key = (vm["mae"], -vm["obo"])
            if best is None or key < best:
                best, best_ep = key, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: rec[k] for k in rec if k != "val"}), "val", {k: round(rec["val"][k], 4) for k in ("mae", "obo")} if "val" in rec else "", flush=True)

    torch.save({"state": best_state, "cfg": cfg, "epoch": best_ep}, out / "best.pt")
    model.load_state_dict(best_state)
    res = {"best_epoch": best_ep, "params": n_par}
    vp = predict(model, val, cfg, device)
    res["val"] = score(vp, val)
    preds = {"val": vp}
    if test:
        tp = predict(model, test, cfg, device)
        res["test"] = score(tp, test)
        preds["test"] = tp
    (out / "predictions.json").write_text(json.dumps(preds, indent=1))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
