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

from temporac_cvpr.data import BASE, Video, attach_desc, attach_mllm, batch_loader, load_videos, to_device
from temporac_cvpr.metrics import count_metrics
from temporac_cvpr.models import build_model

DEFAULT = {
    "head": "temporac", "rates": [1, 2, 4], "d": 256, "layers": 2, "heads": 8, "conv_k": 5, "dropout": 0.1,
    "spatial": "attn", "pool_to": 7, "router": "soft", "use_tempogram": True, "router_temp": 1.0,
    "share_experts": False, "max_lag": 160, "tempo_win": 48, "detach_tempo": True, "fuse_layers": 1,
    "tsm_lag": 96, "tsm_heads": 4, "n_period_bins": 64,
    "epochs": 80, "batch": 8, "lr": 2e-4, "wd": 0.05, "warmup": 3, "grad_clip": 1.0,
    "w_density": 100.0, "w_count": 1.0, "w_period": 0.5, "sigma_frac": 0.25, "density_kind": "gauss",
    "crop_prob": 0.5, "crop_min": 0.6, "eval_every": 1, "ema": 0.0, "workers": 4,
    "mix_level": "feature", "router_smooth": 0, "moe_loss": "none",
    "tg_bias": False, "tg_input": False, "warp_prob": 0.0, "warp_vmax": 1.5,
    "tc_decode": False, "tc_win": 24, "periodic_pool": False, "pp_max_lag": 80, "pp_win": 24,
    "tempo_gate": False, "gate_temp": 0.5, "tempo_norm": 0, "tn_fmax": 2.0,
    "period_head": False, "matched_gate": False, "tokens_per_cycle": 8.0, "gate_sigma": 1.0,
    "bias_kind": "tempogram", "tempo_src": "finest", "rate_fusion": "concat", "rate_dropout": 0.0,
    "shift_prob": 0.0, "shift_min_period": 8.0, "shift_eval": False,
    "flip_root": "", "flip_prob": 0.0, "flip_eval": False,
}


def load_rates(cfg: dict) -> list[int]:
    """Rates read from disk: the model's pyramid, plus one coarser level per rate when the
    pyramid-shift (exact 2x tempo) view is used for training or evaluation."""
    rates = set(cfg["rates"])
    if cfg.get("shift_prob", 0) > 0 or cfg.get("shift_eval"):
        rates |= {2 * s for s in cfg["rates"]}
    return sorted(rates)


def shift_video(v: Video, rates: list[int]) -> Video:
    """Exact token pyramid of the video replayed twice as fast.

    A rate-s clip of V' = V[0::2] starting at frame j reads V[2j + 2s f], i.e. the rate-2s
    clip of V starting at 2j; clip hops (4s vs 8s raw frames) and grid cells (2s vs 4s raw
    frames) correspond one to one, so the rate-s stream of V' *is* the rate-2s stream of V.
    Cycle bounds halve; the count is unchanged.
    """
    nv = copy.copy(v)
    nv.feats = {s: v.feats[2 * s] for s in rates}
    nv.T = math.ceil(v.T / 2)
    nv.cycles = v.cycles / 2.0 if len(v.cycles) else v.cycles.copy()
    if v.mllm_pos is not None:
        nv.mllm_pos = v.mllm_pos / 2.0
    return nv


def can_shift(v: Video, min_period: float) -> bool:
    """Only accelerate when every cycle stays >= min_period frames (no aliasing at the finest rate)."""
    if not len(v.cycles):
        return True
    p = np.abs(v.cycles[:, 1] - v.cycles[:, 0])
    p = p[p > 0]
    return (not len(p)) or float(p.min()) / 2.0 >= min_period


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
    if v.mllm_pos is not None:  # semantic tokens keep their content; only their frame positions move
        nv.mllm_pos = v.mllm_pos - f0
    nv.count = float("nan")
    return nv


def warp_video(v: Video, rates: list[int], rng: random.Random, vmax: float) -> Video:
    """Tempo-warp training augmentation applied consistently to every rate stream.

    Three source segments are replayed at speeds log-uniform in [1/vmax, vmax]; each
    rate's token sequence is resampled (linear interpolation between cells) at the
    warped cell centres, and annotated cycle bounds are mapped through the same
    piecewise-linear time map, so the repetition count is unchanged.
    """
    T = v.T
    if T < 64:
        return v
    c = sorted(rng.uniform(0.15, 0.85) for _ in range(2))
    src = np.array([0.0, c[0] * T, c[1] * T, float(T)])
    sp = [math.exp(rng.uniform(-math.log(vmax), math.log(vmax))) for _ in range(3)]
    out = np.concatenate([[0.0], np.cumsum(np.diff(src) / np.array(sp))])
    T2 = max(16, int(round(out[-1])))
    out = out * (T2 / out[-1])
    nv = copy.copy(v)
    # the (cheap) parameters are drawn here; the resampling itself runs in the collate workers
    nv.warp = (src, out, T2)
    nv.count = float("nan")
    return nv


def tempo_normalize(v: Video, K: float, fmax: float) -> Video:
    """TempoNorm: uniformly resample every stream so one (label-free) period spans ~K
    finest-stream cells.  Deterministic, applied identically in training and inference;
    the count is unchanged (cycle bounds follow the same linear time map)."""
    p = getattr(v, "tempo_p", float("nan"))
    if not np.isfinite(p) or p <= 0 or getattr(v, "warp", None) is not None:
        return v
    f = float(np.clip(K / p, 1.0 / fmax, fmax))
    T2 = max(16, int(round(v.T * f)))
    nv = copy.copy(v)
    nv.warp = (np.array([0.0, float(v.T)]), np.array([0.0, float(T2)]), T2)
    return nv


def attach_tempo(vids: list[Video], rate: int, cache: Path) -> None:
    from temporac_cvpr.data import global_period

    store = json.loads(cache.read_text()) if cache.exists() else {}
    changed = False
    for v in vids:
        k = f"{v.key}|r{rate}|T{v.T}"
        if k not in store:
            store[k] = global_period(v.feats[rate])
            changed = True
        v.tempo_p = store[k]
    if changed:
        try:
            cache.write_text(json.dumps(store))
        except OSError:
            pass


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
    if cfg.get("moe_loss", "none") == "expected" and "expert_density" in out:
        # competitive mixture-of-experts objective (Jacobs et al., 1991): the routing-weighted
        # expected squared error of the rate experts; experts specialise and the router is pushed
        # towards the locally most accurate rate.  Replaces (does not add to) the density term.
        err = (out["expert_density"] - gt.unsqueeze(-1)) ** 2  # B G S
        l_den = ((out["route"] * err).sum(-1) * mask).sum() / mask.sum()
    else:
        l_den = (((pred - gt) ** 2) * mask).sum() / mask.sum()
    cp, cg = pred.sum(1), gt.sum(1)
    l_cnt = (torch.abs(cp - cg) / (cg + 1.0)).mean()
    loss = cfg["w_density"] * l_den + cfg["w_count"] * l_cnt
    logs = {"l_den": float(l_den), "l_cnt": float(l_cnt)}
    if cfg.get("w_count_abs", 0) > 0:  # absolute count term (Sight&Sound, App. D): high counts get their weight
        l_abs = F.smooth_l1_loss(cp, cg, beta=1.0)
        loss = loss + cfg["w_count_abs"] * l_abs
        logs["l_cnt_abs"] = float(l_abs)
    if "ans_loss" in out:  # LLM answer tokens (caption in the alignment stage, decimal count afterwards)
        loss = loss + cfg.get("w_ans", 0.1) * out["ans_loss"]
        logs["l_ans"] = float(out["ans_loss"])
    if "period_logits" in out:  # RepNet-style baseline needs its own period supervision
        bins = None
        logits = out["period_logits"]
        nb = logits.shape[-1]
        lo, hi = math.log(2.0), math.log(2.0 * cfg["tsm_lag"])
        tgt = ((torch.log(batch["period"].clamp_min(2.0)) - lo) / (hi - lo) * (nb - 1)).round().long().clamp(0, nb - 1)
        on = batch["periodic"] * mask
        l_per = (F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none") * on).sum() / on.sum().clamp_min(1)
        logs["l_per"] = float(l_per)
        if "periodic_prob" in out:  # RepNet-style baseline: periodicity classification as well
            l_on = (F.binary_cross_entropy(out["periodic_prob"].clamp(1e-4, 1 - 1e-4), batch["periodic"],
                                           reduction="none") * mask).sum() / mask.sum()
            l_per = l_per + l_on
            logs["l_on"] = float(l_on)
        loss = loss + cfg["w_period"] * l_per
    return loss, logs


class EvalLoader:
    """Fixed evaluation batches with persistent workers (built once per split)."""

    def __init__(self, vids: list[Video], cfg: dict, workers: int = 4):
        from temporac_cvpr.data import BatchSet

        self.batches = make_batches(vids, 4, random.Random(0), shuffle=False)
        self.loader = torch.utils.data.DataLoader(
            BatchSet(self.batches, cfg["rates"], cfg["sigma_frac"], cfg["density_kind"]), batch_size=None,
            num_workers=workers, pin_memory=True, persistent_workers=workers > 0, prefetch_factor=2 if workers else None)


@torch.no_grad()
def predict(model, vids: list[Video], cfg: dict, device, loader: EvalLoader | None = None) -> dict:
    model.eval()
    preds = {}
    loader = loader or EvalLoader(vids, cfg, cfg.get("workers", 6))
    for b, batch in zip(loader.batches, loader.loader):
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
    srng = random.Random(args.seed + 7919)  # pyramid-shift augmentation stream
    frng = random.Random(args.seed + 104729)  # mirrored-view augmentation stream
    device = torch.device("cuda")

    manifest = []
    for m in args.manifest:
        manifest += json.loads(Path(m).read_text())
    train_splits = set(args.train_split.split(","))  # several = joint training on the union of training sets
    # several validation splits = one training run, one selected checkpoint per target dataset
    val_names = args.val_split.split(",")
    test_names = args.test_split.split(",")
    if len(val_names) > 1 and not args.no_test:
        assert len(test_names) == len(val_names), "one test split per validation split"
    # Test splits are loaded only after training: every video, rate and encoder depth is its own memory
    # map, and train+val+test of the three-dataset union exceeds the per-process map limit (65530).
    splits = train_splits | set(val_names)
    t0 = time.time()
    lrates = load_rates(cfg)
    aux = tuple(cfg.get("aux_features", ()))
    vids = load_videos(manifest, args.features, lrates, splits, cfg["pool_to"], aux)
    train = [v for v in vids if v.split in train_splits]
    if cfg.get("repeat"):  # balanced multi-source sampling, e.g. {"RepCountA": 4, "UCFRep": 4}
        train = [v for v in train for _ in range(int(cfg["repeat"].get(v.dataset, 1)))]
    vals = {n: [v for v in vids if v.split == n] for n in val_names}
    tests = {}
    val = vals[val_names[0]]
    print(f"loaded train {len(train)} val {[len(x) for x in vals.values()]} in {time.time() - t0:.0f}s", flush=True)
    cfg["in_dim"] = int(vids[0].feats[cfg["rates"][0]].shape[-1])  # 768 ST-MAE ViT-B, 1024 V-JEPA 2 ViT-L
    flip_map = {}
    if cfg.get("flip_root"):  # mirrored view of every video (same keys and frame grid)
        assert not cfg.get("aux_features"), "mirrored view not available for auxiliary encoders"
        flip_map = {v.key: v for v in load_videos(manifest, cfg["flip_root"], lrates, splits, cfg["pool_to"])}
        assert all(v.key in flip_map and flip_map[v.key].T == v.T for v in vids)
    if cfg.get("domains"):  # e.g. {"RepCountA": 0, "Countix": 1, "UCFRep": 2}
        cfg["n_domains"] = max(cfg["domains"].values()) + 1

    def prepare(vlist: list) -> None:
        """Frozen video-MLLM features (one npz per video) and dataset-token indices."""
        if cfg.get("mllm_root"):
            gk, vl = cfg.get("mllm_glob", ["prompt_h:2", "answer_h"]), int(cfg.get("mllm_vis_layer", 1))
            attach_mllm(vlist, cfg["mllm_root"], gk, vl, load_vis=bool(cfg.get("mllm_stream")))
        if cfg.get("domains"):
            for v in vlist:
                v.domain = cfg["domains"][v.dataset]
        if cfg.get("llm_desc_roots"):  # per-video action / one-repetition text for the LLM prompt
            attach_desc(vlist, cfg["llm_desc_roots"])

    prepare(vids + list(flip_map.values()))
    if cfg.get("mllm_root"):
        cfg["mllm_glob_dim"] = int(len(vids[0].mllm_glob))
        if cfg.get("mllm_stream"):
            cfg["mllm_vis_dim"] = int(vids[0].mllm_vis.shape[1])
    if cfg.get("tempo_norm", 0) > 0:  # label-free period per video, then deterministic resampling
        attach_tempo(vids, cfg["rates"][0], Path(args.features) / f"tempo_global_r{cfg['rates'][0]}.json")
        vals = {n: [tempo_normalize(v, cfg["tempo_norm"], cfg["tn_fmax"]) for v in vs] for n, vs in vals.items()}
        val = vals[val_names[0]]

    model = build_model(cfg).to(device)
    if cfg.get("init_from"):  # a trained checkpoint (e.g. a TempoRAC counter as the LLM model's decoder)
        ck0 = torch.load(cfg["init_from"], map_location="cpu", weights_only=True)
        st = ck0["state"]
        target = model
        if hasattr(model, "base") and not any(k.startswith("base.") for k in st):
            target = model.base  # plain TempoRAC checkpoint -> the decoder inside TempoRAC-LLM
        missing, unexpected = target.load_state_dict(st, strict=False)
        print("init_from", cfg["init_from"], "into", type(target).__name__, "missing", missing,
              "unexpected", unexpected, flush=True)
    if hasattr(model, "base") and cfg.get("freeze_base"):
        # adapter-style LLM stage: the converged TempoRAC decoder stays fixed; only the projectors and the
        # LoRA factors learn, and the zero-initialised LLM residual starts from the decoder's own output
        for p in model.base.parameters():
            p.requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, "base") and cfg.get("freeze_base"):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["wd"])
    elif hasattr(model, "base") and cfg.get("lr_base"):  # pretrained decoder at a lower learning rate
        base_ids = {id(p) for p in model.base.parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) in base_ids], "lr": cfg["lr_base"]},
                  {"params": [p for p in model.parameters() if id(p) not in base_ids and p.requires_grad]}]
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per_ep = math.ceil(len(train) / cfg["batch"])
    total = cfg["epochs"] * steps_per_ep
    warm = cfg["warmup"] * steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / total))))
    (out / "config.json").write_text(json.dumps({"cfg": cfg, "args": vars(args), "params": n_par}, indent=1))
    log = open(out / "log.jsonl", "a")
    has_val = any(vals[n] for n in val_names)
    bests = {n: [None, None, -1] for n in val_names}  # selection key, state, epoch per target
    val_loaders = {n: EvalLoader(vs, cfg, 2) for n, vs in vals.items() if vs}
    ema = None
    if cfg.get("ema", 0) > 0:  # exponential moving average of weights, evaluated instead of raw weights
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    for ep in range(cfg["epochs"]):
        model.train()
        if hasattr(model, "base") and cfg.get("freeze_base"):
            model.base.eval()  # frozen decoder: no dropout, deterministic features for the LLM path
        t1 = time.time()
        agg = {}
        batches = make_batches(train, cfg["batch"], rng, shuffle=True)
        if cfg.get("flip_prob", 0) > 0:  # independent stream, applied before crops and shifts
            batches = [[flip_map[v.key] if frng.random() < cfg["flip_prob"] else v for v in b] for b in batches]
        batches = [[crop_video(v, lrates, cfg["crop_min"], rng) if rng.random() < cfg["crop_prob"] else v
                    for v in b] for b in batches]
        if cfg.get("shift_prob", 0) > 0:  # independent stream: crops identical with/without shifting
            batches = [[shift_video(v, cfg["rates"]) if (srng.random() < cfg["shift_prob"]
                        and can_shift(v, cfg["shift_min_period"])) else v for v in b] for b in batches]
        if cfg.get("warp_prob", 0) > 0:
            wrng = random.Random(rng.random())  # separate stream: crops identical with/without warping
            batches = [[warp_video(v, cfg["rates"], wrng, cfg["warp_vmax"]) if wrng.random() < cfg["warp_prob"]
                        else v for v in b] for b in batches]
        if cfg.get("tempo_norm", 0) > 0:
            batches = [[tempo_normalize(v, cfg["tempo_norm"], cfg["tn_fmax"]) for v in b] for b in batches]
        for batch in batch_loader(batches, cfg["rates"], cfg["sigma_frac"], cfg["density_kind"], cfg.get("workers", 6)):
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(batch)
            o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}
            loss, lg = compute_loss(o, batch, cfg)
            if not torch.isfinite(loss):  # recorded and skipped; a run with many of these is a failed run
                agg["nonfinite_steps"] = agg.get("nonfinite_steps", 0.0) + 1.0
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1.0 - cfg["ema"])
            for k, v in lg.items():
                agg[k] = agg.get(k, 0.0) + v / steps_per_ep
        rec = {"epoch": ep, "time": round(time.time() - t1, 1), **{k: round(v, 5) for k, v in agg.items()},
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        if not has_val:  # fixed-epoch protocol (e.g. retraining on all training videos): keep the last weights
            evm = ema if ema is not None else model
            st = {k: v.detach().cpu().clone() for k, v in evm.state_dict().items()}
            for n in val_names:
                bests[n] = [None, st, ep]
        elif (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            evm = ema if ema is not None else model
            allm, st = {}, None
            for n in val_names:
                vm = score(predict(evm, vals[n], cfg, device, val_loaders[n]), vals[n])
                allm[n] = vm
                key = (vm["mae_p"], -vm["obo_p"])  # official positive-count protocol
                if bests[n][0] is None or key < bests[n][0]:
                    st = st or {k: v.detach().cpu().clone() for k, v in evm.state_dict().items()}
                    bests[n] = [key, st, ep]
            rec["val"] = allm[val_names[0]]
            if len(val_names) > 1:
                rec["val_all"] = allm
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: rec[k] for k in rec if k not in ("val", "val_all")}), "val",
              {k: round(rec["val"][k], 4) for k in ("mae_p", "obo_p", "mae", "obo")} if "val" in rec else "", flush=True)

    multi = len(val_names) > 1
    for n in val_names:  # checkpoints first, so a failure while mapping the test splits loses no training
        torch.save({"state": bests[n][1], "cfg": cfg, "epoch": bests[n][2], "target": n},
                   out / f"best{'_' + n if multi else ''}.pt")
    if not args.no_test:  # release the training videos' memory maps, then map the test splits
        import gc

        n_maps = lambda: sum(1 for _ in open("/proc/self/maps"))
        before = n_maps()
        keep = {v.key for vs in vals.values() for v in vs}
        flip_map = {k: v for k, v in flip_map.items() if k in keep}
        del train, batches, vids
        gc.collect()
        print(f"memory maps {before} -> {n_maps()} after releasing the training videos", flush=True)
        t0 = time.time()
        tvids = load_videos(manifest, args.features, lrates, set(test_names), cfg["pool_to"], aux)
        if cfg.get("flip_root"):
            flip_map.update({v.key: v for v in load_videos(manifest, cfg["flip_root"], lrates, set(test_names),
                                                           cfg["pool_to"])})
        prepare(tvids + [flip_map[v.key] for v in tvids if v.key in flip_map])
        if cfg.get("tempo_norm", 0) > 0:
            attach_tempo(tvids, cfg["rates"][0], Path(args.features) / f"tempo_global_r{cfg['rates'][0]}.json")
            tvids = [tempo_normalize(v, cfg["tempo_norm"], cfg["tn_fmax"]) for v in tvids]
        tests = {n: [v for v in tvids if v.split == n] for n in test_names}
        print(f"loaded test {[len(x) for x in tests.values()]} in {time.time() - t0:.0f}s", flush=True)

    # count-invariant test-time views: 2x tempo (exact pyramid shift) and mirror; raw per-view predictions
    # are kept so every view combination can be analysed; selection above used the original view only
    views = {}
    if cfg.get("shift_eval"):
        views["shift"] = lambda v: shift_video(v, cfg["rates"])
    if flip_map and cfg.get("flip_eval"):
        views["flip"] = lambda v: flip_map[v.key]
        if cfg.get("shift_eval"):
            views["flipshift"] = lambda v: shift_video(flip_map[v.key], cfg["rates"])
    multi = len(val_names) > 1
    summary = {}
    for i, n in enumerate(val_names):
        sfx = f"_{n}" if multi else ""
        _, best_state, best_ep = bests[n]
        torch.save({"state": best_state, "cfg": cfg, "epoch": best_ep, "target": n}, out / f"best{sfx}.pt")
        model.load_state_dict(best_state)
        val = vals[n]
        test = tests.get(test_names[i] if multi else test_names[0], []) if tests else []
        res = {"best_epoch": best_ep, "params": n_par}
        preds = {}
        if val:
            vp = predict(model, val, cfg, device, val_loaders[n])
            res["val"] = score(vp, val)
            preds["val"] = vp
        if test:
            tp = predict(model, test, cfg, device)
            res["test"] = score(tp, test)
            preds["test"] = tp
        for name, vs in (("val", val), ("test", test)):
            if not vs or not views:
                continue
            allp = [preds[name]]
            for vn, fn in views.items():
                vp = predict(model, [fn(v) for v in vs], cfg, device)
                preds[f"{name}_{vn}"] = vp
                res[f"{name}_{vn}"] = score(vp, vs)
                allp.append(vp)
            res[f"{name}_tta"] = score({k: float(np.mean([p[k] for p in allp])) for k in allp[0]}, vs)
        (out / f"predictions{sfx}.json").write_text(json.dumps(preds, indent=1))
        (out / f"result{sfx}.json").write_text(json.dumps(res, indent=1))
        summary[n] = res
        print("RESULT", n, json.dumps(res), flush=True)
    if multi:  # written last: its presence marks the run as complete for the launcher
        (out / "result.json").write_text(json.dumps({"targets": summary}, indent=1))


if __name__ == "__main__":
    main()
