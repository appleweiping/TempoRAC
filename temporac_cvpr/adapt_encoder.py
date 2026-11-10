"""Count-aware LoRA adaptation of the ST-MAE encoder, before the token pyramid is extracted.

Why: in CountLLM (CVPR 2025, Table 7, RepCount val) LoRA on the video encoder alone takes the frozen
setting from 0.409 to 0.239 MAE -- the largest single gain in their model -- and Context (CVPR 2020)
drops from 0.147 to 0.653 on UCFRep when its encoder is frozen.  ESCounts and TempoRAC use the same
frozen ST-MAE and sit at the same UCFRep level (~0.21).

How (end-to-end training through the full overlapping pyramid is too expensive on two GPUs):
each training video contributes one window of K=4 non-overlapping 16-frame clips at a random stride
s in {2,4,8} (all pyramid rates are adapted); frames are decoded on the fly (shorter side 224, centre
crop, random horizontal flip).  LoRA (rank 8) on q/k/v/proj of all 12 blocks; a small counting head
(attention pooling over space, 2-layer temporal transformer with a stride embedding, softplus density
per token) predicts, for every temporal token (2s frames), the integral of the annotated density over
its frames.  The epoch is selected on validation windows (mean relative window-count error, averaged
over datasets).  The LoRA factors are merged and saved in the original checkpoint format
(``model_state``), so ``extract_features.py --ckpt`` re-extracts the whole pyramid with it.
Usage: python -m temporac_cvpr.adapt_encoder OUT_DIR [--epochs 15 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from temporac_cvpr.data import BASE, cycles_from_bounds, density_target
from temporac_cvpr.vit_encoder import load_stmae_vitb

R = Path("/root/autodl-tmp/TempoRAC")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
STEM = {"RepCountA": "RepCountA", "Countix": "Countix", "UCFRep": "UCFRep"}


class EncLoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale

    def merged(self) -> torch.Tensor:
        return self.base.weight.data + (self.B.data @ self.A.data) * self.scale


def add_enc_lora(enc: nn.Module, r: int) -> list:
    out = []
    for blk in enc.blocks:
        for name in ("q", "k", "v", "proj"):
            lo = EncLoRA(getattr(blk.attn, name), r)
            setattr(blk.attn, name, lo)
            out.append((blk.attn, name, lo))
    for p in enc.parameters():
        if not any(p is lo.A or p is lo.B for _, _, lo in out):
            p.requires_grad_(False)
    return out


class CountHead(nn.Module):
    def __init__(self, d: int = 256, rates=(2, 4, 8)):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(768, d)
        self.score = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.rate = nn.Embedding(len(rates), d)
        self.rate_index = {s: i for i, s in enumerate(rates)}
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.out[-1].bias, -3.0)

    def forward(self, tok: torch.Tensor, s: int, n_tok: int) -> torch.Tensor:
        # tok: (B*K, 8, 14, 14, 768) -> (B, K*8) density per temporal token
        BK = tok.shape[0]
        x = self.proj(self.norm(tok.flatten(2, 3)))  # BK 8 196 d
        a = torch.softmax(self.score(x).squeeze(-1), -1)  # attention pooling over space
        x = (a.unsqueeze(-1) * x).sum(2).reshape(-1, n_tok, x.shape[-1])  # B, K*8, d
        pos = torch.arange(n_tok, device=x.device).float()
        div = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device).float() * (-math.log(1e4) / x.shape[-1]))
        pe = torch.zeros(n_tok, x.shape[-1], device=x.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos[:, None] * div), torch.cos(pos[:, None] * div)
        x = x + pe + self.rate(torch.tensor([self.rate_index[s]], device=x.device))
        return F.softplus(self.out(self.enc(x)).squeeze(-1))


def index_items(splits: set) -> list:
    man = json.loads((R / "data/UNION/manifest_dev.json").read_text())
    items = []
    for it in man:
        if it["split"] not in splits:
            continue
        fs = it.get("fsplit") or it["split"]
        meta = json.loads((R / "features" / f"{STEM[it['dataset']]}_stmae_p4" / "meta" /
                           f"{fs}__{it['video_id']}.json").read_text())
        T = int(meta["frames_decoded"])
        n_ann = it.get("num_frames_csv")
        scale = T / float(n_ann) if (n_ann and abs(n_ann - T) > 1) else 1.0
        items.append({"file": it["file"], "a": int(it.get("frame_range", [0, 0])[0]), "T": T,
                      "cycles": cycles_from_bounds(it.get("bounds", []), scale), "dataset": it["dataset"],
                      "split": it["split"], "key": f"{fs}__{it['video_id']}"})
    return items


class WindowSet(torch.utils.data.Dataset):
    """One window of K clips at stride s per video and epoch (fixed start / stride for validation)."""

    def __init__(self, items, K: int = 4, rates=(2, 4, 8), train: bool = True, seed: int = 0):
        self.items, self.K, self.rates, self.train, self.seed = items, K, rates, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import decord

        it = self.items[i]
        rng = random.Random((self.seed, self.epoch, i).__hash__()) if self.train else random.Random(i)
        s = rng.choice(self.rates) if self.train else 4
        K = max(1, min(self.K, it["T"] // (16 * s)))
        L = K * 16 * s
        start = rng.randrange(0, max(1, it["T"] - L + 1)) if self.train else max(0, (it["T"] - L) // 2)
        start -= start % (BASE * s)  # tokens align with whole cells of the 2-frame grid
        idx = [min(start + s * f, it["T"] - 1) for f in range(K * 16)]
        vr = decord.VideoReader(it["file"], ctx=decord.cpu(0), num_threads=1)
        h0, w0 = vr[0].shape[:2]
        sc = 224 / min(h0, w0)
        nh, nw = max(224, round(h0 * sc)), max(224, round(w0 * sc))
        vr = decord.VideoReader(it["file"], ctx=decord.cpu(0), width=nw, height=nh, num_threads=1)
        fr = vr.get_batch([it["a"] + j for j in idx]).asnumpy()
        top, left = (nh - 224) // 2, (nw - 224) // 2
        fr = np.ascontiguousarray(fr[:, top:top + 224, left:left + 224])
        if self.train and rng.random() < 0.5:
            fr = np.ascontiguousarray(fr[:, :, ::-1])
        # target: integral of the annotated density over each token's 2s frames (= s base cells)
        d = density_target(it["cycles"], it["T"])
        c0 = start // BASE
        tgt = np.array([d[c0 + k * s: c0 + (k + 1) * s].sum() for k in range(K * 8)], dtype=np.float32)
        return torch.from_numpy(fr), torch.from_numpy(tgt), s, K, it["dataset"]


def collate_windows(batch):
    # group by (stride, K) so a batch has one shape; the sampler below guarantees a single group
    fr = torch.stack([b[0] for b in batch])
    tg = torch.stack([b[1] for b in batch])
    return fr, tg, batch[0][2], batch[0][3], [b[4] for b in batch]


def run_batches(model, head, fr, s, K, device):
    B = fr.shape[0]
    clip = fr.to(device, non_blocking=True).view(B * K, 16, 224, 224, 3).permute(0, 4, 1, 2, 3).float().div_(255.0)
    clip = (clip - MEAN.to(device)) / STD.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tok = model(clip)
        return head(tok, s, K * 8).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: videos per split")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    train = index_items({"rc_train", "cx_train", "ucf_train"})
    val = index_items({"rc_valid", "cx_val", "ucf_dev"})
    if args.limit:
        train = [x for ds in STEM for x in [y for y in train if y["dataset"] == ds][: args.limit]]
        val = [x for ds in STEM for x in [y for y in val if y["dataset"] == ds][: args.limit]]
    enc, rep = load_stmae_vitb(str(R / "pretrained/VIT_B_16x4_MAE_PT.pyth"))
    loras = add_enc_lora(enc, args.rank)
    enc.to(device)
    blocks = enc.blocks  # gradient checkpointing per transformer block

    def enc_forward(x):
        x = enc.patch_embed(x)
        b, c, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = enc.pos_embed_spatial.repeat(1, t, 1) + torch.repeat_interleave(enc.pos_embed_temporal, h * w, dim=1)
        pos = torch.cat([enc.pos_embed_class, pos], dim=1)
        x = torch.cat([enc.cls_token.expand(b, -1, -1), x], dim=1) + pos
        for blk in blocks:
            x = ckpt.checkpoint(blk, x, use_reentrant=False)
        x = enc.norm(x)
        return x[:, 1:].reshape(b, t, h, w, c)

    head = CountHead().to(device)
    params = [p for _, _, lo in loras for p in (lo.A, lo.B)] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    print(f"train {len(train)} val {len(val)} lora params {sum(p.numel() for _, _, lo in loras for p in (lo.A, lo.B))}",
          flush=True)
    tset, vset = WindowSet(train, train=True, seed=args.seed), WindowSet(val, train=False)

    def loader(ds, shuffle):
        # batches must share (stride, K): sample a stride per batch by grouping indices after drawing them
        groups = {}
        for i in range(len(ds)):
            it = ds.items[i]
            rng = random.Random((ds.seed, ds.epoch, i).__hash__()) if ds.train else random.Random(i)
            s = rng.choice(ds.rates) if ds.train else 4
            K = max(1, min(ds.K, it["T"] // (16 * s)))
            groups.setdefault((s, K), []).append(i)
        batches = []
        for g in groups.values():
            if shuffle:
                random.shuffle(g)
            batches += [g[k:k + args.batch] for k in range(0, len(g), args.batch)]
        if shuffle:
            random.shuffle(batches)
        return torch.utils.data.DataLoader(ds, batch_sampler=batches, collate_fn=collate_windows,
                                           num_workers=args.workers, prefetch_factor=2)

    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / steps))))
    best, log = None, open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        tset.epoch = ep
        enc.train()
        head.train()
        t0, tot, n = time.time(), 0.0, 0
        for fr, tg, s, K, _ in loader(tset, True):
            pred = run_batches(enc_forward, head, fr, s, K, device)
            tg = tg.to(device)
            loss = 100 * F.mse_loss(pred, tg) + (pred.sum(1) - tg.sum(1)).abs().div(tg.sum(1) + 1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot, n = tot + float(loss), n + 1
        enc.eval()
        head.eval()
        errs = {}
        with torch.no_grad():
            for fr, tg, s, K, dss in loader(vset, False):
                pred = run_batches(enc_forward, head, fr, s, K, device)
                e = (pred.sum(1).cpu() - tg.sum(1)).abs() / (tg.sum(1) + 0.1)
                for ds, x in zip(dss, e.tolist()):
                    errs.setdefault(ds, []).append(x)
        val_err = float(np.mean([np.mean(v) for v in errs.values()]))
        rec = {"epoch": ep, "loss": round(tot / max(n, 1), 4), "val_window_rel_err": round(val_err, 4),
               **{f"val_{k}": round(float(np.mean(v)), 4) for k, v in errs.items()},
               "time": round(time.time() - t0, 1), "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if best is None or val_err < best:
            best = val_err
            state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()
                     if not k.endswith((".A", ".B")) and ".base." not in k}
            for attn, name, lo in loras:  # merged weights under the original parameter names
                prefix = [k for k, m in enc.named_modules() if m is attn][0]
                state[f"{prefix}.{name}.weight"] = lo.merged().cpu().clone()
                state[f"{prefix}.{name}.bias"] = lo.base.bias.data.cpu().clone()
            torch.save({"model_state": state, "epoch": ep, "val_window_rel_err": val_err, "rank": args.rank},
                       out / "stmae_adapted.pth")
    print("ADAPT_DONE best", best, flush=True)


if __name__ == "__main__":
    main()
