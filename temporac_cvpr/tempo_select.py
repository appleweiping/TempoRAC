"""Language-guided tempo selection: a LoRA-tuned language model decides which tempo view to trust.

Why (validation evidence, view_tempo.py / view_class.py / view_knn.py): the density decoder counts
exact tempo-changed views of the token pyramid (0.5x: shifted down, needs the rate-1 stream; 1x;
2x: shifted up).  Each view is right in a different tempo regime -- the original view over-counts slow
repetitions (half cycles), the 2x view under-counts fast ones.  The model's own count cannot tell the
regimes apart (view_gate.py, negative), but the *action* can: a gate that only knows the action class,
estimated leave-one-out, beats the plain two-view average on all three benchmarks, and so does one
that only knows the MLLM's description.  Knowing what is repeated tells how fast it plausibly repeats.

What: the language model of Qwen3-VL-8B (LoRA r=16 on q/k/v/o) reads a text prompt -- the frozen
MLLM's description of the action and of one repetition, the duration of the counted span, and the K
candidate counts with the cycle duration each implies -- and answers with a letter.  The softmax over
the K letter tokens weights the views; the reported count is sum_k w_k c_k, still an integral of
decoder densities.  Training minimises the protocol error of that mixture, |sum_k w_k c_k - c|/(c+0.1),
so each example weighs as much as the choice matters.  Candidate order is random in training and
averaged over cyclic rotations at inference (no position bias).

How it is trained without leakage: on out-of-fold predictions (make_folds.py, runs/oof5): a stage-1
decoder never predicts its own training videos, otherwise every view is almost right and there is
nothing to learn (why refining a converged decoder on its own training videos gave ~0).  The epoch is
selected on the validation splits with the five stage-1 seeds' validation predictions.
Baselines in the same file: plain two-view average, kNN on the description embedding, an MLP gate.

Usage:
  python -m temporac_cvpr.tempo_select build OUT_DIR
  python -m temporac_cvpr.tempo_select knn   OUT_DIR [--views 1x,2x]
  python -m temporac_cvpr.tempo_select mlp   OUT_DIR [--views ...] [--seed 0]
  python -m temporac_cvpr.tempo_select llm   OUT_DIR [--views ...] [--no-desc] [--epochs 4] [--seed 0]
  python -m temporac_cvpr.tempo_select apply OUT_DIR NAME     # test / QUVA rows, once, after all decisions
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/TempoRAC")
SH = ["rc", "cx", "ucf"]
VAL = {"rc": "rc_valid", "cx": "cx_val", "ucf": "ucf_dev"}
TRN = {"rc": "rc_train", "cx": "cx_train", "ucf": "ucf_train"}
DOM = {"RepCountA": 0, "Countix": 1, "UCFRep": 2}
DESC = ["RepCountA_desc_bge", "UCFRep_desc_bge", "Countix_desc_bge", "QUVA_desc_bge"]
MAIN = "adapt_dev/wo_llm_adapted_s{}"
SEEDS = range(5)
LETTERS = "ABC"
SPEED = {"half": 0.5, "1x": 1.0, "2x": 2.0}


# ----------------------------------------------------------------------------------------------- data
def mae(p, g) -> float:
    p, g = np.asarray(p, float), np.asarray(g, float)
    return float(np.mean(np.abs(np.round(p) - g) / (g + 0.1)))


def obo(p, g) -> float:
    p, g = np.asarray(p, float), np.asarray(g, float)
    return float(np.mean(np.abs(np.round(p) - g) <= 1))


def views_of(E: dict, pre: str = "predictions") -> dict:
    """{view: {key: count}} from an eval_saved file or a training-time predictions file."""
    out = {}
    for name, v in ((pre, "1x"), (f"{pre}_shift", "2x"), (f"{pre}_slow", "half")):
        if name in E:
            out[v] = E[name]
    return out


def build(out: Path) -> None:
    man = {f"{it['split']}__{it['video_id']}": it for it in json.loads((R / "data/UNION/manifest_dev.json").read_text())}
    quva = {f"{it['split']}__{it['video_id']}": it for it in json.loads((R / "data/QUVA/manifest.json").read_text())}
    desc = {}
    for d in DESC:
        for line in (R / "features" / d / "descriptions.jsonl").read_text().splitlines():
            x = json.loads(line)
            desc[(d.split("_")[0], x["key"])] = (x["action"], x["repetition"])

    def row(key, it, cs: dict, role, seed, feat="UNION_stmaeA_b8_p4", dname=None):
        fkey = f"{it.get('fsplit', it['split'])}__{it['video_id']}"
        m = json.loads((R / "features" / feat / "meta" / f"{fkey}.json").read_text())
        ds = it.get("dataset") or "QUVA"
        act, rep = desc[(dname or ds, fkey)]
        return {"key": key, "fkey": fkey, "ds": ds, "dname": dname or ds, "gt": float(it["count"]),
                "c": {k: float(v) for k, v in cs.items()}, "dur": m["frames_decoded"] / float(m.get("fps") or 30.0),
                "act": act, "rep": rep, "role": role, "seed": seed}

    def add(V: dict, lookup, role, seed, rename=None, **kw):
        for k in V["1x"]:
            mk = rename(k) if rename else k
            rows.append(row(mk, lookup[mk], {v: V[v][k] for v in V}, role, seed, **kw))

    rows = []
    for f in range(5):
        d = R / "runs" / "oof5" / f"fold{f}_s0"
        if not (d / "result.json").exists():
            print("missing (not finished):", d.name)
            continue
        for s in SH:
            fe = d / f"eval_oof_{s}.json"
            V = views_of(json.loads(fe.read_text())) if fe.exists() else \
                views_of(json.loads((d / f"predictions_{VAL[s]}.json").read_text()), "test")
            add(V, man, "oof", f, rename=lambda k, s=s: f"{TRN[s]}__{k.split('__', 1)[1]}")
    for sd in SEEDS:
        d = R / "runs" / MAIN.format(sd)
        for s in SH:
            fe = d / f"eval_val_{s}.json"
            V = views_of(json.loads(fe.read_text())) if fe.exists() else \
                views_of(json.loads((d / f"predictions_{VAL[s]}.json").read_text()), "val")
            add(V, man, "val", sd)
            ft = d / f"eval_test_{s}.json"
            if ft.exists():
                add(views_of(json.loads(ft.read_text())), man, "test", sd)
        fq = d / "eval_quva.json"
        if fq.exists():
            add(views_of(json.loads(fq.read_text())), quva, "quva", sd, feat="QUVA_stmaeA_b8_p4", dname="QUVA")
    rows = [r for r in rows if r["gt"] > 0]  # protocol: positive counts
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("rows", Counter((r["role"], r["ds"], ",".join(sorted(r["c"]))) for r in rows))


def load_rows(out: Path, views: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if all(v in r["c"] for v in views)]
    for r in rows:
        r["cv"] = [r["c"][v] for v in views]
    return rows


def report(rows: list[dict], W: np.ndarray, name: str, role: str = "val") -> dict:
    """Per dataset: mean over seeds of MAE/OBO of the weighted views vs the plain 1x/2x average."""
    res = {}
    for ds in ("RepCountA", "Countix", "UCFRep", "QUVA"):
        idx = [i for i, r in enumerate(rows) if r["role"] == role and r["ds"] == ds]
        if not idx:
            continue
        per_m, per_a, per_o = [], [], []
        for sd in sorted({rows[i]["seed"] for i in idx}):
            ii = [i for i in idx if rows[i]["seed"] == sd]
            g = np.array([rows[i]["gt"] for i in ii])
            C = np.array([rows[i]["cv"] for i in ii])
            p = (W[ii] * C).sum(1)
            avg = np.array([0.5 * (rows[i]["c"]["1x"] + rows[i]["c"]["2x"]) for i in ii])
            per_m.append(mae(p, g))
            per_o.append(obo(p, g))
            per_a.append(mae(avg, g))
        res[ds] = {"mae": float(np.mean(per_m)), "obo": float(np.mean(per_o)), "mae_avg": float(np.mean(per_a)),
                   "rel": float(np.mean(per_m) / np.mean(per_a) - 1),
                   "mae_sd": float(np.std(per_m, ddof=1)) if len(per_m) > 1 else 0.0,
                   "obo_sd": float(np.std(per_o, ddof=1)) if len(per_o) > 1 else 0.0, "seeds": len(per_m)}
    rel = float(np.mean([v["rel"] for k, v in res.items() if k != "QUVA"])) if res else float("nan")
    print(f"{name:30s} [{role}] " + "  ".join(f"{k} {v['mae']:.4f}/{v['obo']:.3f} ({v['rel']:+.3f})" for k, v in res.items())
          + f"  | mean rel {rel:+.4f}", flush=True)
    return {"per_ds": res, "mean_rel": rel}


def errors(rows: list[dict]) -> np.ndarray:
    g = np.array([r["gt"] for r in rows])[:, None]
    return np.abs(np.round(np.array([r["cv"] for r in rows])) - g) / (g + .1)


def save(out: Path, name: str, rows: list[dict], W: np.ndarray, views: list[str], extra: dict) -> None:
    res = {"name": name, "views": views, **extra, "val": report(rows, W, name, "val")}
    (out / f"{name}.json").write_text(json.dumps({**res, "keys": [[r["role"], r["seed"], r["key"]] for r in rows],
                                                  "W": W.tolist()}))


# ------------------------------------------------------------------------------------- simple gates
def embeddings(rows: list[dict], field: str) -> np.ndarray:
    E = np.stack([np.load(R / "features" / f"{r['dname']}_desc_bge" / f"{r['fkey']}.npz")[field].astype(np.float32)
                  for r in rows])
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


def knn_weights(S: np.ndarray, k: int, Y: np.ndarray, stake: np.ndarray) -> np.ndarray:
    nn_ = np.argsort(-S, axis=1)[:, :k]
    sw = stake[nn_]  # (n, k)
    num = (sw[..., None] * Y[nn_]).sum(1)
    den = sw.sum(1, keepdims=True)
    K = Y.shape[1]
    return np.where(den > 0, num / np.maximum(den, 1e-8), 1.0 / K)


def run_knn(out: Path, views: list[str]) -> None:
    rows = load_rows(out, views)
    oof = np.array([r["role"] == "oof" for r in rows])
    E_ = errors(rows)
    best_v = E_.min(1, keepdims=True)
    Y = (E_ == best_v).astype(float)
    Y /= Y.sum(1, keepdims=True)
    stake = E_.max(1) - E_.min(1)
    Cv = np.array([r["cv"] for r in rows])
    g = np.array([r["gt"] for r in rows])
    best = None
    for field in ("desc_h", "rep_h", "act_h"):
        E = embeddings(rows, field)
        M = E[oof]
        S = M @ M.T
        np.fill_diagonal(S, -np.inf)
        for k in (8, 16, 32, 64):  # (field, k) by leave-one-out on the OOF memory itself, no validation data
            W = knn_weights(S, k, Y[oof], stake[oof])
            loo = mae((W * Cv[oof]).sum(1), g[oof])
            print(f"knn {field} k={k}: OOF leave-one-out MAE {loo:.4f}")
            if best is None or loo < best[0]:
                best = (loo, field, k)
    _, field, k = best
    E = embeddings(rows, field)
    W = np.full((len(rows), len(views)), 1.0 / len(views))
    rest = ~oof
    W[rest] = knn_weights(E[rest] @ E[oof].T, k, Y[oof], stake[oof])
    save(out, f"knn_{'-'.join(views)}_{field}_k{k}", rows, W, views, {"field": field, "k": k})


def run_mlp(out: Path, views: list[str], seed: int) -> None:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rows = load_rows(out, views)
    E = embeddings(rows, "desc_h")
    num = np.array([[math.log(r["dur"])] + [math.log(max(c, .5)) for c in r["cv"]]
                    + [math.log(r["dur"] / max(c, .5)) for c in r["cv"]] for r in rows]) / 3.0
    dom = np.eye(3)[[DOM.get(r["ds"], 0) for r in rows]]
    X = torch.tensor(np.concatenate([E, num, dom], 1), dtype=torch.float32)
    C = torch.tensor([r["cv"] for r in rows], dtype=torch.float32)
    g = torch.tensor([r["gt"] for r in rows])
    idx = torch.tensor([i for i, r in enumerate(rows) if r["role"] == "oof"])
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, len(views)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best = None
    for ep in range(30):
        net.train()
        perm = idx[torch.randperm(len(idx))]
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]
            w = torch.softmax(net(X[j]), -1)
            loss = (((w * C[j]).sum(1) - g[j]).abs() / (g[j] + 0.1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            W = torch.softmax(net(X), -1).numpy()
        r = report(rows, W, f"mlp ep{ep}", "val")
        if best is None or r["mean_rel"] < best[0]:
            best = (r["mean_rel"], ep, W.copy())
    save(out, f"mlp_{'-'.join(views)}_s{seed}", rows, best[2], views, {"epoch": best[1]})


# ------------------------------------------------------------------------------------ LLM selector
def prompt(r: dict, order: list[int], views: list[str], use_desc: bool) -> str:
    D = r["dur"]

    def hyp(c: float) -> str:
        c = max(c, 0.1)
        return f"{c:.1f} repetitions, one every {D / c:.2f} s"

    K = len(order)
    q = f"A video clip of {D:.1f} seconds shows an action that is repeated several times."
    if use_desc:
        q += f" The repeated action is {r['act']}. One repetition: {r['rep']}."
    q += f" The repetitions were counted {K} times, at different playback speeds."
    q += "".join(f" {LETTERS[j]}: {hyp(r['cv'][order[j]])}." for j in range(K))
    q += f" Which count is correct? Answer {', '.join(LETTERS[:K - 1])} or {LETTERS[K - 1]}."
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


class TextSelector:
    def __init__(self, path: str, K: int, r: int = 16):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        from temporac_cvpr.llm_head import add_lora
        full = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        self.lm, self.head = full.model.language_model, full.lm_head
        del full.model.visual
        for p in list(self.lm.parameters()) + list(self.head.parameters()):
            p.requires_grad_(False)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm.to(self.dev).eval()
        self.head.to(self.dev).eval()
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.loras = nn.ModuleList(add_lora(self.lm, r=r))
        self.tok = AutoTokenizer.from_pretrained(path)
        ids = [self.tok(L, add_special_tokens=False)["input_ids"] for L in LETTERS[:K]]
        assert all(len(i) == 1 for i in ids) and len({i[0] for i in ids}) == K, ids
        self.ans = [i[0] for i in ids]

    def weights(self, batch_rows: list[dict], orders: list[list[int]], views: list[str], use_desc: bool, train: bool):
        """(n, K) weights of the views (in `views` order) for every (row, candidate order)."""
        import torch
        ids = [self.tok(prompt(r, o, views, use_desc), add_special_tokens=False)["input_ids"] for r, o in zip(batch_rows, orders)]
        L = [len(x) for x in ids]
        x = torch.full((len(ids), max(L)), self.tok.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros_like(x)
        for i, s in enumerate(ids):  # right padding; the answer is read at the last real position
            x[i, : L[i]] = torch.tensor(s)
            am[i, : L[i]] = 1
        x, am = x.to(self.dev), am.to(self.dev)
        if self.lm.training != train:
            self.lm.train(train)
        with torch.set_grad_enabled(train):
            H = self.lm(input_ids=x, attention_mask=am, use_cache=False).last_hidden_state
            last = H[torch.arange(len(ids), device=self.dev), torch.tensor(L, device=self.dev) - 1]
            # only the K answer logits, in float32 (bf16 logits are quantised to ~0.1 at this magnitude)
            p = torch.softmax(last.float() @ self.head.weight[self.ans].float().t(), -1)  # (n, K) by letter
        # letter j shows view order[j]  ->  weight of view v = p[:, j] where order[j] == v
        inv = torch.tensor([[o.index(v) for v in range(len(o))] for o in orders], device=self.dev)
        return torch.gather(p, 1, inv)

    def infer(self, rows: list[dict], idx: list[int], views: list[str], use_desc: bool, bs: int = 64) -> np.ndarray:
        import torch
        K = len(views)
        rots = [[(j + s) % K for j in range(K)] for s in range(K)]  # cyclic rotations of the candidate list
        W = np.zeros((len(rows), K))
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            acc = 0
            with torch.no_grad():
                for o in rots:
                    acc = acc + self.weights([rows[i] for i in j], [o] * len(j), views, use_desc, False)
            W[j] = (acc / len(rots)).cpu().numpy()
        return W


def run_llm(out: Path, views: list[str], use_desc: bool, epochs: int, seed: int, smoke: bool) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = load_rows(out, views)
    K = len(views)
    name = f"llm_{'-'.join(views)}_{'desc' if use_desc else 'nodesc'}_s{seed}"
    sel = TextSelector(str(R / "pretrained/Qwen3-VL-8B-Instruct"), K)
    print("prompt example:\n" + prompt(rows[0], list(range(K)), views, use_desc), flush=True)
    oof = [i for i, r in enumerate(rows) if r["role"] == "oof"]
    evaluate = [i for i, r in enumerate(rows) if r["role"] == "val"]
    if smoke:
        oof, evaluate = oof[:32], evaluate[:64]
    params = list(sel.loras.parameters())
    print("trainable", sum(p.numel() for p in params), "oof", len(oof), "val", len(evaluate), flush=True)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    bs = 16
    steps = epochs * math.ceil(len(oof) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / steps))))
    perms = list(itertools.permutations(range(K)))
    best, log = None, []
    for ep in range(epochs):
        t0 = time.time()
        random.shuffle(oof)
        tot = 0.0
        for b in range(0, len(oof), bs):
            j = oof[b:b + bs]
            br = [rows[i] for i in j]
            w = sel.weights(br, [list(random.choice(perms)) for _ in j], views, use_desc, True)
            C = torch.tensor([r["cv"] for r in br], device=sel.dev)
            g = torch.tensor([r["gt"] for r in br], device=sel.dev)
            loss = (((w * C).sum(1) - g).abs() / (g + 0.1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(j)
        W = np.full((len(rows), K), 1.0 / K)
        W[evaluate] = sel.infer(rows, evaluate, views, use_desc)[evaluate]
        r = report(rows, W, f"{name} ep{ep}", "val")
        log.append({"epoch": ep, "train_loss": tot / len(oof), "val": r, "time": time.time() - t0,
                    "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0})
        print(f"train loss {log[-1]['train_loss']:.4f}  {log[-1]['time']:.0f}s  mem {log[-1]['gpu_mem_gb']:.1f} GB", flush=True)
        if best is None or r["mean_rel"] < best[0]:
            best = (r["mean_rel"], ep)
            torch.save({"lora": sel.loras.state_dict(), "epoch": ep, "use_desc": use_desc, "views": views}, out / f"{name}.pt")
            save(out, name, rows, W, views, {"epoch": ep})
    (out / f"{name}_log.json").write_text(json.dumps({"name": name, "best_epoch": best[1], "log": log}, indent=1))
    print("BEST", name, best, flush=True)


def apply(out: Path, name: str) -> None:
    """Weights for the test / QUVA rows of a trained selector; scored once, after all decisions."""
    import torch
    S = json.loads((out / f"{name}.json").read_text())
    views = S["views"]
    rows = load_rows(out, views)
    if name.startswith("llm"):
        ck = torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True)
        sel = TextSelector(str(R / "pretrained/Qwen3-VL-8B-Instruct"), len(views))
        sel.loras.load_state_dict(ck["lora"])
        idx = [i for i, r in enumerate(rows) if r["role"] in ("val", "test", "quva")]
        W = np.full((len(rows), len(views)), 1.0 / len(views))
        W[idx] = sel.infer(rows, idx, views, ck["use_desc"])[idx]
    else:  # stored weights cover every row of the table (computed from the OOF memory / trained gate)
        pos = {tuple(k): i for i, k in enumerate(S["keys"])}
        Wall = np.array(S["W"])
        W = np.stack([Wall[pos[(r["role"], r["seed"], r["key"])]] for r in rows])
    res = {role: report(rows, W, name, role) for role in ("val", "test", "quva")}
    preds = {}
    for i, r in enumerate(rows):
        if r["role"] in ("test", "quva"):
            preds.setdefault(f"{r['role']}_s{r['seed']}", {})[r["key"]] = {
                "pred": float((W[i] * np.array(r["cv"])).sum()), "w": W[i].tolist(), "c": r["c"], "gt": r["gt"]}
    (out / f"{name}_final.json").write_text(json.dumps({"views": views, "results": res, "predictions": preds}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "knn", "mlp", "llm", "apply"])
    ap.add_argument("out")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--views", default="1x,2x")
    ap.add_argument("--no-desc", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out, views = Path(a.out), a.views.split(",")
    assert all(v in SPEED for v in views) and 2 <= len(views) <= len(LETTERS), views
    if a.cmd == "build":
        build(out)
    elif a.cmd == "knn":
        run_knn(out, views)
    elif a.cmd == "mlp":
        run_mlp(out, views, a.seed)
    elif a.cmd == "llm":
        run_llm(out, views, not a.no_desc, a.epochs, a.seed, a.smoke)
    else:
        apply(out, a.name)


if __name__ == "__main__":
    main()
