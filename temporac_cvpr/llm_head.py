"""TempoRAC-LLM: a LoRA-tuned language model reasons over the temporal video tokens.

Borrowed from CountLLM (CVPR 2025): the LLM must be *trained* for counting (LoRA r = 16 on the
attention projections), must see a *sequence* of temporal video tokens, receives a structured
prompt with a description of the periodic action, and answers with a zero-padded decimal count.
Our own parts: the video tokens come from the multi-rate, two-depth token pyramid and its temporal
encoder; the LLM's hidden states at the video positions are fed back into the density decoder, so
the reported count is still the integral of a frame-level density (no clip aggregation); the video
is given twice ("echo") so that every second-pass position has seen the description and the whole
video under causal attention.

The 8B language model is loaded once onto the GPU, frozen, and deliberately *not* registered as a
sub-module: checkpoints and optimiser state only contain the TempoRAC decoder, the projectors and the
LoRA factors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporac_cvpr.models import ConcatHead


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A dropout(x); W frozen and not registered (kept out of state_dict)."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        object.__setattr__(self, "base", base)  # frozen 8B weights stay outside the module tree
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero: starts as the frozen LLM
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.drop(x) @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale


def add_lora(lm: nn.Module, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=16, alpha=32.0, dropout=0.05):
    loras = []
    for name, mod in list(lm.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                lo = LoRALinear(child, r, alpha, dropout).to(child.weight.device)
                setattr(mod, t, lo)
                loras.append(lo)
    return loras


class TempoRACLLM(nn.Module):
    def __init__(self, cfg: dict, rates: list[int]):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        self.base = ConcatHead(cfg, rates)
        d = cfg["d"]
        full = Qwen3VLForConditionalGeneration.from_pretrained(cfg["llm_path"], dtype=torch.bfloat16)
        lm, head = full.model.language_model, full.lm_head
        del full.model.visual
        for p in list(lm.parameters()) + list(head.parameters()):
            p.requires_grad_(False)
        lm.cuda().eval()
        head.cuda().eval()
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        object.__setattr__(self, "lm", lm)
        object.__setattr__(self, "lm_head", head)
        object.__setattr__(self, "tok", AutoTokenizer.from_pretrained(cfg["llm_path"]))
        D = lm.config.hidden_size
        self.task = cfg.get("llm_task", "count")  # "caption" (alignment stage) | "count" (instruction tuning)
        self.loras = nn.ModuleList(add_lora(lm, r=cfg.get("lora_r", 16)) if cfg.get("llm_lora") else [])
        self.win = int(cfg.get("llm_window", 8))  # base cells per video token (8 cells = 16 frames)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D), nn.GELU(), nn.Linear(D, D))
        self.back = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        nn.init.zeros_(self.back[-1].weight)  # starts as the pretrained TempoRAC decoder
        nn.init.zeros_(self.back[-1].bias)
        self._cache = {}

    # ---- prompt pieces -------------------------------------------------------------------
    def _ids(self, text: str) -> list[int]:
        if text not in self._cache:
            self._cache[text] = self.tok(text, add_special_tokens=False)["input_ids"]
        return self._cache[text]

    def _texts(self, desc, count):
        act, rep = desc
        if self.task == "caption":  # alignment: the video tokens must carry the description
            pre = "<|im_start|>user\nVideo: "
            mid = "\nVideo again: "
            post = "\nDescribe the repeated action and what makes up one repetition.<|im_end|>\n<|im_start|>assistant\n"
            ans = f"action: {act}; one repetition: {rep}<|im_end|>"
        else:  # counting with a structured prompt (CountLLM's periodicity description, here per video)
            pre = (f"<|im_start|>user\nA video shows an action repeated several times. The repeated action is {act}. "
                   f"One repetition: {rep}. Video: ")
            mid = "\nVideo again: "
            post = "\nHow many complete repetitions are in the video?<|im_end|>\n<|im_start|>assistant\n"
            ans = f"count: {int(count):04d}<|im_end|>"
        return pre, mid, post, ans

    # ---- forward -------------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        out = self.base(batch)
        h, mask = out["hidden"], batch["mask"]  # (B, G, d) on the 2-frame grid
        B, G, d = h.shape
        W = self.win
        Gv = math.ceil(G / W)
        hp = F.pad(h.float(), (0, 0, 0, Gv * W - G)).view(B, Gv, W, d)
        mp = F.pad(mask.float(), (0, Gv * W - G)).view(B, Gv, W)
        tokens = (hp * mp.unsqueeze(-1)).sum(2) / mp.sum(2, keepdim=True).clamp_min(1.0)  # (B, Gv, d)
        nv = (mp.sum(2) > 0).sum(1).tolist()  # video tokens per sample
        emb_v = self.proj(tokens).to(torch.bfloat16)
        embed = self.lm.get_input_embeddings()
        dev = h.device
        counts = batch["density"].float().sum(1).round().clamp(0, 9999).tolist()
        want_ans = self.training
        seqs, vpos, apos, aids = [], [], [], []
        for i in range(B):
            pre, mid, post, ans = self._texts(batch["desc"][i], counts[i])
            ids = [self._ids(pre), self._ids(mid), self._ids(post)]
            e = [embed(torch.tensor(x, device=dev)) for x in ids]
            v = emb_v[i, : nv[i]]
            parts = [e[0], v, e[1], v, e[2]]
            start2 = len(ids[0]) + nv[i] + len(ids[1])  # second (echo) pass of the video tokens
            vpos.append(torch.arange(start2, start2 + nv[i], device=dev))
            L = sum(p.shape[0] for p in parts)
            if want_ans:
                a = self._ids(ans)
                parts.append(embed(torch.tensor(a, device=dev)))
                apos.append(torch.arange(L - 1, L - 1 + len(a), device=dev))  # positions predicting answer tokens
                aids.append(torch.tensor(a, device=dev))
            seqs.append(torch.cat(parts, 0))
        Lmax = max(s.shape[0] for s in seqs)
        x = torch.zeros(B, Lmax, seqs[0].shape[1], dtype=torch.bfloat16, device=dev)
        am = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):  # right padding; causal attention never looks at the padding
            x[i, : s.shape[0]] = s
            am[i, : s.shape[0]] = 1
        if self.lm.training != self.training:
            self.lm.train(self.training)  # gradient checkpointing is only active in training mode
        H = self.lm(inputs_embeds=x, attention_mask=am, use_cache=False).last_hidden_state  # (B, Lmax, D)
        hv = torch.zeros(B, Gv, H.shape[-1], dtype=H.dtype, device=dev)
        for i in range(B):
            hv[i, : nv[i]] = H[i, vpos[i]]
        up = torch.repeat_interleave(self.back(hv.float()), W, dim=1)[:, :G]  # back to the 2-frame grid
        h2 = h + up.to(h.dtype)
        out["density"] = self.base.decode(h2, batch) * mask.float()
        if want_ans:
            logits = self.lm_head(torch.cat([H[i, apos[i]] for i in range(B)], 0))
            out["ans_loss"] = F.cross_entropy(logits.float(), torch.cat(aids, 0))
        return out
