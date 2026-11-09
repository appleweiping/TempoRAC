"""Frozen video MLLM (Qwen3-VL) features for counting, cached once per video.

For every video (inside its counting span, the same span as the token pyramid) we sample
``--frames`` frames uniformly, resize them (shorter side 256, longer side a multiple of 32,
at most 448) and run the MLLM with a counting instruction.  Saved per video
(``OUT/<fsplit>__<video_id>.npz``):

- ``prompt_h``  (L1, D): hidden state of the last prompt token at layers ``--prompt-layers``
- ``answer_h``  (D,):    final-layer hidden state of the last generated token
- ``vis``       (G, L2, D): per temporal group (2 sampled frames), the spatial mean of the
  visual-token hidden states at layers ``--vis-layers``; time-aligned semantic stream
- ``group_frames`` (G, 2): first/last sampled frame of each group (relative to the span start)
- ``T``: frames in the span
Generated answers (text and parsed count, a zero-shot reference) go to ``answers_shard*.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

PROMPT = ("The video shows an action that is repeated several times. What is the repeated action, "
          "what makes up one complete repetition, and how many complete repetitions are there? "
          "Answer in the form: action: ...; one repetition: ...; count: N.")


def load_span(it: dict, n_frames: int):
    import decord

    vr = decord.VideoReader(it["file"], ctx=decord.cpu(0), num_threads=2)
    n0, fps = len(vr), float(vr.get_avg_fps() or 30.0)
    a, b = it.get("frame_range", [0, n0 - 1])
    b = min(int(b), n0 - 1)
    T = b - a + 1
    rel = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
    if len(rel) % 2:  # groups of two frames (temporal patch 2)
        rel = np.append(rel, rel[-1])
    h0, w0 = vr[0].shape[:2]
    s = 256 / min(h0, w0)
    H = max(256, min(448, int(round(h0 * s / 32)) * 32))
    W = max(256, min(448, int(round(w0 * s / 32)) * 32))
    del vr
    vr = decord.VideoReader(it["file"], ctx=decord.cpu(0), width=W, height=H, num_threads=2)
    frames = vr.get_batch((rel + a).tolist()).asnumpy()  # n H W 3
    return frames, rel, T, fps


class SpanSet(torch.utils.data.Dataset):
    def __init__(self, items, n_frames):
        self.items, self.n = items, n_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            return it, *load_span(it, self.n), None
        except Exception as e:  # recorded, never silently dropped
            return it, None, None, None, None, repr(e)


def parse_count(text: str):
    m = re.search(r"count\s*[:=]\s*(\d+)", text, re.I) or re.search(r"(\d+)\s*(?:times|repetitions)", text, re.I)
    return int(m.group(1)) if m else None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--prompt-layers", default="12,24,36")
    ap.add_argument("--vis-layers", default="24,36")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.video_utils import VideoMetadata

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.manifest).read_text())
    items = [it for k, it in enumerate(items) if k % args.num_shards == args.shard]
    fkey = lambda it: f"{it.get('fsplit', it['split'])}__{it['video_id']}"
    todo = [it for it in items if not (out / f"{fkey(it)}.npz").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"shard {args.shard}: {len(items)} items, {len(todo)} to do", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").cuda().eval()
    vid_tok = model.config.video_token_id
    PL = [int(x) for x in args.prompt_layers.split(",")]
    VL = [int(x) for x in args.vis_layers.split(",")]
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    dl = torch.utils.data.DataLoader(SpanSet(todo, args.frames), batch_size=None, num_workers=args.workers,
                                     prefetch_factor=2)
    ans_f = open(out / f"answers_shard{args.shard}.jsonl", "a")
    t0, fails = time.time(), []
    for n, (it, frames, rel, T, fps, err) in enumerate(dl):
        key = fkey(it)
        if err is not None:
            fails.append({"key": key, "error": err})
            print("FAIL", key, err, flush=True)
            continue
        # the DataLoader's default conversion turns numpy outputs into tensors
        frames, rel, T, fps = torch.as_tensor(frames), np.asarray(rel), int(T), float(fps)
        meta = VideoMetadata(total_num_frames=T, fps=fps, frames_indices=[int(x) for x in rel],
                             height=int(frames.shape[1]), width=int(frames.shape[2]))
        vid = frames.permute(0, 3, 1, 2).contiguous()  # T 3 H W (channels first, explicit)
        inp = proc(text=[text], videos=[vid], return_tensors="pt",
                   videos_kwargs={"do_sample_frames": False, "video_metadata": [meta],
                                  "size": {"shortest_edge": 32 * 32, "longest_edge": 10 ** 9}})
        if n == 0:  # record the timestamps the model actually sees
            s = proc.tokenizer.decode(inp["input_ids"][0])
            print("TIMESTAMPS", re.findall(r"<[\d.]+ seconds>", s)[:4], "...", re.findall(r"<[\d.]+ seconds>", s)[-2:],
                  f"(span {T / fps:.1f}s)", flush=True)
        inp = {k: v.cuda() for k, v in inp.items()}
        g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                           output_hidden_states=True, return_dict_in_generate=True)
        hs0 = g.hidden_states[0]  # prompt pass: (layers+1) x (1, P, D)
        ids = inp["input_ids"][0]
        pos = (ids == vid_tok).nonzero().squeeze(1)
        t_g, h_p, w_p = [int(x) for x in inp["video_grid_thw"][0]]
        per = (h_p // 2) * (w_p // 2)
        assert len(pos) == t_g * per, (len(pos), t_g, per)
        vis = torch.stack([hs0[l][0, pos].view(t_g, per, -1).float().mean(1) for l in VL], 1)  # G L2 D
        prompt_h = torch.stack([hs0[l][0, -1].float() for l in PL], 0)
        answer_h = g.hidden_states[-1][-1][0, -1].float() if len(g.hidden_states) > 1 else hs0[-1][0, -1].float()
        gen = proc.batch_decode(g.sequences[:, ids.shape[0]:], skip_special_tokens=True)[0]
        relp = rel.reshape(-1, 2)
        np.savez(out / f"{key}.npz", prompt_h=prompt_h.cpu().numpy().astype(np.float16),
                 answer_h=answer_h.cpu().numpy().astype(np.float16), vis=vis.cpu().numpy().astype(np.float16),
                 group_frames=relp.astype(np.int32), T=np.int32(T), fps=np.float32(fps))
        ans_f.write(json.dumps({"key": key, "count_gt": it.get("count"), "answer": gen,
                                "count_pred": parse_count(gen), "frames": int(len(rel)),
                                "tokens": int(ids.shape[0])}) + "\n")
        ans_f.flush()
        if n % 20 == 0:
            print(f"[{args.shard}] {n + 1}/{len(todo)} {key} tok={ids.shape[0]} {(time.time() - t0) / 60:.1f}min "
                  f"| {gen[:90]!r}", flush=True)
    (out / f"failures_shard{args.shard}.json").write_text(json.dumps(fails, indent=1))
    print("MLLM_DONE", args.shard, "failures", len(fails), flush=True)


if __name__ == "__main__":
    main()
