"""Action-level semantic vectors from the MLLM's own description of the repeated action.

The frozen MLLM (mllm_features.py) answers "action: ...; one repetition: ...; count: N".
We keep the description and drop the number, so the conditioning carries what is counted
(shared by all videos of an action) rather than a video-specific state that also encodes the
MLLM's count guess.  Texts are embedded with a sentence encoder (BGE, CLS pooling, L2-normalised).
Output per video ``OUT/<fsplit>__<video_id>.npz``: ``desc_h`` (whole description), ``act_h``
(action only), ``rep_h`` (one-repetition clause only), each (768,) float16, plus the texts in
``descriptions.jsonl``.
Usage: python -m temporac_cvpr.desc_embed MLLM_DIR OUT_DIR ENCODER_DIR
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PAT = re.compile(r"action\s*:\s*(.*?);\s*one repetition\s*:\s*(.*?)(?:;\s*count\s*:|$)", re.I | re.S)


@torch.no_grad()
def main() -> None:
    src, out, enc_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    from transformers import AutoModel, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for f in sorted(src.glob("answers_shard*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            rows[r["key"]] = r  # last answer per video wins (reruns append)
    tok = AutoTokenizer.from_pretrained(enc_dir)
    enc = AutoModel.from_pretrained(enc_dir).cuda().eval().half()
    keys = sorted(rows)
    recs = []
    for k in keys:
        m = PAT.search(rows[k]["answer"])
        act, rep = (m.group(1).strip(), m.group(2).strip()) if m else (rows[k]["answer"], rows[k]["answer"])
        recs.append({"key": k, "action": act, "repetition": rep,
                     "desc": f"action: {act}; one repetition: {rep}", "parsed": bool(m)})

    def embed(texts):
        outs = []
        for i in range(0, len(texts), 256):
            b = tok(texts[i:i + 256], padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
            h = enc(**b).last_hidden_state[:, 0].float()  # CLS pooling (BGE)
            outs.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(outs)

    E = {name: embed([r[field] for r in recs]) for name, field in
         (("desc_h", "desc"), ("act_h", "action"), ("rep_h", "repetition"))}
    with open(out / "descriptions.jsonl", "w") as fo:
        for i, r in enumerate(recs):
            np.savez(out / f"{r['key']}.npz", **{n: E[n][i].astype(np.float16) for n in E})
            fo.write(json.dumps(r) + "\n")
    print(out, len(recs), "parsed", sum(r["parsed"] for r in recs), flush=True)


if __name__ == "__main__":
    main()
