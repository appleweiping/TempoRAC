# Why CountLLM's LLM helps, and what TempoRAC borrows (2026-11-10)

User question (11-10): "Did you train the LLM? Think about why other people's LLM works, borrow it,
stop repeated trial and error."  Answer: no -- every TempoRAC variant so far used a frozen Qwen3-VL and
only one global hidden state (FiLM).  This is the root cause of its weak / non-generalising effect.

## CountLLM (Yao et al., CVPR 2025; arXiv 2503.17690v2) -- facts from the paper

Read from the arXiv HTML (Sec. 1-4.1, Tables 1-2 verbatim; Tables 5-7 only via a summary -- re-check
the exact numbers before citing them).

- Pipeline: video encoder (VideoMAE or Video Swin, Kinetics) -> **periodicity transformer** (64 learnable
  queries, 12 layers, cross-attention to encoder features) -> linear projector -> **Vicuna-7B**.
  The LLM sees 64 periodic tokens per 32-frame clip followed by text.
- **LoRA r = 16** in the attention layers of both the video encoder and the LLM (Sec. 4.1).
- Structured template: {periodic tokens}, {periodicity description (GPT-4, class-agnostic, manually
  refined)}, {question}, {answer}. Answer "[abcd,e,f]": 4-digit zero-padded count of the clip plus flags
  for an incomplete cycle at the clip start / end; video count = sum over clips with a boundary rule.
- Progressive training: (1) video-caption contrastive alignment on WebVid-10M (10 ep), (2) periodicity
  alignment: repetitive vs non-repetitive clips against the periodicity description (50 ep),
  (3) instruction tuning with LoRA, token cross-entropy on the answer (50 ep). 8x A100.
- Ablations (RepCount val, summary-only): all frozen (no periodicity transformer) 0.498; periodicity
  transformer without LoRA 0.409; no alignment stages 0.402; no periodicity description 0.305;
  no decimal-string answer 0.392; full 0.166.  => the LLM must be **trained for counting**, must see a
  **temporal token sequence**, and benefits from **alignment before tuning** and a **structured prompt**.

## Our evidence (consistent with the above)

Frozen Qwen3-VL FiLM: small validation gains that were largely model-selection noise; on test, removing
it was not worse (final_v2: 0.187 vs 0.196 mean MAE). Condition dropout recovers a validation gain
(last-10-epoch mean -5.4% vs no LLM), but the LLM still never learns anything about counting.

## Design (TempoRAC-LLM)

1. Video tokens: the TempoRAC temporal encoder output on the 2-frame grid is average-pooled over windows
   of 8 cells (16 frames) -> at most ~165 tokens per RepCount video; MLP projector to the LLM width.
   (CountLLM's periodicity tokens; ours come from the multi-rate, two-depth pyramid.)
2. LLM: language model of Qwen3-VL-8B-Instruct (same size class as Vicuna-7B), **LoRA r = 16 on
   q/k/v/o**, gradient checkpointing, bf16.
3. Structured prompt: video tokens + the per-video "action / one repetition" description produced by
   Qwen3-VL (count removed) + question; answer "count: dddd" (4-digit, CountLLM's decimal string).
4. Outputs: (a) answer tokens, cross-entropy (as CountLLM); (b) the LLM's last hidden states at the video
   token positions, projected back and added to the temporal-encoder features, feed the TempoRAC density
   head (density MSE). Reported count = density sum (precise, no clip aggregation needed).
5. Progressive training: stage 1 alignment (LLM frozen, no LoRA): train encoder + projector so that the
   LLM reproduces the description from the video tokens (captioning loss); stage 2: LoRA + density +
   answer losses, joint over the three datasets with dataset tokens and balanced sampling.
6. Compute: ~250 tokens per sample; one 24 GB GPU (GPU0; GPU1 is shared with another project).
