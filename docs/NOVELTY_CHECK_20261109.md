# Closest prior work per TempoRAC idea (delegated primary-source check, 2026-11-09)

Sources were read on arXiv/ar5iv where possible; CVF-hosted PAMS material and the
KSEM 2023 paper were only available as abstracts/snippets (marked). Re-verify before
camera-ready.

| Idea | Verdict | Closest work and difference |
|---|---|---|
| Multi-rate token pyramid (one frozen ViT at strides 2/4/8, tokens fused per time step) | incremental (new in RAC as a design) | Gan et al., KSEM 2023 "Temporal Repetition Counting Based on Multi-stride Collaboration" (abstract only) fuses **counts** of several strides; TransRAC/HTRM use 1/4/8-frame windows of a fixed 64-frame sampling and fuse similarity matrices; RepNet picks one test-time rate; ESCounts evaluates rates 4/8/16/32 separately and never fuses; SlowFast/Coarse-Fine fuse separately trained pathways. |
| Tempogram-biased self-attention (local masked autocorrelation, z-scored over lags, per-head learnable scale on attention logits) | probably new | Autoformer replaces attention by global FFT auto-correlation (top-k lags); PENGUIN adds a periodic ALiBi bias with an a-priori period; Quatformer learns period/phase parameters; T5/ALiBi biases are content-independent; RAC papers use TSMs as inputs/features or losses. Reviewer risk: compare against raw similarity as bias and a learned lag table (implemented as `bias_kind=sim/relpos`). |
| Supervised local period choosing the sampling rate (tokens-per-cycle gating) | largely done | Zhang et al., CVPR 2020 (Context-aware, scale-insensitive) predicts per-frame cycle lengths and samples around two predicted cycles (coarse-to-fine); FCA-RAC sets one rate per video from the GT first cycle; Sight&Sound learns one stride per video; RepNet picks one rate per video. |
| Tempo robustness tests / speed augmentation | partly done | PAMS supplement (snippet only): random 0.5–1.5x speed change of one 5–7 s test segment; Zhang 2020 / Runia 2018: global 1x/2x/4x speed-ups on YTSegments; SimPer: speed augmentation for self-supervision. |

Name note: a public repository `github.com/by7yqxhsxu/TempoRAC` (partial PAMS reproduction) uses the same name.
