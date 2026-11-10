# Why Context / Sight&Sound are strong on UCFRep (and how comparable they are) — mechanisms and transfer to TempoRAC

Date: 2026-11-10 (research run 2026-11-09). Scope: read-only literature + released-code audit. No experiments were run; the TempoRAC failure-mode links in §6 are hypotheses for you to test.

## 0. Bottom line

* **Context (CVPR20) UCFRep 0.147 / 0.79**: the protocol matches ours, which I checked against the released code and annotations. It counts inside the annotated segment, on the same 105 val IDs, with the count rounded first, and OBO is defined the same way. The one metric difference is the denominator: it is `g`, ours is `g+0.1`. That makes our MAE on average 2.3% smaller (worst case 4.8%), so our 0.206 is about 0.21 in their units. **The 0.06 gap is real.** The main differences are in how the model is built and selected, not in the metric: the backbone is fine-tuned end-to-end on UCFRep only, there is no sealed test (ablations and checkpoints use the same val set), and inference uses UCFRep-specific constants.
* **Sight&Sound (CVPR21) UCFRep 0.143 / 0.80**: the metric formula matches ours, but the protocol is otherwise **probably comparable and not verifiable**. No UCFRep code was released. The paper states 420/106 (the released split is 421/105). P=24 was found by greedy search, with no held-out set named. The released Countix code trims to the annotated segment and rounds before the metric.
* **On Countix, S&S is not strong**: 0.307 / 0.511. Our 0.238 / 0.69 already beats it on both metrics. Caveat: the S&S Countix subset and retrieval rate are not reported.
* **Why the two papers win on UCFRep, ranked by evidence**: (1) the encoder is fine-tuned in-domain; (2) inputs are normalised to the period of each video or position, at native or near-native frame rate, for fast cycles; (3) UCFRep-only training on a closed set of 23 classes, with no sealed test; (4) a period-based readout `count = L x mean(1/T)` with explicit half/double anchors (Context only).

## 1. Sources and provenance

| Item | Source read |
|---|---|
| Context paper | CVF PDF (`scratchpad/pdfs/zhang_context.pdf`, same as CVF link in `cvf/context.html`); arXiv 2005.08465 also downloaded |
| Context code | GitHub `Xiaodomgdomg/Deep-Temporal-Repetition-Counting` @ `f7011d3e` (same revision as `configs/protocols/ucfrep_526.yaml`): `val_2stream.py`, `datasets/ucf_aug.py`, `utils/myutils.py`, `opts.py`, `dataset.py` |
| UCFRep annotations | `ucf526_annotations.zip`, sha256 `08b2b8a8...` (matches our yaml); statistics below were computed from the 526 .mat files |
| S&S paper | CVF PDF (`/content/CVPR2021/papers/Zhang_Repetitive_Activity_Counting_by_Sight_and_Sound_CVPR_2021_paper.pdf`) + arXiv 2103.13096 (14 pp, includes Appendix A–J) |
| S&S code | GitHub `xiaobai1217/RepetitionCounting` @ `fa4af108` (Countix/Countix-AV only): `eval_all_sr.py`, `run_demo.py`, `generate_csv4sr.py`, `train.py`, `dataloader.py` |
| OVR / MultiCounter / IVAC-P2L | `scratchpad/pdfs/{ovr,multicounter,ivac}.txt` (+ PDFs, re-extracted with `pdftotext -table` for tables) |
| Cross-check | ESCounts Tab. 1b/1c (`pdfs/escounts.txt`) |

Table values were re-extracted in table mode because the plain-text extraction of S&S Table 5 shifts its rows.

## 2. Protocol comparability (question 2)

| | Context (code-verified) | S&S (paper + Countix code) | TempoRAC |
|---|---|---|---|
| Counting window | Annotated segment `[start_frame, end_frame]` (`ucf_aug.py`: `frame_indices = range(begin_t, end_t+1)`). In all 526 .mat files, `duration == end-start+1`, start = first bound, end = last bound, so the segment holds an integer number of cycles | UCFRep: not stated, no code. Countix: segment-trimmed (`video1[start1:end1]`) | Annotated segment (ESCounts convention) |
| GT count | `len(temporal_bound)-1`; videos with <2 cycles are skipped (none are, min = 2) | Same annotations presumably | same |
| Split / n | `annotations/val` = 105 (groups 21–25) | Paper text says "420 / 106" (ESCounts copies this); released split is 421/105 | 105 sealed test |
| Prediction | `counts = round(L * mean(0.5/t_p + 0.5/t_n))` (`val_2stream.py` l.318–323) | `np.round` before metric (`run_demo.py` l.90–97) | round(p) |
| MAE | `|round(p)-g| / g` | Eq. 12, same | `|round(p)-g| / (g+0.1)` |
| OBO | `|round(p)-g| <= 1` | Eq. 13, same | same |
| Encoder | 3D-ResNeXt101, fine-tuned end-to-end (§3.3) | R(2+1)D-18 (code: `r2plus1d_18(pretrained=True)`; paper says "S3D" but cites R(2+1)D [35]), Kinetics init, fine-tuned | frozen ST-MAE ViT-B |
| Training data for UCFRep number | UCFRep train only (421) | UCFRep train only (App. H) | RepCount-A + Countix + UCFRep jointly |
| Model / hyper selection | No dev split. Val is evaluated every epoch; Tables 3–5 ablate on the same val set | P=24 chosen "by greedy search" (App. H), set unnamed | 331/90 group split of train; test touched once |
| Dataset-specific inference priors | `min_cycles=2` for UCFRep (4 for QUVA/YTSeg); initial period search from 0.03·L to 0.35·L, and the period must be < L/2. Both encode the UCFRep fact "segment holds >= 2 cycles" | Count bins 2..35 per 64-frame clip, minimum 2 per clip (Countix code; UCFRep head unreleased) | none (check, §7) |

**Size of the normalisation difference** (UCFRep val, computed): mean of g/(g+0.1) is 0.977, minimum 0.952. If every video were off by exactly 1 count, MAE would be 0.234 with /g and 0.227 with /(g+0.1). So our 0.206 is about 0.21 in Context/S&S units, and OBO is unchanged.

**Reproducibility of Context**: ESCounts evaluated the public Context checkpoint and lists RMSE 2.165, MAE 0.147, OBZ 0.452, OBO 0.790 (Tab. 1c). The number therefore reproduces under the released eval code.

**Verdict**: Context's numbers are metric-comparable to ours, within a deterministic 2–5% normalisation offset in our favour. They are not regime-comparable: fine-tuned specialist, and evaluation-set selection. S&S's are probably metric-comparable, but neither the window nor the split can be audited. In the paper, report them as "fine-tuned, UCFRep-only, val-selected" rows next to frozen-encoder rows, which is ESCounts' framing. ESCounts states its frozen model does not beat these two fine-tuned methods on UCFRep, and shows Context with a frozen encoder ("Context (F)") collapsing to **0.653 / 0.372**.

## 3. Context (Zhang et al., CVPR20): mechanism and what drives its UCFRep score

**Mechanism (paper §3 plus code)**
* **Target**: for each frame i, the previous and next repetition offsets `t_p(i)`, `t_n(i)`. Count = Σ_i (0.5/t_p + 0.5/t_n) over the segment (Eq. 2). The code uses the version sampled at 2^(K-1) positions (Eq. 4), skipping positions whose context window leaves the segment.
* **Scale normalisation is the core**: `update_inputs_2stream` samples 16 frames uniformly over [i−2t_p, i] and 16 over [i+1, i+2t_n]. The 32-frame 112² input therefore always shows about 4 cycles at about 8 frames per cycle, whatever the true period. Fast cycles get duplicated frames, slow cycles are subsampled. The network never sees an aliased cycle.
* **Head**: 7 anchors {0.5, 0.67, 0.8, 1, 1.25, 1.5, 2} × {prev, next}, with IoU>0.5 classification plus log-space regression (Eq. 3, λ=50). The 0.5 and 2.0 anchors are explicit half/double hypotheses.
* **Coarse-to-fine** (Alg. 1, code): at the middle frame, an exhaustive search over 30 log-spaced periods from 0.03·L to 0.35·L, picked by anchor-1.0 confidence, then 4 refinements. Then K=5 stages: positions at N/2^k are initialised from neighbouring estimates and refined, with an EMA merge of w=0.5. 74 forwards, 1.8 s per video.
* **Training**: random crops containing at least 2 annotated cycles (count = cycles covered). Labels are linearly interpolated inside near-periodic stretches (variation < 0.3).

**Ablations (all on the UCFRep val set used as test)**

| Factor | Evidence | Δ MAE / OBO |
|---|---|---|
| Backbone (Tab. 2) | R18 0.213/0.69 → R50 0.190/0.74 → R101 0.187/0.77 → RX101 0.147/0.79 | **−0.066 / +0.10** (largest) |
| Framework vs fixed-scale classifier, same RX101 backbone (Tab. 2) | Levy&Wolf retrained: 0.286/0.68 | −0.139 / +0.11 |
| Separate prev/next period ("Free") vs one shared period (Tab. 4) | 0.177/0.70 → 0.157/0.76 | −0.020 / +0.06 |
| Multi-anchor (Tab. 4) | 0.157 → 0.147, OBO 0.76 → 0.79 | −0.010 / +0.03 |
| Refinement stages (Tab. 3) | St3 0.157, St4 0.156, St5 0.147, St6 0.151 | ≈0.01, noisy |
| Encoder frozen (ESCounts Tab. 1c, "Context (F)") | 0.653/0.372 | +0.51 / −0.42 |
| Speed robustness (Fig. 4, YTSeg) | 1×/2×/4× MAE 5.3/6.2/8.0% vs Levy&Wolf 6.5/17.3% | scale-insensitive |

Reading the ablations together: the period-normalised framework plus a fine-tuned 3D CNN carries the result. Remove fine-tuning and the framework alone is poor (0.653). Absolute error tells a different story from relative error. ESCounts ViT-B, frozen, has RMSE 1.972 against Context's 2.165, but MAE 0.216 against 0.147. So Context's lead is in relative precision on low and mid counts (OBZ 0.452 vs 0.381), not in the high-count tail.

## 4. Sight & Sound (Zhang, Shao, Snoek, CVPR21)

**Mechanism.** The UCFRep number comes from the sight stream only.
* R(2+1)D-18, fine-tuned. The input is a 64-frame 112² clip at temporal stride S. The count of a video is the sum over non-overlapping clips. The tail clip is averaged with a stride S+2 view.
* **Repetition-class mixture** (Eq. 2–3): P count heads, combined by a softmax gate f_v2 learned without labels. A **diversity loss** (Eq. 5, cosine between gate units) stops the gate from collapsing. In code, each head is a 34-bin softmax over counts 2..35 with expectation readout (DEX). P=41 on Countix, **P=24 on UCFRep**, which has 23 classes. App. G: replacing L_div with action-class labels does not help (0.309 vs 0.314).
* **Loss** (Eq. 6): L2 on the count plus λ·|C−l|/l plus λ·L_div, with λ=10.
* **Temporal stride decision** (§3.3, App. A, E): a per-video score head trained with a max-margin ranking loss (m=2.9). The positive is the smallest stride whose clip holds at least 2 cycles; in code, `sr = floor((T+2)/32)+1`, so the period spans 32 frames or fewer in the clip. Negatives are strides covering fewer than 2 cycles, or strides whose prediction deviates by more than θ=0.29 from the positive's prediction (omissions). At test time, strides 1..5 are scored and the argmax is used. **For UCFRep periods (median 29 frames at 25 fps), this selects stride 1–2, i.e. native-rate inputs.**
* **Reliability fusion** (§3.4, App. F): a sigmoid weight between the sight and sound counts. The key trick is to train it on *empirical predictions*: predictions averaged over training epochs whose val loss is below a threshold, not predictions from the final, overfit model.

**Ablations.** There are none on UCFRep; all are on Countix or Countix-AV.

| Factor | Evidence | Number |
|---|---|---|
| Learned per-video stride vs best fixed stride (App. E Tab. 7, Countix sight) | fixed s=1..6: 0.607 / 0.378 / 0.387 / 0.427 / 0.467 / 0.475 | adaptive **0.314** (−17% vs best fixed) |
| Count loss terms (Tab. 3, sight) | L2 0.371/0.424; L2+L_div 0.324/0.478; L2+L_rel 0.370/0.421; all three 0.314/0.498 | gate diversity is the large term |
| L1 (relative) vs L2 (absolute) by count (App. D) | >20 cycles: 0.356 with both, 0.387 without the relative term, **0.553 without L2**; 2-cycle videos: 0.217 with both, 0.348 without the relative term | the absolute term protects high counts |
| Fusion (Tab. 2, Countix-AV) | average 0.300; reliability 0.291; FC fusion 0.301; final-model targets 0.297 (Tab. 9) | small (0.01) |
| Countix (Tab. 5) | sight 0.314/0.498; sound 0.793/0.331; sight&sound **0.307/0.511**; RepNet as reported 0.364/0.697, released checkpoint 0.525/0.289 | weak OBO |

## 5. UCFRep facts that explain the failure modes (computed from the released annotations; 25 fps)

| | train (421) | val/test (105) |
|---|---|---|
| count: mean / median / p90 / max | 6.59 / 5 / 13 / 53 | 6.95 / 5 / 13 / 32 |
| count >= 11 | 14.7% | **21.0%** |
| count == 2 / <= 3 | n/a / 33.5% | 15.2% / 34.3% |
| mean period < 16 frames / < 32 frames | 19.0% / 60.8% | 17.1% / 59.0% |
| period of count>=11 videos (median, range) | 15.2 (4.3–56) | **16.0 (6.5–27.4)** |
| within-video max/min cycle ratio >= 1.5 / >= 2 | 34% / 18% | 41% / 20% |

* High counts in UCFRep are fast cycles of 12–21 frames (JumpRope, SoccerJuggling, Cutting, Shaving, Hammering, Biking). The official split itself shifts toward high counts (14.7% → 21%), so a dev split with 12% under-represents the test regime.
* **Aliasing hypothesis for TempoRAC (verify).** Assume an ST-MAE temporal token at frame stride s covers 2s raw frames (tubelet 2). Then at least 2 tokens per period requires T >= 4s:
  * stride 2: about 97% of test videos are safe.
  * stride 4: 83% are safe.
  * stride 8: only 41% are safe.
  * Most count>=11 videos have T ≈ 16, which sits exactly at the stride-4 limit and is aliased at stride 8.

  If the fusion across strides does not adapt to each video and time step, the aliased streams pull counts down or halve them. Context sidesteps this with about 8 frames per cycle, S&S with stride 1–2 on native frames.

## 6. OVR, MultiCounter, IVAC-P2L (question 3)

| | Core mechanism | Countix / UCFRep handling & numbers | What transfers |
|---|---|---|---|
| **OVR / OVRCounter** (Dwibedi et al., arXiv 2407.17085) | 72K annotations, Kinetics + Ego4D, with count, start/end and free text. ViViT (MAE-pretrained, K=16, S=4) sliding window over 320 frames. A cross-attention "resampler" to per-frame latents, then a 2-layer **AdaLN** counter conditioned on the CLS token or BERT text. Per-frame density peaked at the **mid-point of each period**, MSE ×100. Three loss branches (video-only, matched text, **mismatched text → zero density**) plus a CLIP-style CLS↔text contrastive loss | No UCFRep/Countix evaluation (only dataset statistics: UCFRep 8.2 s / 6.7 counts; Countix 6.1 s / 6.8, range 2–73). OVR-Kinetics MAE 0.39, OBO-error 0.41, OBZ-error 0.75; Ego4D 0.35 / 0.34; RepNet 0.86 / 0.65. Text conditioning equals video-only (0.39 vs 0.39). Counts the whole clip and scores localisation by IoU | (a) **Mismatched-prompt negatives with a zero target**, to make the weak Qwen3-VL FiLM causally used; OVR shows text adds selectivity, not accuracy. (b) AdaLN conditioning instead of FiLM. (c) OVR as extra training data, *after* checking Kinetics-ID overlap with the Countix test |
| **MultiCounter** (Tang et al., arXiv 2409.04035) | Multi-instance: ResNet50-FPN, instance queries, a mixed spatio-temporal module, and a CDM block (self-similarity matrix + attention). **Per-frame period velocity** (classification over 1/period bins, up to T/2) plus a periodicity mask. Count = Σ_t (1/period_t) over frames inside the mask. Clip sampling rate chosen so each clip holds at least 2 reps | Trained on synthetic MultiRep (built from RepCount). Zero-shot UCFRep **0.432 / 0.343** (RepCount 0.232 / 0.527). No Countix | A dense period-velocity head as a second readout (handles within-video tempo change; 41% of UCFRep test videos); self-similarity features as a tempo-invariant cue. Only cross-domain evidence |
| **IVAC-P2L** (Wang et al., arXiv 2403.11959) | TransRAC backbone (Video Swin, multi-scale self-similarity S1/S4/S8, density map) plus a **pull loss** (cycle embeddings → their mean) and a **push loss** (interval embeddings away from the cycle mean). Random Count Augmentation: if count > 15, with p=0.5 resample the count to U[1,15] and crop the video | Trained on RepCount-A only. Zero-shot **UCFRep val 0.503 / 42.0%**, **Countix val 0.507 / 43.1%** (TransRAC re-run 0.596 / 32.0, 0.574 / 38.5). RepCount 0.402 / 34.4. Ablations: pull or push each about 0.02–0.03 MAE; RCA changes MAE 0.398 → 0.402 and OBO 32.5 → 34.4%; L=128 frames beats 64 (0.380 vs 0.402) | Pull loss on per-cycle pooled tokens as an auxiliary (weak evidence; on Countix the cycles are pseudo-labels). Its RCA lowers counts, which is the wrong direction for our failure. The **L=128 > 64** result is another data point that denser temporal sampling helps |

Common lesson: training on RepCount alone transfers poorly to UCFRep (0.43–0.64), so in-domain tempo coverage dominates.

## 7. Transfer to TempoRAC (question 4)

The priority mixes the strength of the evidence with cost on frozen tokens.

| # | Idea (source) | Failure mode targeted | Concrete form in TempoRAC | Evidence strength | Cost |
|---|---|---|---|---|---|
| 1 | **Per-time stride-reliability gate with Nyquist-derived labels** (S&S §3.3, Tab. 7) | fast / high-count under-count; half/double from aliased strides | Softmax weights over the stride 2/4/8 density maps, per token (not per video). Supervise with a ranking or CE loss: a stride is positive if T_gt/(2s) >= 2 tokens and the window holds at least 2 cycles; it is negative if out-of-fold predictions deviate by more than 0.3. Train the gate on out-of-fold or epoch-averaged predictions (S&S App. F empirical predictions) so it does not learn from overfit train outputs | Medium-high (−17% vs best fixed stride on Countix) | Low |
| 2 | **Count-level loss: absolute plus relative** (S&S Eq. 6, App. D) | high-count under-count; low-count instability | Add λ_a·(ΣD−c)² + λ_r·|ΣD−c|/c to the density MSE | High for the >20-count bucket (0.553 → 0.356 when L2 is present) | Very low |
| 3 | **Period head with readout `L/T̂` and 0.5×/2× arbitration** (Context Eq. 2/4, Tab. 4; MultiCounter period velocity) | half/double; high counts (relative count error equals relative period error) | Log-period classification over [4 frames, L/2] with residual regression on pooled segment tokens, plus a dense 1/T(t) head. Final count = learned gate between ΣD and L·mean(1/T̂). If ΣD/(L/T̂) ≈ 0.5 or 2, choose with a small verifier. UCFRep and RepCount labels come from bounds; Countix uses L/c, the same assumption as our pseudo-cycles | Medium (multi-anchor +0.03 OBO, Free +0.06 OBO; requires fine-tuned features in Context) | Low-medium |
| 4 | **Two-pass canonical-tempo re-encoding** (Context's 8 frames/cycle; S&S stride selection; MultiCounter ≥2 reps per clip) | fast reps aliased at fixed strides; period variation within an octave | Pass 1: current model → T̂. Pass 2: re-extract ST-MAE tokens at a continuous stride s* = T̂/k (k ≈ 8 sampled frames per cycle; fractional strides by frame interpolation), run the head, and sum the density. Train with random continuous strides (cache strides 1, 1.5, 2, 3, 4, 6, 8), generalising the 2× pyramid-shift view | Medium (the mechanism behind the Fig. 4 speed robustness; not isolated by any ablation) | Medium (extra encoder passes) |
| 5 | **Gated mixture of density heads with diversity loss** (S&S Eq. 2–5, Tab. 3, Fig. 4) | dataset/class tempo priors; weak FiLM conditioning | Replace or augment FiLM with P density heads plus a softmax gate (from dataset token + pooled video, optionally the VLM embedding) and a cosine diversity loss. P ≈ number of tempo "types"; S&S picked 24 on UCFRep and 41 on Countix | Medium (Countix 0.371 → 0.324 from L_div) | Low |
| 6 | **High-count / fast-tempo augmentation and test-like dev selection** (inverse of IVAC RCA; S&S stride negatives) | count-distribution shift (dev 12% vs test 21% ≥11) | Time-compress slow training videos (stride ×2/×3 on cached tokens), concatenate annotated segments, and select models on a dev split stratified to about 21% ≥11 (or reweight dev by count bin) | Low-medium (RCA effects are mixed; our dev/test gap of 0.124 vs 0.206 is the motivation) | Low |
| 7 | **≥2-cycles prior inside the segment** (Context `min_cycles=2`) | 0/1 predictions on 2–3-count videos (34% of test) | Diagnostic first: count test or dev predictions with round(p) < 2. If they matter, clamp to 2 and disclose it as a dataset-protocol prior; Context and S&S both bake this in | High for the diagnostic | Trivial |
| 8 | **Mismatched-prompt negatives / AdaLN** (OVRCounter §4) | weak VLM FiLM | Train with wrong-class prompts → zero density, plus a CLIP-style alignment of the pooled video and the VLM text token | Medium (makes conditioning matter; no accuracy gain shown) | Low |
| 9 | Partial encoder adaptation (Context Tab. 2; ESCounts "Context (F)") | general UCFRep gap | LoRA or last-block tuning of ST-MAE on the joint data | Strongest documented factor, but it changes the frozen-encoder story | High |
| 10 | Pull loss on cycle tokens (IVAC) | cycle/phase consistency | Auxiliary cosine pull of per-cycle pooled tokens to their mean (UCFRep and RepCount only) | Weak | Low |

Not worth transferring: S&S audio fusion (UCFRep is evaluated sight-only); Context's 74-forward exhaustive search (use cached multi-stride tokens instead).

## 8. Cheap diagnostics before building anything

1. Bin current dev and test errors by GT period in tokens for each stride (T/(2s)). If the error jumps where T/(2s) < 2, that confirms the aliasing hypothesis behind #1, #4 and #6.
2. Per-stride counts: does the stride-8 or stride-4 map alone give ≈0.5× on count≥11 videos?
3. Fraction of predictions with round(p) < 2 (#7).
4. Oracle stride: pick the stride from GT period per video. If MAE drops a lot, #1/#4 have headroom; if not, the problem is in the head or features.
5. Ratio ΣD / (L/T_gt) histogram: mass near 0.5 or 2 sizes the gain available from #3.

## 9. Unverified or open

* S&S on UCFRep: counting window (segment or whole clip), exact split (106 or 105), whether the stride module was used (App. H implies yes), and the set used for the P=24 search.
* Context pretraining: the paper says ActivityNet, the README says Kinetics checkpoint.
* S&S Table 3: the caption says Countix-AV, but the sight column equals the Countix result in Table 5 (0.314/0.498), so the sight column is likely Countix.
* The S&S Countix test subset size is unreported; the code reads `*_examples_clean.csv`.
* OVR-Kinetics: overlap with Countix test IDs is unchecked.
