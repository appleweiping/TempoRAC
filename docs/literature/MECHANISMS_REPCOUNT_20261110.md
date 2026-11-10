# Why PAMS and PoseRAC work on RepCount-A, and what TempoRAC can borrow

Date: 2026-11-10 (written 2026-11-09). Read-only literature analysis; no experiment was run
and no test label was opened. All numbers below are the authors' own unless marked.

Sources read:

- PAMS, CVPR 2026 Findings, pp. 8143-8152 (CVF open access), plus supplement. The plain-text
  dumps garble the column order of Tables 1-2, so the tables were re-extracted from the PDFs
  with `pdftotext -table`.
- PoseRAC-v1: arXiv 2303.08450v2 (2023).
- PoseRAC-ICONIP'24: "PoseRAC: Enhancing Repetitive Action Counting with Salient Poses",
  LNCS 15290, pp. 255-270, DOI 10.1007/978-981-96-6588-4_18 (full PDF from the PKU ADSP lab
  page).
- Released `MiracleDance/PoseRAC/eval.py`.
- In-repo audits: `docs/baselines/POSERAC_V1_OFFICIAL_AUDIT.md` and
  `docs/baselines/POSERAC_ICONIP24_AUDIT.md`.
- Joint-angle PoseRAC (arXiv 2308.08632): one line only.
- LMRL, "Localization-Aware Multi-Scale Representation Learning for RAC" (arXiv 2501.07312):
  brief section.
- Excluded: `ittakestwo.txt` is "It Takes Two: A Duet of Periodicity and Directionality for
  Burst Flicker Removal" (CVPR). It is image restoration, not RAC.
- In-repo PAMS reconstruction: `results/dev-negative/README.md` and
  `results/dev-negative/pams_full_multiscale_readout_diagnosis.md`.

---

## 0. Bottom line

1. **PAMS (0.129 / 0.842) is mostly a read-out story, and it runs on skeletons.** Its own
   ablation (Table 2) credits the single largest RepCount gain to inference-time,
   period-adaptive peak counting:
   - the adaptive counter alone takes the baseline from 0.230 to 0.167, and the multi-expert
     vote takes it to 0.153;
   - the self-supervised loss alone reaches 0.184.

   Every component is normalised by a per-video period estimate: training windows, smoothing,
   peak spacing, and the FFT reference count.
2. **PAMS components are sub-additive, not "synergistic" as claimed.** On RepCount the
   individual MAE gains sum to -0.123, but the full model gains only -0.101. The same holds
   for OBO and for Countix MAE.
3. **PAMS numbers are unverified and partly odd.**
   - Zero-shot Countix→RepCount equals in-domain to three decimals.
   - Zero-shot RepCount→UCFRep (0.156) beats in-domain UCFRep (0.208).
   - The paper never says how the Period Head is trained or which pose extractor is used.
   - The in-repo reconstruction failed on UCFRep dev84 (best NMAE about 0.63-0.73; the
     "full" variant collapsed to a positional waveform).

   Borrow the ideas, not the numbers.
4. **PoseRAC works because counting is turned into per-frame two-state classification plus
   an ordered hysteresis trigger.** Most of its accuracy comes from how well the states are
   recognised: the pose estimator and loss choices move MAE by 0.05-0.12, and an image-level
   classifier with the same trigger collapses to MAE 0.72-0.76.
5. **PoseRAC's headline number is not comparable.**
   - v1's released evaluator picks the class channel whose count is closest to the
     ground-truth count (a GT-count oracle).
   - ICONIP'24 instead uses a Kinetics-700 UniFormerV2-L recogniser, 1k ControlNet-synthesised
     images per salient pose, and 200 real frames per action.
   - Its UCFRep v1 number is measured on a 21-video, 5-class subset.
6. **Most transferable to TempoRAC** (detail in §4):
   - (a) a period-normalised phase-contrastive auxiliary loss with explicit **anti-phase
     negatives**;
   - (b) exact cross-tempo consistency between our two pyramid views (PAMS's multi-scale
     consistency, made exact);
   - (c) a class-agnostic two-state / phase head that gives an ordered, discrete second count;
   - (d) count-stratified and octave-error reporting.

   Post-hoc octave voting alone is risky: our own WARP-PHASE label-free half/double selector
   failed on symmetric motions (`CLAIMS_FROM_RESULTS_WARP_PHASE.md`).

---

## 1. PAMS (Gao, Li, Li; CVPR 2026 Findings)

### 1.1 Mechanism

#### Input and encoder

- **Input:** skeleton `X ∈ R^{T×K×3}`, flattened and normalised per frame. The paper names
  neither the pose extractor nor K. The 3-D coordinates are consistent with
  MediaPipe/BlazePose 33×3, which is what PoseRAC uses.
- **Augmentation:** rotation, scaling, jitter.
- **Encoder E:** 4-layer Transformer (d=512, 16 heads, FFN 2048, sinusoidal PE, about 12M
  parameters). It outputs L2-normalised per-frame embeddings `z_t` (Sec. 3.3).

#### Training loss: "PAMS TCC" (Sec. 3.1.2), applied to `z_t` rather than to a 1-D signal

1. **Period estimate.** `T_est` is taken from the FFT of the autocorrelation of the current
   embeddings (stop-gradient), or from a pose-energy proxy.
2. **Multi-scale windows.** `W_k = round(s_k·T_est)` with `s ∈ {0.5, 1.0, 1.5}`. The motivation
   for period-adaptive windows is that a fixed window is "too short to capture a full cycle of
   slower ones".
3. **Positives** for anchor `z_i`:
   - temporally adjacent frames (local smoothness);
   - one "cyclically corresponding" frame chosen by the highest self-similarity (TSM) inside
     the window.
4. **Negatives:** other temporal locations, other videos, and hard negatives sampled from other
   K-means clusters. Clusters are re-computed every 5 epochs, and the number of negatives
   matches the number of positives (Sec. 3.1.3).
5. **Per-window loss:** multi-positive InfoNCE (τ=0.1), averaged over the three scales.
6. **Baseline in Table 2:** the same encoder trained with plain TCC.

#### Read-out

- **Period Head H:** a 2-layer MLP (hidden 128) maps `z_t` to a scalar "period stream"
  `p_t`. **The paper does not say how H is trained**; it says only that H is used for
  inference.
- **Multi-Expert Consensus (Alg. 1, Sec. 3.2):**
  1. The FFT of `p` gives `T_est` and a reference count `c_ref = T/T_est`.
  2. Three experts (Fast/Medium/Slow) each do the following:
     - Gaussian-smooth `p`. The supplement names a "Sigma Multiplier" and a "Distance
       Multiplier", so smoothing and minimum peak spacing are presumably multiples of `T_est`;
       the values are not disclosed.
     - Detect peaks above a dynamic height threshold
       `θ[t] = w_l(μ+kσ)_long + (1-w_l)(μ+kσ)_short`, i.e. a local z-score, which is
       amplitude-invariant.
     - Require a prominence of at least `f_p·range`.
  3. The final count is the majority count across experts. If all three disagree, the count
     closest to `c_ref` is used.
- **Output:** an integer count.

#### Phase handling

- PAMS has no explicit phase variable.
- Phase alignment is implicit in the positive selection. Fig. 5 shows embeddings tracing a
  closed loop per cycle.

### 1.2 Which mechanism matters (Table 2, re-extracted; MAE/OBO)

| Configuration (Table 2) | RepCount | Countix | UCFRep |
|---|---|---|---|
| Baseline (plain TCC) | 0.230 / 0.543 | 0.381 / 0.501 | 0.340 / 0.423 |
| + PAMS loss, single scale | 0.201 / 0.612 | 0.234 / 0.571 | 0.306 / 0.485 |
| + PAMS loss, multi-scale | 0.184 / 0.674 | 0.196 / 0.633 | 0.266 / 0.578 |
| + adaptive counter, 1 expert (baseline features) | 0.167 / 0.706 | 0.228 / 0.605 | 0.293 / 0.626 |
| + adaptive counter, 3 experts (baseline features) | 0.153 / 0.753 | 0.143 / 0.727 | 0.245 / 0.645 |
| Full | **0.129 / 0.842** | **0.098 / 0.879** | **0.208 / 0.686** |

What the table shows on RepCount:

- The period-adaptive dynamic-threshold counter is the largest single lever (-0.063 MAE /
  +0.163 OBO).
- Period-adaptive windows in the loss give -0.029. Making the loss multi-scale adds a further
  -0.017 / +0.062.
- Multi-expert voting adds -0.014 on RepCount, but -0.085 on Countix (0.228→0.143). Octave
  and scale consensus therefore matters most where tempi are most diverse.
- The gains are sub-additive. MAE: -0.046 + -0.077 = -0.123 if additive, versus -0.101 for the
  full model. OBO: +0.341 if additive, versus +0.299. This contradicts the claim in Sec. 4.3.

Supplement evidence relevant to our failure modes:

- **Off-by-N accuracy (Supp. T3).**
  - Within 0: 0.526; within 1: 0.842; within 3: 0.901; within 5: 0.9145.
  - So about 8.5% of the 152 test videos are off by more than 5, even for PAMS.
  - Exact-count accuracy (52.6%) is what drives the high OBO.
- **Bland-Altman (Supp. Fig. 11).** Mean bias is -1.36 (systematic under-count), with
  limits of agreement [-14.5, +11.8]. The outliers are high-count videos.
- **Accuracy by count range under clean input (Supp. T5).**

  | Count range | MAE / OBO |
  |---|---|
  | 0-5 | 0.127 / 0.956 |
  | 6-10 | 0.087 / 0.867 |
  | 11-20 | 0.140 / 0.824 |
  | 21-40 | 0.111 / 0.759 |
  | 41+ | 0.173 / **0.615** |

  The high-count weakness is shared with us.
- **Local tempo fluctuation (Supp. T4).** Speeding up or slowing down a random 5-7 s segment
  by 0.5-1.5× gives 0.134 / 0.745. MAE barely moves, but OBO falls by 0.10, mostly in
  videos with 11+ repetitions.
- **Training-data fraction (Supp. T6).** 25%: 0.246 / 0.663; 50%: 0.141 / 0.761;
  100%: 0.129 / 0.842. The label-free objective still depends strongly on data volume.
- **Leave-one-category-out (Supp. T8, 8 categories).** The hardest held-out category is
  bench_pressing (0.251 / 0.625); the easiest are jump_jack (0.035) and pommelhorse (0.032).
- **Global speed change (Sec. 4.4, Fig. 6).** Over 0.1-2.0× speed, MAE stays at 0.12-0.13 and
  OBO at 0.83-0.84 (plot only).
- **Octave example (Fig. 4).** On a video with 45 repetitions, the Fast and Medium experts both
  count 45 and the Slow expert counts 20. The vote discards the outlier. This is exactly our
  half-frequency aliasing case.

### 1.3 Protocol and comparability

- **Modality.**
  - PAMS uses skeletons only. Pose extraction, keypoint layout, frame resampling, and handling
    of frames without a detected pose are all unstated.
  - Countix contains many non-human or partial-body repetitions, yet PAMS reports 0.098 / 0.879
    on it without explaining how such videos are handled or which Countix subset was used.
- **Splits.** The supplement uses the RepCount test set with N=152. The leave-one-out table
  lists only the 8 PoseRAC-pose classes; battle_rope and others are absent.
- **Tuning risk.** About 5 inference hyperparameters (σ and distance multipliers, height
  factor k, prominence f_p, long-window weight w_l) are analysed only on "the RepCount
  dataset" (Supp. 7.10). The split is unspecified, so test-set tuning cannot be ruled out.
- **Inconsistencies.**
  - The supplement cites "Table 3 of the main paper" and a "25% data" result in the main text;
    neither exists in the final paper.
  - Per-scale losses are described as both summed and averaged.
  - The supplement's sensitivity study varies a fixed "TCC Window Size" (20 or 30), which
    conflicts with period-adaptive windows.
- **Table 1 oddities.**
  - Countix→RepCount equals in-domain (0.129 / 0.842).
  - RepCount→UCFRep (0.156 / 0.762) and Countix→UCFRep (0.174 / 0.723) both beat in-domain
    UCFRep (0.208 / 0.686).
  - The SPKDB-Net row repeats the same Countix and UCFRep values.
  - The "PoseRAC (ICONIP'24)" row (0.236 / 0.560, 0.387 / 0.497, 0.312 / 0.452) does not match
    ICONIP'24 Table 3 (0.226 / 0.570, 0.305 / 0.530, 0.146 / 0.803). Its RepCount and UCFRep
    values are the v1 arXiv numbers, and the UCFRep one comes from a 21-video subset (§2.3).
- **Output type.** PAMS outputs integers, which inflates exact-count accuracy and OBO
  relative to an unrounded continuous density sum. Check that our OBO rounds predictions,
  as ESCounts does.
- **Reproducibility.** The in-repo independent reconstruction on UCFRep dev84 (MediaPipe
  33×3, 256 frames) is in `results/dev-negative/README.md`:
  - the literal variant reached NMAE 0.63-0.73;
  - the trained-head variant collapsed: every fully valid video predicted 31, because the
    sinusoidal PE drove an 8-frame positional waveform
    (`pams_full_multiscale_readout_diagnosis.md`);
  - the Table 2 proxy pattern passed 0/3 seeds.

### 1.4 Our reading of why PAMS tops RepCount-A

1. **The domain fits the modality.** RepCount is single-person fitness video, so skeletons
   strip background. Even the TCC baseline (0.230) matches supervised skeleton methods
   (0.21-0.24).
2. **Everything is expressed in period units.** Loss windows, smoothing, peak spacing, and the
   reference count all scale with `T_est`, which makes the pipeline approximately
   tempo-invariant.
3. **Two independent octave guards:**
   - experts with different smoothing scales vote;
   - the FFT reference breaks ties.
4. **Amplitude-invariant integer read-out.** Local z-score thresholds count oscillations, not
   mass. This is why exact-count accuracy is 52.6%.
5. **Possible protocol advantages** (unverified): see §1.3.

### 1.5 What transfers to an RGB-token density counter

- **P1. Phase-contrastive auxiliary loss on the temporal-encoder outputs `h_t`.**
  - Recipe:
    - take the period from annotations rather than PAMS's self-estimate: RepCount and UCFRep
      have per-cycle boundaries, and Countix gives segment length / count;
    - positives at `t ± kT` (same phase) plus ±1 grid step;
    - **negatives at `t ± T/2` (anti-phase) and `t ± T/4`**, plus other videos;
    - windows `{0.5, 1, 1.5}·T`; InfoNCE with τ=0.1.
  - Why it fits our failures:
    - An anti-phase negative separates "half-way down" from "half-way up". These have the
      same pose and opposite velocity. Features that encode motion magnitude have period T/2,
      and this is the plausible cause of our 2× over-count on symmetric motions.
    - Positives exactly one period apart push the embedding period to T, which also counters
      half-frequency aliasing.
  - PAMS's own positive selection (highest similarity inside the window) could pick the T/2
    frame on such motions and reinforce the error. Annotation-derived periods avoid this.
  - Evidence: Table 2 rows 2-3 (RepCount -0.046 MAE / +0.131 OBO; Countix -0.185 / +0.132).
  - Risk: positional collapse if `h_t` carries absolute PE. Gate on static, time-shuffled, and
    synthetic-period inputs first, as in the in-repo diagnosis.
- **P2. Exact cross-tempo consistency (PAMS multi-scale consistency, made exact).**
  - The pyramid shift gives an exact 2× view, so we can require
    `d_2×(t') ≈ 2·d(2t')`, `Σd_2× = Σd`, and same-phase positives across views.
  - Labels are not needed, so the loss can run on unlabeled training-split videos.
  - Today the 2× view is used only as augmentation and as a second test view. A
    prediction-agreement term adds a constraint that augmentation does not.
  - Targets: the UCFRep dev→test gap (regularisation) and fast-repetition under-counting (the
    fast view is tied to the easier slow view).
  - Evidence: Table 2 multi-scale increment (RepCount -0.017 / +0.062; Countix -0.038 / +0.062)
    and the tempo-robustness results (Fig. 6, Supp. T4).
- **P3. Period-normalised density targets.**
  - If the per-cycle Gaussian σ is fixed in frames, short cycles merge into a featureless
    plateau, which matches our "flat and low" density.
  - Make σ ∝ cycle length (e.g. 0.2·len), each Gaussian unit-mass. This is the target-side
    analogue of PAMS's period-adaptive windows (Sec. 3.1.2).
- **P4. An oscillation-based second count (amplitude-invariant).**
  - Compute `c_spec = valid_duration / T_est`, with `T_est` from the autocorrelation of `h_t`
    or of the density.
  - It counts how often the signal oscillates rather than how much mass it has, so it helps
    exactly when density is flat and low but still oscillating.
  - Combine it with the density sums of both pyramid views.
  - Prefer a learned or validated arbiter to PAMS's hand-written vote. Our WARP-PHASE
    label-free half/double selector failed its stability gate on symmetric motions:
    agreement 246/1000, and 50/66 symmetric half/double selections
    (`CLAIMS_FROM_RESULTS_WARP_PHASE.md`).
  - Evidence: Table 2 rows 4-5 (RepCount 0.230→0.167→0.153; Countix 0.381→0.228→0.143) and
    Fig. 4.
- **P5. Self-supervised pre-training of the temporal encoder** on unlabeled video, using P1
  with self-estimated periods. Do this only after the P1 gates pass. Supp. T6 shows the
  payoff scales with data volume, but the in-repo reconstruction shows that a self-estimated
  period is the fragile part.

---

## 2. PoseRAC: v1 (arXiv 2303.08450) and ICONIP'24 (LNCS 15290)

### 2.1 Mechanism

#### v1 (2023)

- **Pose extraction:** BlazePose 33×3 keypoints per frame.
- **Model:** keypoint-MLP embedding, then a 6-layer Transformer whose sequence runs over
  keypoints. It is spatial attention within one frame; **there is no temporal modelling**.
  A 2-layer MLP follows, with a sigmoid per class.
- **Targets:** per class, salient pose I → 1 and pose II → 0.
- **Annotation:** Pose Saliency Annotation. Two manually defined salient poses per class are
  marked in the training videos, e.g. squat = upright / squatting.
- **Training:** only salient frames are used, with BCE + λ·cosine-triplet loss (λ=0.01).
- **Counting:** the Action-trigger state machine. Per the released config (in-repo audit) it
  uses EMA smoothing (momentum 0.4), enter threshold 0.78, and exit threshold 0.4. One count
  is added each time pose I and then pose II are triggered in order.

#### ICONIP'24

- **Pose extraction:** 3D OpenPose.
- **Model:** 6-layer pose-wise encoder, then a 2-layer Transformer decoder in which one action
  query per class produces one score.
- **Targets:** pose I = 1, pose II = 0, non-salient = 0.5.
- **Training data:**
  - 1k ControlNet + Stable Diffusion images per salient pose (the "zero-shot" setting);
  - 200 real salient frames per action (the "few-shot" setting).
- **Class at test time:** given by **UniFormerV2-L (Kinetics-700)**.
- **Counting:** the same training-free two-threshold trigger. It counts when
  "two salient poses appear in sequence" (Fig. 1).
- **Open-set variant:** CLIP text embeddings serve as the action queries.

#### Why it works

- Counting reduces to an ordered two-state detector with hysteresis:
  - jitter between the thresholds is ignored;
  - a pause adds nothing;
  - a half cycle never completes the A→B sequence, so it cannot double-count.
- Accuracy is then limited by how well the states are recognised. ICONIP T9 reports
  salient-pose accuracy of 0.93 on RepCount for the pose Transformer, versus 0.52 and 0.61 for
  the image-level baselines.

### 2.2 Ablations: which part matters

- **Representation is the dominant factor.**
  - Swapping pose+encoder for a per-frame image classifier under the same trigger:
    - v1 T6: ResNet-50 0.556 / 0.181 and ViT-32 0.517 / 0.195, versus 0.236 / 0.560;
    - ICONIP T4 (RepCount val): 0.762 / 0.103 and 0.723 / 0.116, versus 0.221 / 0.576.
  - Encoder type, ICONIP T4: MLP 0.316 / 0.486, 1D-CNN 0.357 / 0.439, Transformer
    0.221 / 0.576.
  - Pose estimator, ICONIP T8: HRNet 0.269 / 0.496, BlazePose 0.240 / 0.561, ViTPose
    0.243 / 0.551, RTMPose 0.241 / 0.562, 3D-OpenPose 0.221 / 0.576. v1 T4: BlazePose
    0.236 / 0.560 vs ViTPose 0.305 / 0.463.
- **Metric learning.**
  - v1 T5: BCE only 0.317 / 0.486; with triplet at λ=0.01 / 0.05 / 0.1: 0.236 / 0.560,
    0.289 / 0.501, 0.328 / 0.462.
  - ICONIP T5: BCE only 0.339 / 0.450; with triplet at λ=0.01: 0.221 / 0.576.
  - Results are highly sensitive to λ.
- **Training data (ICONIP Fig. 3).** OBO rises from about 0.4 to about 0.6 as synthetic images
  go from 0.25k to 1k per pose.
- **Supervision setting (ICONIP T3, RepCount test).** Zero-shot 0.328 / 0.425; few-shot
  0.226 / 0.570.
- **Text priors (ICONIP T6).** CLIP-text queries hurt closed-set accuracy (RepCount
  0.319 / 0.483 vs 0.221 / 0.576; UCFRep 0.287 / 0.639 vs 0.146 / 0.803) and are needed only
  for open-set. This is consistent with our weak Qwen FiLM conditioning.
- **Joint angles (arXiv 2308.08632, T1-2).** Adding 5 averaged joint angles moves the result
  from 0.236 / 0.559 to 0.211 / 0.599. More state-informative input means a better trigger.

### 2.3 Protocol and comparability

- **v1 evaluator leaks the ground-truth count.** The released `eval.py` computes a count for
  each of the 8 class channels and keeps the channel with the smallest
  `|gt-pred|/gt`. This was confirmed against the GitHub source and the in-repo audit.
  v1's 0.236 / 0.560 is therefore oracle-assisted. The in-repo oracle-free run of the official
  checkpoint on UCFRep dev84 gave NMAE 0.706 / OBO 0.238 (a different domain, but indicative).
- **Splits.**
  - RepCount-pose: 8 classes, 487 training videos (re-annotated), 151 test videos (v1 T1).
  - UCFRep-pose (v1): 5 classes, 89 training and **21 test** videos, not the standard
    105-video split.
  - ICONIP'24 uses the standard 152 / 105 / Countix splits (T1). Its ablations run on the
    RepCount validation split.
- **Extra resources.**
  - ICONIP'24 relies on the UniFormerV2-L class recogniser and the synthetic data.
  - Both versions require manual per-class salient-pose definitions, so PoseRAC is closed-set.
- **Metric.** MAE is normalised by the ground-truth count (v1 eq. 12; ICONIP eq. 8), and the
  output is an integer.

### 2.4 What transfers to an RGB-token density counter

- **R1. Class-agnostic two-state / phase head, learned from cycle boundaries.** This replaces
  manual salient poses.
  - Targets: per grid step, `(cos φ, sin φ)` with `φ = 2π·(t-s_c)/(e_c-s_c)` inside annotated
    cycle c. Equivalently, state A near cycle boundaries and state B near mid-cycle. On
    Countix, use a uniform phase inside the segment at low weight, or mask it.
  - Read-out:
    - unwrapped-phase winding divided by 2π (differentiable); or
    - a PoseRAC-style A→B→A hysteresis count as a discrete second expert next to the density
      sum.
  - Why it fits our failures:
    - one down+up stroke is one revolution, so symmetric 2× over-counts become a phase error
      that the loss can see;
    - a pause stalls the phase, so no count is added;
    - a discrete read-out targets our OBO gap: 0.72 against PAMS's 0.842 at similar MAE,
      where PAMS achieves 52.6% exact counts.
  - Our tokens differ from PoseRAC's failed image-level baselines: ST-MAE tubelets carry motion
    and the temporal encoder adds context.
  - Evidence: PoseRAC beats TransRAC without any temporal model once the states are
    recognised well (v1 T3; ICONIP T3, T9).
- **R2. Ordered-state constraint as an auxiliary.** Penalise density mass that is not
  bracketed by an A→B→A sequence. This is a soft version of the trigger and suppresses
  half-cycle bumps.
- **Do not transfer:** class-specific states, test-time class recognisers, synthetic pose
  images. They are closed-set and conflict with our class-agnostic, multi-dataset claim.

---

## 3. LMRL (arXiv 2501.07312), brief

### Mechanism

- **Features:** Video Swin, computed on 4-frame windows with stride 1.
- **MPR branch (multi-scale similarity):**
  - learnable bilinear similarity `x_iᵀWx_j`, followed by 1×1 and dilated convolutions;
  - per-scale max-pooling of the similarity map (kernel 2·k) plus self-attention.
- **RFL branch (foreground localisation):** a dilated-residual TCN supervised with
  foreground/background BCE plus a smoothing loss. Labels come for free: frames inside
  annotated cycles are foreground.
- **Triplet loss:** pulls foreground frames together and pushes background away.
- **Count head:** density transformer, trained with `|c-ĉ|/ĉ` + MSE.

### Results

- RepCount-A: 0.420 / 0.382 (Table I). RepCount→UCFRep: 0.712 / 0.395. This is far below
  PAMS and PoseRAC.
- **Loss ablation (Table III).** The check-mark columns are unreadable in the text dump; the
  row assignment below is inferred from the paper's prose.
  - Without the auxiliary losses: 0.4565 / 0.2566.
  - All losses: 0.4200 / 0.3816.
  - Without the density loss: 0.4330 / 0.3223.
  - Localisation accuracy rises from 51 to 78 once localisation is supervised.
- **Similarity ablation (Table IV).** TSM 0.539 / 0.270 → MPR 0.420 / 0.382.

### Transfer

- **L1.** A cheap inside/outside-cycle BCE auxiliary on `h_t` could reduce density mass during
  pauses and non-repetitive segments, and may help the UCFRep dev→test gap.
- The evidence is weak, because absolute accuracy is low.

---

## 4. Failure mode → mechanism map

| Our failure mode | Mechanism to try | Evidence it acts there | Cost |
|---|---|---|---|
| Under-count, fast or high-count (flat/low density) | P3 period-normalised targets; P4 oscillation count; P2 2×-view consistency | PAMS T2 rows 4-5; Supp. T5 shows 41+ is weak even for PAMS (OBO 0.615) | Low |
| Half-frequency aliasing | P1 positives at ±T; P4 with a learned arbiter (not a raw vote) | PAMS Fig. 4 (Slow expert 20 vs 45); T2 Countix multi-expert -0.085 | Medium |
| 2× over-count on symmetric strokes | P1 anti-phase negatives; R1 phase winding; R2 ordered states | PoseRAC trigger (ordered I→II); v1 T3 / ICONIP T3; P1 is our extension (PAMS has no anti-phase negative) | Medium |
| UCFRep dev→test gap | P2 label-free consistency on unlabeled train videos; L1; P5 later | PAMS Supp. T6 (data volume), Supp. T8 (leave-one-out); LMRL T3 (weak) | Medium |
| OBO lag at similar MAE (0.72 vs 0.842) | R1 discrete second count; verify OBO rounding; report Off-by-N and exact-count accuracy | PAMS Supp. T3 (exact 0.526) | Low |

Reporting to add, for comparability with PAMS:

- Off-by-N for N = 0-5 (Supp. T3).
- Accuracy by count range: 0-5, 6-10, 11-20, 21-40, 41+ (Supp. T5).
- Bland-Altman bias and limits of agreement.
- Octave-error rate: share of videos with pred/gt in [0.45, 0.55] or [1.8, 2.2].
- A segment-speed perturbation test (Supp. T4 protocol: one random 5-7 s segment at 0.5-1.5×,
  10 trials per video).

---

## 5. What would make TempoRAC novel relative to them

1. **Class-agnostic RGB counting across all three datasets.**
   - PAMS and PoseRAC both need a detectable human skeleton.
   - PoseRAC also needs a closed set of manually defined per-class salient poses, plus either a
     GT oracle (v1 code) or a large action recogniser (ICONIP).
2. **Tempo handling by construction rather than by estimation.**
   - PAMS derives windows from a noisy self-estimated `T_est` (the fragile part in our
     reconstruction) and fixes octave errors with hand-tuned peak experts.
   - TempoRAC has exact multi-rate tokens and an exact 2× pyramid shift. Making that
     equivalence a training constraint (P2) is an exact, label-free version of PAMS's
     "multi-scale consistency" and can run on unlabeled video.
3. **Learned, end-to-end count read-out** instead of about 5 hand-set inference parameters. A
   learned arbiter over count hypotheses (density per view, oscillation count, phase winding)
   would replace PAMS's majority-vote/FFT rule. The planned LoRA-LLM over temporal tokens is
   one candidate arbiter. A language prior about what one repetition of an action is (e.g.
   down+up) targets the symmetric-stroke ambiguity directly. ICONIP T6 warns that text
   priors can hurt closed-set accuracy, so this must be shown, not assumed.
4. **Explicit, direction-aware phase supervision.**
   - Neither paper uses anti-phase negatives or a continuous phase target.
   - PAMS's phase alignment is implicit, and its similarity-chosen positive can lock onto T/2
     on symmetric motions.
   - PoseRAC's two states are discrete, manual, and class-specific, with a non-differentiable
     trigger.
   - A class-agnostic phase head with a differentiable winding count (R1), together with
     anti-phase contrast (P1), is new relative to both.

---

## 6. Suggested dev-only order (no test use; one change per run)

1. **Diagnostics only:**
   - Off-by-N and count-stratified tables for the current model;
   - octave-error rate;
   - check whether the target σ is fixed and whether OBO uses rounded counts.
2. **P3** period-normalised targets: a one-field change.
3. **P2** exact pyramid-view consistency: labeled data first, then unlabeled train-split videos.
4. **P1** anti-phase contrastive auxiliary. Run the positional-collapse gate (static,
   shuffled, and synthetic inputs) before scoring dev.
5. **R1** phase head as an auxiliary loss, then as a second count expert. Add P4 and a learned
   arbiter only after R1 and P2 are measured.
