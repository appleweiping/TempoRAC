# Why recent RGB / LLM repetition counters work: CountLLM verification and transfer to TempoRAC

Prepared 2026-11-09 (file date 2026-11-10). Read-only literature study.
Sources: text extractions plus PDFs in `D:\Temp\claude\3c515764271446d6\0a02953f-f1e8-44b0-afff-6beb3c0cb0a6\scratchpad\pdfs\`
(countllm_cvf, escounts, d2stx, detrc, fcarac, htrm, mfl, fullres).
Verification: wherever the text extraction lost check-marks or scrambled columns, the table was re-read from a rendered
PDF page: CountLLM T1-T3 and T5-T7, ESCounts T1-T4, D2-STX T1/T2, HTRM T VI, MFL T4/T5/T7, Full-Res T2-T5.
Metric convention everywhere: MAE (lower is better) / OBO (higher is better).

TempoRAC reference (joint training, 5 seeds, test): RepCount-A 0.142/0.72, UCFRep 0.206/0.70, Countix 0.238/0.69.

---

## 0. Summary

1. **CountLLM's gains come mostly from adapting the visual side, not from the LLM alone.** In Table 7, LoRA on the video encoder with a frozen LLM reaches 0.239/0.546. LoRA on the LLM with a frozen encoder reaches only 0.317/0.498, and freezing both gives 0.409/0.357. A linear projector straight into the LLM, without the periodicity transformer and its alignment stages, is the worst setting (0.498/0.260). This matches our negative result with frozen-Qwen FiLM.
2. **Periodicity-aware alignment of the compressor is the second big factor** (Table 6: 0.402/0.368 with no alignment, 0.236/0.551 with P-A only, 0.166/0.631 with both). P-A is in effect a binary classifier: repetitive clips vs non-repetitive clips, scored against **one fixed class-agnostic text embedding**. The generic video-caption alignment (S-A, WebVid) adds little on its own (0.363/0.417).
3. **The answer format matters**: the count is a fixed-width decimal string built from existing digit tokens. Adding 10,000 new count tokens collapses to 0.392/0.387 (Table 5). The fixed description is also worth about 0.14 MAE (removing it gives 0.305/0.476).
4. Among the RGB counters, the best-evidenced cheap mechanisms are:
   - a **count-consistency loss** next to density MSE (ESCounts T3g; FCA-RAC T7);
   - **exemplar cross-attention during training**, with a learned zero-shot latent used at test (ESCounts T3a-d);
   - **dense temporal sampling** (ESCounts T8; Full-Res T3);
   - **local or directional temporal context** against symmetric-motion double counting (HTRM T V; MFL T4; D2-STX T3/T4);
   - **multi-scale or tempo-normalised inputs** (HTRM T VII; FCA-RAC T5).
5. On **UCFRep**, methods that fine-tune or adapt the encoder win consistently. Frozen ST-MAE ViT-B ESCounts gets 0.216/0.704 (TempoRAC is at this level). Fine-tuned Context gets 0.147/0.790 and CountLLM with LoRA-V gets 0.116/0.839. ESCounts says this explicitly, and Context collapses to 0.653/0.372 when its encoder is frozen. TempoRAC's UCFRep gap is most likely an encoder-adaptation gap, not a head or LLM gap.

---

## 1. CountLLM (Yao et al., CVPR 2025): verified from the full text

### 1.1 Architecture (Sec. 3.2, Fig. 2)
- Video encoder: VideoSwin-T or VideoMAE (ref [54] is VideoMAE V2), Kinetics-pretrained.
- **Periodicity transformer** (PT): 12 layers with **64 learnable "periodicity queries"** that cross-attend to the encoder features. This is a Q-Former-style compressor, meant to keep periodic content and discard background and blur.
- Linear projector: maps the queries to 64 "periodic tokens" of LLM width.
- LLM: Vicuna-7B. The count is read **only from the generated text**; there is no density head anywhere.
- LoRA r=16 in the attention layers of **both the video encoder and the LLM** (Sec. 4.1). The PT and projector are trained in stage 3 ("optimize the entire CountLLM"). Full trainability of the PT/projector is implied, not stated.

### 1.2 Instruction template, description, answer (Sec. 3.3, Figs. 2-3)
- Template: `{periodic tokens}, {periodicity description}, {question}, {answer}`.
- **Periodicity description** (paraphrase): repetitive actions recur at periodic time intervals with a consistent motion pattern. Frames look visually similar across repetitions, and each repetition resembles the previous one. Key phrase: "periodic temporal intervals and consistent motion patterns".
  - It was written by GPT-4 under the instruction that it be general and class-agnostic, then manually refined.
  - It is **one fixed text for all videos**. It is not per-video and names neither the action nor the repetition.
- Question: how many times repetitive actions appear in the video; give a numerical result.
- **Answer** `[abcd,e,f]`:
  - `abcd` is the clip count, zero-padded to 4 digits (0000-9999), so no new vocabulary is needed.
  - `e` and `f` flag an incomplete cycle at the clip start and end.
- **Clip aggregation**: the video count is the sum of the clip counts, plus 1 for every adjacent pair where `f`=1 in the earlier clip and `e`=1 in the later one. Fig. 3 example: 9 = 5 + 3 + 1. Partial cycles are therefore counted only when two halves meet.

### 1.3 Progressive training (Sec. 3.4, Fig. 4)
| Stage | Data | Trainable | Objective | Clip length / epochs |
|---|---|---|---|---|
| 1 Semantic-aware (S-A) | WebVid-10M video-caption pairs | PT (video encoder frozen, implied by Fig. 4; LLM unused) | Video-text contrastive, CLIP text encoder; similarity g = max over the 64 queries; softmax with temperature; CE both directions (Eqs. 6-8) | 16 frames, 10 epochs |
| 2 Periodicity-aware (P-A) | "RAC datasets": **positives = clips containing repetitive actions, negatives = clips without** | PT | Sigmoid(g(Z, E_per)) against **the single periodicity-description embedding**, BCE (Eqs. 9-10) | 50 epochs (16 frames implied by wording) |
| 3 Instruction tuning | RAC datasets | LoRA(V), LoRA(L), PT, projector | Token-level CE on the tokenised GT answer (Eq. 11) | 32-frame clips, 50 epochs |

- How the stage-2 negatives are mined is **not specified**. The likely source is the non-repetitive spans of RepCount, but this is not stated.
- **Data in stages 2-3**: "RAC datasets" (RepCount, UCFRep, Countix). The paper does not say whether training was joint.
  - Hyper-parameters are tuned per dataset (RepCount and Countix on val; UCFRep reported on its val split).
  - Table 3 runs a separate "trained on RepCount" protocol.
  - Together these imply **per-dataset training** for Tables 1-2.
- How per-clip counts and e/f flags are derived for **Countix**, which has no cycle boundaries, is **not explained**.

### 1.4 Frames, clips, hyper-parameters
- Training uses 16-frame clips (stages 1-2) and 32-frame clips (stage 3).
- At test time, "frames are sampled at intervals to ensure each clip contains 32 frames". Not reported: the clip duration in seconds, the stride or fps, the number of clips per video, and any overlap.
- Reported: 8x A100, LoRA r=16, 64 queries, a 12-layer PT, and the epoch counts above.
- **Not reported**: learning rates, batch sizes, optimiser, schedule, LoRA alpha, decoding settings, number of seeds. Code is not mentioned in the paper.

### 1.5 Exact ablation tables (all on the **RepCount val set**; the full model is 0.166/0.631 on val, 0.162/0.639 on test)

**Table 5: instruction conversation**
| Setting | MAE | OBO |
|---|---|---|
| w/o periodicity description | 0.305 | 0.476 |
| w/o decimal string answers (new learnable tokens 0000-9999) | 0.392 | 0.387 |
| All (CountLLM) | **0.166** | **0.631** |

**Table 6: progressive training (S-A = semantic-aware, P-A = periodicity-aware)**
| S-A | P-A | MAE | OBO |
|---|---|---|---|
| - | - | 0.402 | 0.368 |
| yes | - | 0.363 | 0.417 |
| - | yes | 0.236 | 0.551 |
| yes | yes | **0.166** | **0.631** |

**Table 7: model design (LoRA on Phi_V = video encoder, Phi_L = LLM)**
| Block | LoRA V | LoRA L | MAE | OBO | Reading |
|---|---|---|---|---|---|
| w/o Periodicity Transformer (linear projector only; S-A and P-A removed) | yes | yes | 0.498 | 0.260 | worst row |
| w/ PT (S-A and P-A included) | - | - | 0.409 | 0.357 | **frozen encoder + frozen LLM** |
| w/ PT | yes | - | 0.239 | 0.546 | **frozen LLM**, encoder LoRA |
| w/ PT | - | yes | 0.317 | 0.498 | frozen encoder, LLM LoRA |
| w/ PT | yes | yes | **0.166** | **0.631** | full |

Tables 5 and 6 implicitly use the full LoRA(V+L) setting: their "All" row equals the full model.

**Main results, for context**
- T1, RepCount test / UCFRep val:
  - CountLLM VideoMAE 0.162/0.639 and 0.116/0.839.
  - CountLLM VSwinT 0.209/0.552 and 0.153/0.802.
  - ESCounts (as listed) 0.213/0.563 and 0.216/0.704.
- T2, Countix: VideoMAE 0.215/0.801, VSwinT 0.246/0.706.
- T3, cross-dataset: RepCount to UCFRep 0.139/0.816; RepCount to Countix 0.281/0.730.
- T4, OVR-Kinetics: 0.26/0.65 against OVRCounter 0.39/0.59.

### 1.6 What the ablations imply
- **Visual adaptation is the largest single factor.** LoRA-V alone adds +0.189 OBO over frozen-both; LoRA-L alone adds +0.141. Visual tokens that are not adapted and not periodicity-aligned give the LLM little to reason over.
- **Periodicity alignment is a discriminative "is this repetitive?" pretext task.** Generic semantic alignment helps only in combination with it. This is weak evidence that semantic or caption-style grounding is not what counting needs.
- **The constant description works like a task-specifying prefix.** No experiment separates "text semantics" from "any fixed prefix / soft prompt".
- **Clip-level answers keep the LLM's numeric range small.** The incomplete-cycle flags exist only because of clip splitting.

### 1.7 Pitfalls and inconsistencies in CountLLM
- All ablations are on val. Main-table baselines are copied, with protocols mixed:
  - The "Full" row in T1 (UCFRep 0.461/0.333) is Full-Res's **RepCount-to-UCFRep cross-dataset** number, not in-domain.
  - The RepNet Countix row lists OBO 0.303, whereas RepNet/ESCounts report 0.697 (MAE 0.364 is the same).
- The unspecified details listed in 1.3-1.4 make exact reproduction difficult: test clip length and stride, Countix clip labels and flags, stage-2 negatives, LR and batch size.
- The limitations paragraph names compute cost, and D2-STX (Sec. 4.7) contrasts CountLLM's 8x A100 with its own single 4090 and 10-20 ms inference.

---

## 2. RGB counters: mechanism, read-out, encoder, sampling, cues, key ablations

### 2.1 ESCounts (Sinha et al., ACCV 2024)
- **Mechanism**: encoder-decoder.
  - Video tokens are kept at full spatio-temporal resolution: each 64-frame window gives 8x14x14 = 1,568 tokens, C=512.
  - Decoder: L=2 cross-attention blocks (video self-attention, then cross-attention to S exemplar encodings, averaged), then L'=3 shifted 3-D window self-attention blocks (window 4x7x7).
  - A learned latent z0 stands in for exemplars and is the only thing used at test (zero-shot).
- **Exemplars**: annotated single repetitions, 16 frames sampled uniformly between start and end. They come from the same video, or (with p=0.4) from another video of the same class. Training samples |S| from {0,1,2}.
- **Read-out**: sum of a 1-D density. Loss = density MSE + normalised count MAE |c - sum d|/c. Density target: Gaussians centred on each cycle with sigma = 0.5.
- **Encoder**: **frozen ST-MAE ViT-B, K400** (ref [17] = Feichtenhofer et al., the same family as TempoRAC's backbone). Also tested with frozen SwinT, RX3D101 and R(2+1)D18.
- **Sampling**: every 4th frame, 64-frame sliding windows, 16x224x224 per window. Time-shift TTA averages |K|=4 shifted density maps.
- **Key ablations** (RepCount test, Table 3 and appendix):

| Ablation | Result |
|---|---|
| T3a: self-attention-only decoder vs exemplar cross-attention | 0.273/0.470 vs **0.213/0.563** |
| T3b: |S|=0 vs {0,1} vs {0,1,2} vs {0..4} | 0.240/0.519, 0.228/0.546, **0.213/0.563**, 0.215/0.559 |
| T3c: exemplars only from the same video / from another video of a different class / from another video of the same class | 0.224/0.521, 0.270/0.464, 0.213/0.563 |
| T3d: p=0 / 0.4 / 1.0 | 0.240/0.545, 0.213/0.563, 0.245/0.532 |
| T3e: sigma variable / 0 / 0.5 / 1.0 | 0.301/0.457, 0.241/0.510, **0.213/0.563**, 0.223/0.552 |
| T3f: shifts |K|=1 vs 4 | 0.221/0.552 vs 0.213/0.563 |
| T3g: density MSE only vs MSE + count MAE | 0.273/0.532 vs **0.213/0.563** |
| T8: sample every 4 / 8 / 16 / 32 frames | 0.213/0.563, 0.268/0.521, 0.296/0.482, 0.346/0.444 |
| T7: window (4,7,7) vs full attention | 0.213/0.563 vs 0.227/0.533 |
| T4: test-time exemplars, 3 shots from the same video | RepCount 0.206/0.596; UCFRep 0.198/0.723 (zero-shot 0.216/0.704) |

- Per-bin analysis (Fig. 6) uses equal-size bins:
  - Errors grow with count.
  - Density prediction is hardest for the shortest repetitions (XS bin < 0.96 s, i.e. about 20% of RepCount test).
  - Longer videos are harder.
- **Countix**: there are no cycle boundaries, so pseudo-labels divide each video uniformly by the GT count. Results: 0.276/0.673 (ViT-B), 0.293/0.701 (R(2+1)D18). Localisation on Countix is weak because of the pseudo-labels.
- **UCFRep**: reported on val. Results: 0.216/0.704 (ViT-B), 0.247/0.731 (RX3D101).
  - Methods that **fine-tune** their encoders on UCFRep are better: Context 0.147/0.790, Sight and Sound 0.143/0.800.
  - Context with a frozen encoder drops to 0.653/0.372 (T1c).
  - Cross-dataset RepCount to UCFRep: 0.317/0.571.

### 2.2 D2-STX (Wang et al., CVPR 2026 Findings)
- **Mechanism**: ESCounts-style video branch plus a **pose branch**: BlazePose 33 keypoints mapped to 25, encoded by an NTU-pretrained CTR-GCN, with exemplar cross-attention per branch. Fusion is decoupled:
  - frame-wise spatial cross-attention (joints <-> pixels, per frame);
  - then joint-wise or pixel-wise temporal cross-attention, which reuses the spatial attention weights to build per-joint video sequences (or per-pixel pose sequences).
  - Branch densities are fused as alpha D_p + (1 - alpha) D_v with a learnable alpha.
- **Read-out**: sum of the density. Loss = masked density MSE (20% of target positions masked) + normalised count MAE with epsilon = 0.1.
- **Encoders**: SwinT, RX3D101 or R(2+1)D18, chosen per dataset to match baselines; CTR-GCN for pose. Training has three stages (video branch 50 epochs, pose branch 35, fusion 25), with the other parts frozen at each stage. A frozen video encoder (ESCounts practice) is implied but not stated.
- **Sampling**: 64-frame sliding window, "16-frame intervals"; 16x224x224 input to 8x7x7 tokens.
- **Key ablations** (SwinT):

| Ablation | Result |
|---|---|
| T3 RepCount: pose / video / pose+video | 0.843/0.173, 0.298/0.403, **0.255/0.500** |
| T3 UCFRep: pose / video / pose+video | 0.464/0.394, 0.324/0.614, **0.249/0.670** |
| T4 fusion: concat / add / decision / video-query cross-attention / **pose-query cross-attention** / bidirectional | 0.347/0.393, 0.288/0.447, 0.264/0.453, 0.284/0.447, **0.255/0.500**, 0.250/0.493 |

- Pose alone is poor; pose used as **query** into dense video features is the best fusion.
- **Results** (T1/T2):
  - UCFRep 0.236/0.777 (RX3D101 + pose).
  - Countix 0.278/0.734 (R(2+1)D18 + pose).
  - Cross-dataset RepCount to UCFRep 0.297/0.643; RepCount to Countix 0.359/0.605.
  - The authors concede that CountLLM is better (Sec. 4.7).

### 2.3 DeTRC (Ma et al., "Efficient Action Counting with Dynamic Queries")
- **Mechanism**: DETR-style **set prediction of cycles**.
  - Per-frame features (T = 512 frames, no down-sampling) go through a 2-layer encoder.
  - Query selection keeps the top Q=40 encoder tokens by "repetitive" confidence. **Dynamic Action Query**: these selected encoder features initialise the decoder action queries (4 decoder layers).
  - Heads predict a class ("repetitive" vs "others") and (midpoint, duration). Training uses Hungarian matching with L1 + gIoU.
  - **Inter-query contrastive learning**: InfoNCE pulls together queries classified as repetitive and pushes away the rest.
- **Read-out**: count = number of queries with p > tau (tau = 0.2). Q=40 caps the count per window; how longer videos are handled is not stated.
- **Encoder**: TSN, I3D or ViT. Frozen vs fine-tuned is not stated (batch 64 at T=512 suggests pre-extracted features). AdamW, lr 2e-3, 80 epochs.
- **Ablations** (RepCount test, TSN backbone, T4): w/o DAQ 0.3542/0.4172; w/o ICL 0.4035/0.4040; full 0.2809/0.4570. Best backbone (ViT, T1): 0.2622/0.5430.
- **Short / medium / long periods** (<30 frames / 30-60 / >60 frames per cycle), OBO: ViT gets 0.2558 / 0.5909 / 0.7031. TransRAC gets 0.0233 / 0.2955 / 0.4375. Most of the advantage over TransRAC is on fast cycles, from not down-sampling.
- **Countix**: not used. **UCFRep**: cross-dataset only, ViT 0.5435/0.4184 (T2). Fig. 5 notes that UCFRep annotates only cycle start points.

### 2.4 FCA-RAC (Lu et al., 2024)
- **Protocol caveat**: the start and end of the **first cycle are given at test time**, so results are not comparable with standard RAC.
- **Mechanisms**:
  - **Adaptive sampling**: each video is resampled so that the first cycle spans k=4 frames (rate R = 4/N). This normalises tempo across videos.
  - **MTGC**: the first-cycle features, interpolated to 3, 4 and 5 steps, are convolved as kernels across the video.
  - **TKA**: the top-K=10 nearest first-cycle embeddings from the training set act as extra kernels, fused by attention pooling.
- **Loss**: density MSE on the first cycle only, plus lambda x normalised count MAE (lambda = 10). There is no full-density supervision.
- **Encoder**: Video Swin-T, frozen (GPU memory). Sliding 4-frame clips with step 2; 3-D conv followed by spatial pooling.
- **Ablations**:

| Ablation | Result |
|---|---|
| T5 RepCount: FC-V baseline / single kernel (4) / MTGC 3-5 / MTGC 2-6 | 0.360/0.32, 0.332/0.37, **0.316/0.38**, 0.355/0.37 |
| T5 Countix-AV: same rows | 0.411/0.52, 0.361/0.56, **0.341/0.57**, 0.393/0.53 |
| T4: w/o TKA vs with TKA, regular split | 0.316/0.38 vs 0.268/0.47 |
| T4: w/o TKA vs with TKA, open-set re-split | 0.553/0.25 vs 0.415/0.33 |
| T6 fusion (RepCount and UCFRep): avg / attention / max | 0.285/0.45 and 0.182/0.72; **0.268/0.47 and 0.150/0.77**; 0.305/0.40 and 0.223/0.67 |
| T7 loss: count MAE only / lambda=1 / 10 / 20 | 0.327/0.37, 0.293/0.42, **0.268/0.47**, 0.288/0.45 |

- Note on T5: an extreme kernel-scale range (2-6) hurts. The text quotes a TKA re-split MAE gain of 0.108, but the table values give 0.138.
- **Countix**: Countix-AV subset with **manually annotated first cycles**, 0.330/0.58.
- **UCFRep**: in-domain 0.150/0.77; cross-dataset from RepCount 0.301/0.55 (T3).

### 2.5 HTRM-Net (Li et al., IEEE TMM)
- **Mechanism**: multi-scale features (windows of 1, 4 and 8 frames, 50% overlap) from frozen Video Swin-T, spatially max-pooled.
- Per scale:
  - **bi-modal TSM**: multi-head self-attention TSM plus a multi-head **dual-softmax** TSM (row softmax x column softmax, a soft mutual nearest neighbour), added together;
  - **random matrix dropping**: a whole TSM channel is zeroed with p = 0.3;
  - **local temporal context**: depthwise 1-D conv with window 2K+1 (K=2), then pointwise conv, concatenated as an extra TSM channel. It is motivated explicitly by pull-up up/down sub-actions that look alike but move in opposite directions, to remove **double counting**.
- Multi-scale 3-D conv fusion, then a transformer layer and an MLP produce the density.
- **Read-out**: sum of the density; loss is density MSE only. T = 64 frames uniformly sampled; lr 1e-4, batch 32.
- **Ablations** (RepCount test):

| Ablation | Result |
|---|---|
| T V: w/o local temporal context / w/o bi-modal TSM / w/o RMD / w/o multi-scale fusion / full | **0.4357/0.2715**, 0.3856/0.3576, 0.3723/0.3443, 0.4155/0.3377, 0.3543/0.3576 |
| T VI: SA+Dropout / SA+RMD / DS+Dropout / DS+RMD / SA+DS+Dropout / SA+DS+RMD | 0.4006/0.3046, 0.3905/0.3311, 0.3917/0.3112, 0.3858/0.3311, 0.3831/0.3509, 0.3543/0.3576 |
| T VII scales: V1 / V2 / V3 / V1,2 / V1,3 / all | 0.5803/0.1987, 0.4146/0.3113, 0.4327/0.2914, 0.4037/0.3178, 0.3914/0.3178, 0.3543/0.3576 |

- **Countix**: explicitly excluded (no fine-grained annotations).
- **UCFRep**: in-domain 0.3483/0.5143 (MFL-protocol table); cross-dataset from RepCount 0.5205/0.391; QUVA 0.5021/0.230.

### 2.6 MFL (Li and Xu, WACV 2024)
- **Mechanism**: a TransRAC-style RGB branch plus a **flow branch** (RAFT).
  - **Motion attention module**: flow features are the query, RGB features the key/value.
  - **TSM reconstruction loss**: frame features re-expressed through the row-normalised TSM (diagonal removed) must reconstruct themselves, with random masking. This strengthens correspondence between the same phase in different cycles.
  - **Variance-integrated loss weights**: the per-video variance sets how much each branch's loss counts.
- **Read-out**: the average of the RGB and flow densities, summed.
- **Encoder**: frozen Video Swin-T (K400); RX3D101 at 112 px for UCFRep. 64 frames, multi-scale as in TransRAC; batch 128, 200 epochs.
- **Ablations** (RepCount test):

| Ablation | Result |
|---|---|
| T4: baseline / +flow (naive) / +MAM / +reconstruction / +loss weights | 0.4736/0.2450, 0.4808/0.2715, **0.4159/0.3377**, 0.3909/0.3576, 0.3841/0.3860 |
| T5: reconstruction on no branch / flow / RGB / both | 0.4159/0.3377, 0.4030/0.3444, 0.3929/0.3444, 0.3909/0.3576 |

- In T4, naive flow does **not** help; flow helps only as a query into RGB.
- **Countix**: not used. **UCFRep**: in-domain 0.3879/0.5100; cross-dataset from RepCount 0.5227/0.3500.

### 2.7 Full-Resolution Repetition Counting (Li et al., 2023)
- **Mechanism**: offline frozen Video Swin features at **5 fps over the whole video** (no fixed 64-frame down-sampling), padded per batch. A dilated TCN (6 residual blocks, dilation doubling, skip connections) feeds a multi-head-attention TSM, then 2-D conv + FC, a transformer, and an FC density head.
- **Read-out**: sum of the density; L2 loss. lr 8e-6, batch 48, 200 epochs.
- **Ablations** (RepCount):

| Ablation | Result |
|---|---|
| T3 frame rate 1 / 2 / 3 / 4 / 5 / 6 fps | 0.4366/0.30, 0.6455/0.2667, 0.5231/0.3067, 0.4343/0.32, **0.4103/0.3267**, 0.4131/0.3133 (non-monotonic) |
| T4: no TCN / vanilla k1 / vanilla k3 / dilated k3 | 0.4683/0.2667, 0.578/0.1933, 0.4573/0.24, **0.4103/0.3267** |
| T5: TCN blocks | 6 is best; deeper is worse |

- **Countix**: not used. **UCFRep and QUVA**: cross-dataset only, 0.4608/0.3333 and 0.4952/0.25.

---

## 3. Cross-paper numbers vs TempoRAC

TempoRAC is joint-trained; every paper below trains per dataset, and several pick a different backbone per dataset. UCFRep is the 105-video val split everywhere.

| Method (encoder, adaptation) | RepCount-A test | UCFRep | Countix test | RepCount to UCFRep |
|---|---|---|---|---|
| **TempoRAC** (frozen ST-MAE ViT-B, joint) | **0.142/0.72** | 0.206/0.70 | 0.238/0.69 | n/a |
| CountLLM (VideoMAE + LoRA, Vicuna-7B + LoRA) | 0.162/0.639 | **0.116/0.839** | **0.215/0.801** | **0.139/0.816** |
| CountLLM (VSwinT) | 0.209/0.552 | 0.153/0.802 | 0.246/0.706 | n/a |
| ESCounts (frozen ST-MAE ViT-B) | 0.213/0.563 | 0.216/0.704 | 0.276/0.673 | 0.317/0.571 |
| ESCounts (frozen RX3D101 / R(2+1)D18) | n/a | 0.247/0.731 | 0.293/0.701 | n/a |
| D2-STX (+pose; SwinT / RX3D101 / R(2+1)D18) | 0.255/0.500 | 0.236/0.777 | 0.278/0.734 | 0.297/0.643 |
| Context / Sight and Sound (fine-tuned encoders) | n/a | 0.147/0.790 / 0.143/0.800 | S&S 0.307/0.511 | n/a |
| DeTRC (ViT) | 0.2622/0.5430 | n/a | n/a | 0.5435/0.4184 |
| FCA-RAC (first cycle given at test) | 0.268/0.47 | 0.150/0.77 | Countix-AV 0.330/0.58 | 0.301/0.55 |
| HTRM-Net | 0.3543/0.3576 | 0.3483/0.5143 | n/a | 0.5205/0.391 |
| MFL | 0.384/0.386 | 0.388/0.510 | n/a | 0.523/0.350 |
| Full-Res | 0.410/0.327 | n/a | n/a | 0.461/0.333 |

Readings:
- **RepCount-A**: TempoRAC already beats every paper, CountLLM included. An LLM is not needed for RepCount accuracy.
- **UCFRep and Countix**: TempoRAC's MAE is competitive, but its OBO trails the encoder-adapted methods (CountLLM, Context, Sight and Sound) and the per-dataset-backbone methods (D2-STX, ESCounts with RX3D101 or R(2+1)D18).
- With **the same frozen ST-MAE**, ESCounts sits exactly where TempoRAC is on UCFRep. Its OBO also improves when the backbone is swapped for supervised 3D CNNs (UCFRep 0.704 to 0.731; Countix 0.673 to 0.701).

---

## 4. Transfer to TempoRAC

### 4.1 Failure mode to mechanism
| TempoRAC failure mode | Mechanism | Evidence |
|---|---|---|
| Under-counting fast / high-count videos | (a) normalised count loss next to density MSE | ESCounts T3g (+0.06 MAE / +0.031 OBO); FCA-RAC T7; D2-STX Eq. 14 |
| | (b) keep dense temporal tokens and never pool below ~2 tokens per cycle | ESCounts T8; Full-Res T3; DeTRC short-period OBO |
| | (c) tempo-normalised second pass | FCA-RAC Sec. 3.1, T5 |
| | (d) clip-level counting for the LLM path | CountLLM Sec. 3.3 |
| Half / double-period confusions on symmetric motions | (a) exemplar cross-attention: one annotated cycle defines the unit | ESCounts T3a-d, T4 |
| | (b) local / directional context: depthwise temporal conv; feature differences or flow as query | HTRM T V; MFL T4 |
| | (c) structure cue as query (pose or joints) | D2-STX T3/T4 |
| | (d) mutual-nearest-neighbour (dual-softmax) TSM channel | HTRM T VI |
| UCFRep generalisation | (a) adapt the encoder: LoRA or partial fine-tuning of ST-MAE | CountLLM T7; ESCounts T1c and Sec. 4.2 |
| | (b) a complementary supervised-action stream | ESCounts T1b/c backbone rows |
| | (c) test-time self-exemplars (unverified variant of ESCounts T4) | ESCounts T4 (0.216/0.704 to 0.198/0.723 with true exemplars) |
| | (d) LLM route: strongest cross-dataset number, but tied to (a) | CountLLM T3 |

### 4.2 Ranked recommendations (cheap and well-evidenced first)
1. **Normalised count loss.** L = density MSE + lambda |sum d - c| / (c + epsilon), with lambda in {1, 10} and epsilon = 0.1. Apply it to every view, including the 2x pyramid view.
   - Report MAE/OBO by count quintile and by repetition-length quintile, ESCounts Fig. 6 style, to show the effect on high counts.
   - Countix: lean on the count term and down-weight density MSE on uniform pseudo-labels. FCA-RAC trains density only where it is annotated. This specific weighting is untested.
2. **Encoder adaptation before any LLM work.** LoRA r=16 on ST-MAE attention, or unfreeze the last 2-4 blocks, keeping the current head.
   - CountLLM's biggest factor is LoRA-V.
   - Doing this first de-confounds the LLM experiment: otherwise any gain from the LLM stage may simply be visual adaptation.
   - Watch the compute of three strides; adapting only the stride-4 path is a cheaper first test.
3. **Training-time exemplar cross-attention with a learned zero-shot latent**, ESCounts settings: |S| from {0,1,2}, p = 0.4, exemplars of 16 frames.
   - Cycle boundaries exist for RepCount-A and UCFRep; Countix needs uniform pseudo-cycles.
   - Diagnose with a histogram of pred/GT ratios around 0.5 and 2.0.
4. **Direction-sensitive local context.** Add a depthwise 1-D conv branch (K=2) and first-difference features (x_t+delta - x_t) as a cheap flow proxy, used as **queries** into appearance tokens.
   - MFL and D2-STX both show that the auxiliary modality helps only when used as the query (MFL: naive flow is worse than baseline).
5. **Tempo-adaptive second pass**, generalising the fixed 2x pyramid view: estimate the period from pass-1 density peaks, choose the stride or resample rate so that one cycle spans about 4-8 tokens, then re-predict.
   - Keep the tempo range moderate: FCA-RAC T5 shows a 2-6 scale range losing to 3-5.
6. **Spatial tokens.** If the temporal encoder sees spatially pooled ST-MAE tokens, test ESCounts-style local (4,7,7) window attention over spatial tokens. ESCounts T7 shows local windows beat full attention, and its decoder never pools space before the density head.
7. Minor: time-shift TTA (ESCounts T3f, +0.008 MAE). TempoRAC's pyramid test view probably already covers part of this.

### 4.3 What the planned Qwen3-VL-8B stage needs (from CountLLM, plus checks against the plan)

The plan: encoder tokens pooled per 16 frames, an MLP projector, LoRA r=16, a per-video description, a 4-digit CE answer, and LLM hidden states feeding the density head.

1. **Token rate (critical).** One token per 16 frames gives fewer than 2 tokens per cycle for periods under ~32 frames (about 1.07 s at 30 fps). ESCounts' fastest equal-size bin (<0.96 s average repetition) is about 20% of RepCount test.
   - Use at least one token per 4-8 frames, or a query compressor per clip (CountLLM: 64 queries per 32-frame clip).
   - The density head on LLM hidden states would also be limited to that rate. Keep a residual path from the full-rate pre-LLM tokens into the density head.
2. **Compressor + alignment.** A projector with no periodicity compressor and no alignment is CountLLM's worst row (0.498/0.260).
   - Our count-trained temporal encoder probably plays the PT + P-A role. **Keep the density loss active during stage 1 and stage 2** so the tokens do not drift toward caption semantics.
   - Our stage 1 (projector reproduces a description) corresponds to CountLLM's weak S-A stage, not its strong P-A stage. Add a periodicity-discriminative auxiliary: repetitive vs non-repetitive clips, or a tempo-relation task using our exact 2x pairs. The second is our idea and is untested.
3. **Description.** CountLLM's evidence covers **one constant, class-agnostic description** only.
   - Per-video descriptions, including "what one repetition is", are new. They may genuinely help half/double ambiguity but are unverified.
   - Requirements:
     - (i) generate them identically for train and test, from the video only, never from GT labels or class names;
     - (ii) strip numerals and count words to avoid leakage;
     - (iii) run the ablation constant text vs learned soft prompt of equal length vs per-video description, to separate "text semantics" from "fixed prefix".
4. **Answer format.**
   - Fixed-width digit strings with existing tokens (Table 5). Qwen tokenisers split numbers into single digits; confirm this for Qwen3-VL.
   - Digit-level CE is **not ordinal**. Read the final count from the density head, or blend with an expected value over digit distributions; do not rely on argmax text.
   - For long videos, use clip-level answers with start/end incomplete-cycle flags. RepCount counts reach 141 and CountLLM never answers whole long videos in one shot.
5. **Visual-side LoRA.** If ST-MAE stays frozen, CountLLM's own Table 7 predicts a limited LLM benefit: frozen V + LoRA-L is 0.317/0.498 against 0.166/0.631 for the full model. The FiLM failure with a frozen Qwen state is consistent with frozen V+L (0.409/0.357).
6. **Hidden states into the density head** is not in CountLLM; there is no evidence either way. Ablate: density from pre-LLM tokens only, vs plus LLM states, vs LLM-only.
7. **Protocol.** CountLLM ablates on RepCount val and trains per dataset. Compare TempoRAC against it on test with seeds, keep joint training explicit, and note the baseline inconsistencies listed in 1.7.

### 4.4 Main pitfalls
- Expect LLM gains to concentrate on UCFRep and Countix (short videos, low counts, diverse classes), not RepCount. CountLLM's RepCount OBO (0.639) is below TempoRAC's (0.72).
- CountLLM's headline numbers depend on details it does not report (test clip stride and length, Countix clip labels, LR and batch size). Do not treat 0.116/0.839 on UCFRep as a reachable target until encoder adaptation is in place.
- DeTRC-style query counting caps the count at Q per window, and FCA-RAC needs a test-time first cycle. Neither is directly adoptable in its published form.
