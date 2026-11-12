# TempoRAC current-framework scientific audit — 2026-11-11

**Decision: continue the frozen no-LLM framework; do not restart the retired MultiRep/tempogram route. Scientific submission readiness is not established.** The current results are worth preserving. The immediate work is evidence reconciliation, accurate theory and protocol wording, matched statistical analysis, and a small final-framework analysis package—not another architecture search driven by the same test sets.

**Immediate remedy freeze:** make manuscript/provenance corrections and server-side analysis of already saved predictions only. Do not change trained weights, cached features, targets, test-view rule or reported numerical pipeline. The parent reports that the actual AutoDL server has now been reached and final results are being retrieved; matched-seed CI analysis is in progress. Any additional sensitivity runs described below are conditional future gap-filling, not an instruction to launch during this correction pass. Independent confirmation and the six-domain SOTA goal remain unfulfilled.

Reviewer: `/root/current_science_astra`; requested `gpt-6-astra`, actual backend model/effort not independently attested. This is a fresh same-family scientific review, **provisional**, with concrete negative findings. Scope: static source and manuscript review, version-history and local result-summary review, bounded primary-source web checking. No local experiment, model execution, evaluation, feature construction, dependency installation or server job was performed. Server raw-result reconciliation is owned by the parent and remains necessary.

## Acceptance contract and observed status

The following distinguishes completed work from quality acceptance. These checks were derived from the supplied user instructions before forming the final verdict; this file consolidates the review and does not claim a historical preregistration.

| Requirement | Status | Evidence / unresolved obligation |
|---|---|---|
| Resume latest authorized framework, preserve history | Pass for this review | Local HEAD `489d3ab8fb1b83ffd5f247247bf4d4effe22441c`; execution log sections 3n–3p supersede Nov 4/5 running-state text. No repository edits or old-route restart. |
| Complete qualified data and documented exclusions | Partial | Declared full RepCount-A, UCFRep and retrieved Countix manifests; exact Countix/source overlap ledgers not in the reviewed manuscript. Four evaluated datasets, not six. |
| At least six datasets × seven convincing external frameworks, six-domain SOTA | Not met | Three in-domain datasets plus QUVA zero-shot; literature table on RepCount-A; one reproduced external checkpoint. UCFRep/Countix are below stronger published values; QUVA compared only with ESCounts. Latest main-text selection does not establish this broader requirement. |
| Correctly scoped whole-method external comparisons | Partial | Several literature rows, modal labels and PAMS retained; differing populations, normalization and training resources prevent an unqualified common-protocol SOTA conclusion. |
| At least three coupled components with actual contribution | Partial | Multi-rate representation, shifted view, and token/balanced joint training have plausible coupling and positive RepCount-A mean ablations. Matched-seed uncertainty and the weak second-depth effect are unresolved. |
| Fewer than three loss terms | Pass in inspected frozen configuration | `head=concat`, `w_count=0`, no active LLM/period branch; one density-MSE objective (scaled by 100 in code). |
| Observation motivation figure/table | Partial | Useful validation statistics and knockout prose exist; current manuscript has architecture drawing, no actual observation plot/table. |
| Hyperparameter curves and stability | Not met in current manuscript | No final-framework curves; old generator targets retired tempogram parameters and an incompatible single-dataset feature root. |
| Valid selection and independence disclosures | Not met in manuscript | Per-checkpoint selection uses validation; method and main benchmark were chosen after test inspection. Manuscript claims no test selection. |
| Correct mechanism/formula and source alignment | Not met | Exact finite-video shift theorem has a padding counterexample; grid and loss/target statements differ from implementation. |
| Complete metrics, uncertainty, efficiency, reproduction evidence | Partial | Mean/SD for MAE/OBO, some time/memory claims. Missing same-seed paired uncertainty, fuller metric results, auditable inference-cost coverage and complete reproduction package. |
| Current CVPR rules, final claim/citation audits and all-page render | Not established by this review | Existing draft/build log is not a final assurance receipt. Parent handles current venue and visual audit. |
| Server-only computation, no timer, preserve other projects | Pass for this review | Static reads/writes only; no timer or runtime action. |

## What actually exists now

The newer code lives in `temporac_cvpr/`, and the active manuscript is `paper/cvpr27/`, not the old `paper/sections/` draft. The Nov 8–11 log records a user-authorized move from the failed MultiRep/geometry target and local-tempogram route to supervised community counting benchmarks, then multiple changes informed by validation and later tests. The current choice is the **Nov 11 frozen framework**:

1. Frozen ST-MAE ViT-B/16 features at strides 2, 4, 8, and auxiliary stride 16 for shifted views. Blocks 8 and 12 supply concatenated channels; spatial pooling produces 4×4 features.
2. Separate spatial attention pooling per rate, repetition onto a 2-frame output grid, concatenation/linear fusion, three temporal blocks (depthwise convolution, attention, MLP), one positive density head.
3. Shift augmentation at probability 0.5 subject to a training-label period threshold, plus the mean of original/shifted counts at inference.
4. Joint RepCount-A/UCFRep/Countix training, dataset embedding added to every fused cell, RepCount-A/UCFRep repeated four times, 50 epochs. Each target has its own validation-selected checkpoint from the shared training trajectory.
5. One active loss: density MSE. No LLM, tempogram routing, period auxiliary task or encoder adaptation in the frozen final method.

This is not one identical final checkpoint for every benchmark. It is one shared training trajectory per seed yielding target-specific checkpoints, a distinction that is already partly disclosed and must stay explicit.

`final_tables_frozen.py` combines `final_v2/wo_llm_s0..2` and `frozen_a/full_s3..4`. `make_frozen_specs.py` correctly removes the old MLLM configuration before constructing current ablations; reusing the old LLM-equipped ablations would have been invalid, and this was corrected. The five-seed summary gives:

| Dataset | MAE mean ± sample SD | OBO mean ± sample SD | Seeds |
|---|---:|---:|---:|
| RepCount-A | 0.143115 ± 0.007599 | 0.711258 ± 0.063968 | 5 |
| UCFRep | 0.194398 ± 0.017183 | 0.716190 ± 0.029662 | 5 |
| Countix | 0.235985 ± 0.004407 | 0.705035 ± 0.003897 | 5 |
| QUVA zero-shot | 0.182591 ± 0.011033 | 0.562000 ± 0.041473 | 5 |

These are read from `docs/final_numbers_frozen_20261112.json`, not recomputed from predictions here. The filename is a date label, not proof of a future result: the execution log says completion on Nov 11 at 20:37. Source/config/prediction identity must be checked on the actual AutoDL root recorded by that log, rather than assumed from the older `/mnt/4tssd/lilei/TempoRAC` setup.

## Findings that should be fixed before any stronger claim

### 1. Method selection used test outcomes; the current independence sentence is false

`docs/EXECUTION_LOG_20261108_CLAUDE.md` section 3o explicitly records selection of `final_v2/wo_llm` after test inspection and user approval. Sections 3j–3n show previous test results, LLM redesign and encoder choices. The appendix itself says encoder adaptation raised test error and therefore the encoder is kept frozen. In contrast, `paper/cvpr27/sec/4_experiments.tex:41` says no test video is used for training or selection.

**Required distinction:** checkpoint selection used validation; final method choice and main benchmark emphasis were retrospective. Each checkpoint being scored once does not make the final method an independently selected test evaluation. New random training seeds on the same exposed test videos do not restore independent dataset-level confirmation.

Minimal correction: explicitly disclose the development/test-selection sequence in the main protocol and appendix, preserve all compared variants, and describe the displayed benchmark result as post-selection. Do not silently erase tests or restart a new method search. If the intended final claim specifically requires independent confirmation, freeze this exact method and obtain a genuinely unexposed, source-disjoint evaluation population; do not relabel existing validation/test samples. That is a separate bounded evidence addition, not a reason to discard current experiments.

### 2. The exact finite-video pyramid-shift theorem fails under current endpoint padding

`extract_features.py:138` forms each sampled frame as `min(j + s*f, T-1)`. `train.py:52–66` reuses the original stride-2s stream and calls it the exact encoding of `V[0::2]`.

For even T, actual `V[0::2]` ends at `V[T-2]`; its encoder's repeated last frame is `V[T-2]`. The reused original stream instead pads with `V[T-1]`. A concrete symbolic example is T=32 and s=2: the original stride-4 clip samples 0,4,...,28 and then 31, whereas a stride-2 clip of the 16-frame decimation samples original indices 0,4,...,28 and then 30. Thus the input clips differ. Because the ViT mixes tokens within a clip, this need not affect only the last output token.

The equality is valid for unpadded corresponding clips. It can also be stated for a specifically defined source-first endpoint extension, but that differs from ordinary finite-video decimation followed by its own padding. The scientific claim must state this boundary convention, and `exact/free` must not imply literal equality for every independently decoded finite video.

**Minimal action:** keep the already evaluated implementation; revise the theorem to exact correspondence on supported clips with explicit endpoint handling, or formally define the source-extended operator that the cache actually realizes. A server-only index audit can count affected videos/clips; a model re-encoding study is not necessary merely to prove the symbolic mismatch. Do not modify caches and silently reuse old scores as if they belonged to the changed implementation.

### 3. Several method equations describe a different computation

* Grid: `models.py:32–41` repeats rate-s tokens **s times** to a 2-frame base grid. The paper instead repeats **s/2 times** to the finest rate-2 feature grid and then says the output is on 2-frame cells, without a conversion. Write the 2-frame base grid consistently; sinusoidal indices are base-cell indices, not raw-frame indices.
* Overlap: the first two rate-specific cells are supported by one clip, the next two by two, then three, before interior cells reach four. `every cell is seen by four overlapping clips` is false; the code correctly divides by the actual support count.
* Loss: `train.py:195–198` implements `100 * sum_{i,t} mask_it * (d_it-y_it)^2 / sum_{i,t} mask_it` for the frozen configuration. This is a masked batch-cell average, not equal-weight per-video averages. The factor 100 and batch-cell weighting should be stated. It remains one loss term.
* Target mass: `data.py:91–107` normalizes each Gaussian on an extended grid, then truncates to the valid video/crop. Gaussian tails beyond original video boundaries are discarded too. Therefore `sum y = c` is not exact even for every uncropped video. Describe retained partial mass and avoid a false mass-conservation identity. Preserve targets used by the published runs.
* Tempo coverage: a stride-8, 16-frame clip spans roughly 4.3 seconds at 30 fps, while the stated slow end of median cycles is 4.75 seconds. The sentence saying the coarse level sees two to four **slow** cycles in one clip is not supported by these numbers. Describe larger context without the numerical claim.

### 4. Mean-direction ablations are promising; all components are not yet statistically established

The RepCount-A full mean is lower than every removal's MAE mean. That observation is real in the summary. But full has five seeds and most removals have three. Existing seeds 0–2 must be compared to the same full seeds, with video/source-paired errors and seed variation retained. Report full five-seed results separately.

Second depth is the weakest example: full MAE 0.143115, removal 0.146194 ± 0.014293; removal has **higher OBO** (0.728477 vs 0.711258). This is not evidence of a consistent contribution across metrics. On UCFRep, removing shift yields MAE 0.192012 vs full 0.194398 and OBO 0.777778 vs 0.716190. The OBO difference is material enough to report plainly, rather than reduce the finding to `marginally better` MAE or `tied`.

There is already useful separation of training and inference: original-view predictions of full shift-trained models, and original-view predictions of no-shift-trained models. `wo_shift` has `shift_eval=true`, so its shifted/TTA predictions may also already exist; include the full 2×2 train-shift × test-view table from saved predictions, using the matched seeds. No new model training is needed for this factorial comparison if outputs are present.

Rate knockout at inference is a sensitivity diagnostic, not removal-and-retraining evidence and not identification of learned routing. Zeroing a stream can be off-distribution. Change `the decoder assigns slow repetitions to coarse and fast ones to fine` to the supported statement: errors on slow/fast strata are more sensitive to removal of different streams. The actual model fuses all streams with no explicit temporal routing variable.

### 5. External ranking requires explicit metric/population/resource scope

Current scoring excludes zero-count videos, rounds predictions and divides by g+0.1. Thus the main RepCount-A MAE population is **151 positive-count videos**, despite the table comment describing 152. CountLLM's primary paper Eq. 13 instead writes division by g; its source confirms 0.162/0.639, 0.116/0.839 and 0.215/0.801, but does not resolve all population and sampling details. [CountLLM primary paper](https://arxiv.org/html/2503.17690v2).

This denominator difference alone does not explain away the lead: for positive integer g and identical rounded numerator, the /g error is at most 1.1 times the /(g+0.1) error; 1.1 × 0.143115 is 0.157427, still below 0.162. This is a bound under those assumptions, not a substitute for protocol reconciliation or statistical significance. Existing own predictions should be scored under both definitions on the server, and zero-count cases reported separately.

The proposed method is jointly trained on three RAC datasets; ESCounts comparison is an external whole-method comparison with different training setup. Sharing the pretrained ST-MAE encoder does not mean sharing **the same frozen tokens**: rates, intermediate depth, spatial pooling and training sets differ. Say `same pretrained encoder family/weights` after verifying identity, and attribute gains to the complete changed pipeline. Do not present the literature gap as a causal ablation.

Preserve PAMS and the unknown-protocol caveat instead of hiding it. `lowest reported MAE among the listed RGB-based methods under their stated protocols` is the strongest presently defensible ranking phrasing after own result provenance is confirmed. Unqualified SOTA over all modalities, all datasets, or an independently selected common protocol is unsupported.

The main table has nine distinct non-pose external methods plus a second ESCounts checkpoint row, not ten independent frameworks. A repeated row is not an additional framework, and not every cited venue is a top main conference. At least six independent datasets with complete seven-framework comparisons is absent. Moving negative domains to the appendix preserves results but does not satisfy that requirement.

### 6. Current final-analysis and reproduction material is incomplete

The manuscript has no hyperparameter figure, no complete observation figure/table, and only MAE/OBO despite older plans for OBZ/RMSE and broader error summaries. Countix exclusions are promised in the supplement but no ledger appears in the inspected appendix. `prepare_countix.py` saves at most the first 30 identifiers per excluded category in its report, so the complete exclusion ledger may need reconstruction from the original match/download records, not invention.

`make_union.py` checks feature-key collisions, not shared underlying source video across train/validation/test or across datasets. Existing source/video provenance must be reviewed before asserting all evaluation data are independent of the training union. I found no final-union overlap evidence in the reviewed local artifacts; this is a missing-evidence finding, not an assertion that leakage occurred.

`make_hparam_specs.py` is stale: it sweeps `max_lag`/`tempo_win`, which are inactive in frozen ConcatHead, and hardcodes single-RepCount/old feature paths. Do not run it as the final-framework study. `efficiency.py` likewise assumes 768-dimensional one-depth input, omits the dataset-token input, counts only base rates instead of the auxiliary shifted level, and measures one view. It is not a valid final-method efficiency receipt as written.

The closest multi-stride prior already combines counts from several strides; this is confirmed by its primary publisher abstract, while the full chapter was unavailable. The current token-before-decoding distinction is plausible but does not establish broad novelty of multi-rate processing or the elementary resampling identity. [Gan et al., KSEM 2023](https://link.springer.com/chapter/10.1007/978-3-031-40289-0_24). The Nov 9 novelty document predates the final no-LLM/shift/joint formulation and cannot by itself accept current novelty.

## Bounded next actions, in order

1. **Reconcile existing raw artifacts on the actual server.** Read full config/results/predictions for the five main seeds and each frozen ablation; match checkpoint, manifests, encoder and feature roots. Confirm positive-count n and assert complete expected seed/dataset cells. The table generator silently skips missing files; summary existence alone is insufficient.
2. **Close statistical and selection disclosures without new training.** Generate matched-seed 0–2 comparisons; paired uncertainty over actual source/video units and a separately reported seed dimension; identify the multiple ablation family. Export the shift train/test factorial, both metric definitions and zero-count results. Preserve five-seed headline results and every negative variant. Document retrospective method selection.
3. **Fix manuscript source alignment.** Use the exact implemented grid, actual overlap counts, batch-cell loss/scale and boundary-truncated targets. Bound the shift theorem and describe the nonzero extra coarse-encoding and second-decoder costs. These are documentation fixes to the frozen method, not authorized data/model changes.
4. **Use existing final-framework validation outputs for observation/mechanism figures.** Show natural cycle-duration variation, rate-removal sensitivity by tempo and signed view errors, with n and uncertainty. Explicitly separate diagnostics from retrained ablations. Do not present abandoned-route statistics as measurements of the final model.
5. **Fill final-framework sensitivity only if absent.** A narrow validation-only package can reuse p_shift=0 and 0.5 existing runs and add p_shift=1 (three full-data seeds), plus sigma=0.125/0.25/0.5 with 0.25 reused (six new runs). This is at most nine full-data runs before any broader search, approximately 15.3 GPU-hours if the logged 1.7 hours/run transfers; the actual budget and runtime must be frozen on the live server first. Curves characterize the frozen method; they do not choose a new test winner. If parent discovers existing compatible runs, reuse them instead.
6. **Reconcile external protocols and final packaging.** Retain own ESCounts common-pipeline evidence and literature values with provenance/resource labels. Obtain fresh final claims, citations and current-novelty reviews after corrections. Inspect final all-page render including appendix and check official current-year rules. Do not describe this audit as that final assurance pass.

If a truly independent confirmation set remains a user-required endpoint, identify a legal unexposed source-disjoint population and freeze the full method/metric before evaluation. If six-domain SOTA remains the endpoint, it is still materially unachieved and cannot be closed by this limited repair package. Keep that status explicit; do not reactivate discarded routes or expand to new data silently.

## Minimum claims the evidence currently supports

* A fully specified, frozen-encoder multi-rate counting method with one training loss and a useful cache-reuse view transform.
* Reported five-seed results on three community benchmarks and zero-shot QUVA, subject to raw-server reconciliation and explicit post-selection status.
* Lower RepCount-A MAE means for the full method than the tested component removals; strength/consistency of the effects awaits matched uncertainty.
* Several LLM and encoder-adaptation variations did not improve all test datasets consistently; preserve the trials and their selection history.

It does not currently support universal or independent SOTA, six-dataset coverage, statistical proof that every component helps, a globally exact standard finite-video resampling theorem, or finalized submission-quality assurance. The right continuation is a focused completion of evidence and manuscript fidelity around the latest frozen method.
