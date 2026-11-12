# Runnable contract: known-source-disjoint training correction

**Decision: proceed immediately with three complete decoder training runs.** This is a necessary correction for verified training/evaluation overlap, using the frozen current framework and all remaining qualified training rows. It is not a restart of the research direction, a cache rebuild, a hyperparameter search, or a certification that upstream evaluation annotations are pristine.

Date: 2026-11-11. Scientific designer: `/root/current_science_astra`, requested `gpt-6-astra`, actual backend/effort unattested, same-family provisional. The pinned `experiment-plan/SKILL.md` and its required output-versioning, manifest and language references were read in full; source/installed skill SHA-256 matches `61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a`. User full-data and existing-project constraints override the upstream toy-overfit and default baseline-count suggestions.

## Acceptance table for this stage

| Requirement | Frozen acceptance evidence |
|---|---|
| Remove verified optimization exposure without hiding test rows | Training-only exclusions from a declared source-matching rule, four expected rows; every evaluation row and original label retained byte-for-byte as record values. |
| Preserve the current framework | Identical merged model/training configuration to the verified `final_v2/wo_llm_s0` center, except new manifest/output paths and seed; fresh decoder initialization. |
| Reuse valid expensive work | Existing original ST-MAE weights and block-8/block-12 caches at rates 2/4/8/16, unchanged. No adapted encoder, LLM or feature re-extraction. |
| Complete formal training | Seeds 0,1,2; 50 full epochs over the corrected balanced union; finite completed logs and saved validation-selected checkpoints. |
| No new test-driven selection | Configuration, train exclusions, seed set, inference rule and comparisons fixed before new scores; original-view validation checkpoint selection only. |
| Honest evaluation scope | One scoring pass per saved checkpoint on every original evaluation row; label the result as source-disjoint **training**, inherited evaluation media/labels and post-selection method. |
| Fair component evidence | Later comparisons use the same corrected manifest and matched seeds 0,1,2; old contaminated ablations remain historical. |
| External/SOTA goals | Still unfulfilled until qualified comparisons and further evidence establish them; this stage does not declare success from training completion. |

## Verified rationale

`server/suspect_lance_audit.json` confirms that the actual four RepCount files match their Lance source blobs and stored hashes. `suspect_feature_comparison.stdout.txt` confirms all 16 paired block/rate cache comparisons are identical across the two train/test pairs. The initial concern is therefore no longer merely a stale hash field. The local materializer copied these tested blobs faithfully. The conflicting source annotations remain unresolved; agreement with another CSV copy does not adjudicate them.

The parent additionally reports the second pair's same SHA in three independently archived Nov 4 source trees, while the first pair has visual action/annotation disagreement. This strengthens the need to separate source-exposure correction from attribution of upstream dataset errors. An unavailable old source path is not a reason to leave otherwise useful GPUs idle.

## Exact dataset change

Parent manifest: `/root/autodl-tmp/TempoRAC/data/UNION/manifest_dev.json`, previously recorded SHA-256 `bd171452197b74435e6fecfd19332f3e0e7c428e48b1f3a8aeb2b77c35894a13`; executor must confirm identity before deriving the new file.

New namespace: `/root/autodl-tmp/TempoRAC/data/UNION_known_source_v1/`.

Uniform rule: exclude a row from **optimization only** when its verified exact source bytes or recorded underlying source ID appears in any existing validation/test role. For Countix, block by YouTube source ID, not cropped filename. For UCFRep, retain the already frozen action/group split. Apply the rule to the complete recorded population; do not inspect errors or choose removals according to model performance. The present evidence predicts exactly these four exclusions:

| Optimization row | Matching held-out role |
|---|---|
| `rc_train__stu4_50` | `rc_test__stu4_5` |
| `rc_train__stu3_27` | `rc_test__stu4_27` |
| `cx_train__t6Ny1kl7Mxs_2.16667_9.66667` | `cx_val__t6Ny1kl7Mxs_2.13333_9.50000` |
| `cx_train__s5stimPX3h0_0.00000_2.56000` | `cx_val__s5stimPX3h0_0.00000_2.76000` |

Expected training counts: RepCount-A **756**, Countix **4222**, UCFRep **331**, total **5309 unique videos**. With RepCount-A/UCFRep repeated four times, one epoch has **8570 sampled entries**. Retain validation counts RC131/CX1374/UCF90 and test counts RC152/CX2403/UCF105 unchanged; QUVA100 remains external to optimization. Positive-count evaluation filters are metric-specific and do not delete records.

Write `manifest_dev.json`, `excluded_train.jsonl` including complete original rows, matched held-out IDs and reasons, and a compact derivation receipt with before/after counts, source/hash identities and invariant checks. Do not delete or move media/cache files. If the uniform rule discovers additional groups, record the evidence and revise the derived-manifest contract before launch; do not silently ignore them or improvise error-based filtering. This is a recorded-known-source guarantee, not proof against every unknown near-duplicate or pretraining overlap.

## First three runs — launch now

Group/output: `runs/frozen_known_source_v1/full_s0`, `full_s1`, `full_s2`.

* Use fresh decoder random initialization at seeds **0,1,2**. Do not load a previous counting decoder or optimizer state; those weights depended on exposed training examples.
* Clone the verified merged configuration of `final_v2/wo_llm_s0`: `head=concat`, rates `{2,4,8}`, block8 plus block12, spatial pool4, width256, three temporal blocks, original frozen ST-MAE, one density-MSE loss scaled100, `w_count=0`, no active LLM/period/tempogram mechanism, AdamW lr2e-4/wd0.05, existing warmup/cosine/batch8/crop policy, `shift_prob=.5`, `shift_eval=true`, original shift period threshold, dataset token and repetition factors, **50 epochs**.
* Preserve feature roots `features/UNION_stmae_b8_p4` and auxiliary `features/UNION_stmae_p4`; their files and sample keys remain valid after training-only exclusions. Preserve the original finite-padding and cropped-cache conventions already documented.
* Save the complete merged configuration and source receipt. Compare configuration hashes after canonicalizing only the declared path/seed differences. Do not use stale pre-frozen configs, a new learning rate, a different loss, or the old adapted encoder.
* Training selects one checkpoint per benchmark on the corresponding original-view validation MAE, with the existing OBO tie-break; no test use in checkpoint selection. Retain complete epoch logs, checkpoint IDs and per-video predictions.
* Prefer training without an automatic end-of-run test step, then invoke the already verified saved-checkpoint evaluator once after the frozen run specification is recorded. This is execution separation only, not a change to the model or scoring rule. It does not require waiting for all literature retrieval.

Preflight consists only of server CPU checks for the exact row set/counts, feature existence/key mapping, unchanged feature dimensions, configuration differences, valid launcher/output paths and free resource ownership. Do not launch a toy GPU run, overfit subset or short-epoch substitute. If these checks pass, execute the formal jobs immediately.

## Resource and timing budget

Two currently idle RTX4090D GPUs; at most **two concurrent training processes total**, one per free GPU. Queue the third seed behind the first completion. Recheck ownership immediately before launch and preserve any other project's healthy tasks. The previous full decoder runtime is about1.7 hours/run, giving **5.1 GPU-hours** nominal for the three seeds; plan approximately **5–8 GPU-hours** and **3.4–5 hours wall time** at two-way concurrency. These are estimates, not permission to truncate a formal run when its wall clock differs.

All files, caches, logs and temporary work stay under the verified server root or E: for compact local receipts. No C:/D: downloads, environments, temporary model files or local computation. Keep existing hidden/background launch requirements. Do not create a timer or periodic automation.

Engineering completion means all50 epochs completed, finite losses/outputs, no unexplained skipped nonfinite steps, saved checkpoints and exact population agreement. A completed job is not a SOTA or component-contribution pass.

## Evaluation: one original full-population pass

Evaluate each selected checkpoint on the original corresponding test split using the fixed mean of original and shifted count predictions. Use the original RepCount-selected checkpoint/token for QUVA. Preserve all original rows, including RC `stu4_5`, `stu4_27` and zero-count `train143`.

Frozen primary score: NumPy rounded prediction, positive-count normalized MAE `abs(round(pred)-g)/(g+.1)`, lower is better. Report OBO, OBZ, RMSE, `/g` normalized MAE for positive counts, full-population scores, and the zero-count result as secondary metrics. Do not switch the primary metric to whichever wins.

Call these **new results from known-source-disjoint training on inherited evaluation media/labels**. They do not remove the known test annotation ambiguity or the historical test-informed method choice. Keep original five-seed results and all29 old runs under the old protocol; do not pool seeds from different training manifests.

The current historical matched-three-seed RC mean is `0.13824355882720726`, not a stale number from an earlier intermediate analysis. It must not be used as the baseline for new-protocol ablations, which will use the new three-seed full runs.

## Frozen follow-on matrix for complete component evidence

Freeze the matrix now; do not choose its arms from the new test outcomes. After the first full formal run verifies the launcher/data path, continue on the same bounded queue without another conceptual-review loop:

| Arm | Seeds | Training | Purpose |
|---|---|---|---|
| Full | 0,1,2 | corrected union,50 epochs | main corrected method |
| Without pyramid, s=4 only | 0,1,2 | same corrected union | multi-rate contribution |
| Without shift augmentation | 0,1,2 | same corrected union | shift contribution; save original and shifted predictions for fixed 2×2 train/test-view analysis |
| Without second depth | 0,1,2 | same corrected union | depth contribution |
| Without dataset token | 0,1,2 | same corrected union | token contribution |
| Without balanced repetition | 0,1,2 | corrected union, original80-epoch ablation schedule | sampling contribution with approximately matched updates |
| Single-dataset RC | 0,1,2 | corrected RC756, original80 epochs | joint-training removal |
| Single-dataset CX | 0,1,2 | corrected CX4222, original80 epochs | joint-training removal |
| Single-dataset UCF | 0,1,2 | unchanged UCF331, original80 epochs | joint-training removal |

There are **27 total runs including the first3 full runs**. Copy each arm's validated old frozen specification and apply only the derived manifest/output/seed changes. The initial3 are the immediately runnable minimum; the remaining24 supply the user's full matched component requirement. Estimate up to about46 GPU-hours if every run cost1.7 hours; single-dataset/smaller arms should cost less. Save measured runtimes and revise forecasts from complete jobs; never shorten training to meet the estimate. Reuse any genuinely identical clean single-UCF run if its complete effective training population/configuration/source is proven identical; do not rerun solely for a new directory name.

Analyze matched seeds0–2 only. Average each video's error over seeds before paired source-cluster resampling, report seed SD separately, and disclose that this is conditional on fitted models. Keep the eighteen MAE contrasts as the declared family. No claim that every component helps is made until these results support it; absence of significance is not equivalence.

## Parallel retrieval and external comparison, without stalling training

1. Reuse the original-source inventory already performed; do not retry the eight missing old-server paths. Read the local server checkout of the **official TransRAC README/data instructions** and original release metadata to locate an independent archive. Retrieve the four named RepCount assets or exact archive entries where feasible. Compare media and original annotations; no author/contact message without explicit user authorization. If an independent source provides corrected evaluation media, keep a new evaluation-input version and score all methods on the same complete corrected population; never overwrite the inherited test version.
2. Build a concise availability/protocol matrix for the seven whole-method baselines already central to the project: CountLLM, ESCounts, TransRAC, RepNet, Context, Sight&Sound and MFL. Preserve stronger applicable PAMS results and modality distinctions. Existing ESCounts weights can provide a same-evaluation-input reference; their historical training data are not magically changed by our new split. A strict clean-training comparison needs an official baseline trained on the declared clean training protocol, labelled separately from released checkpoint evaluation.
3. Prefer retrieving official released weights/configs and evaluating exact common inputs over re-running a large unrelated pretraining stage. If a baseline is not reproducible within the verified assets/budget, preserve its literature result with an explicit protocol label; do not replace it by an internal 'style' head and call that an official reproduction.
4. For the still-unmet six-dataset goal, perform one bounded official-source availability pass for YTSegments and ActCount (which accompany published counting work), alongside the four existing datasets. Choose additional evaluation populations by task relevance and legal complete availability **before** scoring, not by anticipated favorable results. MultiRep derivatives or speed perturbations are not substitute independent datasets. No dataset or source is claimed available by this contract.

## Claims and stopping conditions

Primary scientific claim under test: the frozen multi-rate representation and reused shifted view improve counting on the declared training/evaluation protocol. Supporting claim: token plus balanced joint training lets one training trajectory support the three datasets. The anti-claim is that the apparent gain is explained by training/evaluation exposure or by an unmatched protocol.

The next concrete deliverable is three completed, clean-training full-model seeds with original-population predictions and exact provenance, followed by matched corrected ablations. If these are weaker than the external leaders, report the gap and preserve the results. Any subsequent method improvement must start from validation diagnoses within a new explicit experiment contract; it must not be an automatic resurrection of failed LLM/routing routes or a hidden selection on the new test scores.

This stage advances the user objective but does not yet satisfy SOTA, six-domain coverage, all-component contribution or final-paper readiness. Those remain active quality requirements, not waived requirements.
