# TempoRAC independent static evidence review

Date: 2026-11-11 local / 2026-11-12 UTC. Reviewed source: E:\ICASSP_TempoRAC\TempoRAC-CVPR, branch claude/cvpr27-temporac-rac, HEAD 489d3ab8fb1b83ffd5f247247bf4d4effe22441c.

Reviewer session: /root/evidence_review_sol. Requested reviewer: gpt-6.1-sol. Actual runtime model and reasoning tier: NOT ATTESTED by the native tool response; do not substitute requested identity. Relationship: same-family provisional by project policy, no cross-family acceptance. This is a source-and-evidence review with execution-log context, NOT the zero-context paper-claim audit required before submission.

Scope restrictions honored: local static file reading and document writing only. No server connection, no tests, model execution, dependency installation, statistical recomputation, source edits, Git writes, or current_status edits. Current run-result/per-video files were not found in the local repository file inventory. Numerical statements below come from checked-in summaries, not a fresh recalculation. The server executor owns live receipts and computation.

## Acceptance before review

| Requirement | Status | Evidence / remaining obligation |
|---|---|---|
| Claims match actual selection history | FAIL | Paper denies test selection; execution log explicitly records it. |
| Raw result and source linkage | PARTIAL | Summary tuples and generator are present; no current raw-run bindings, source-at-launch receipts or local per-video evidence. |
| Complete expected runs, equal populations, matched comparisons | PARTIAL | Intended 5 main / 3 ablation seeds; generator can silently omit files and mix globbed seeds. |
| Metric/sample population disclosed consistently | PARTIAL | Positive-count protocol in prose; table does not state n=151, historical literature has differing denominators/populations. |
| Cross-dataset source independence | UNESTABLISHED | Union builder checks key collisions only; no current source-overlap matrix. |
| Every component contributes | UNESTABLISHED | Point estimates only, unmatched 5-vs-3 aggregation, no CI; secondary-metric counterexamples preserved in table. |
| Observation/method evidence | PARTIAL | Several diagnostics, but rate knockout bins use duration/count, not annotated cycle medians; no observation figure/table. |
| Hyperparameter/stability evidence | MISSING FOR FINAL FRAMEWORK | Old sweeps/negative development exist; no frozen-framework sensitivity output found. |
| Efficiency | UNESTABLISHED | Claimed timing/parameter numbers lack current raw receipt; script profiles an obsolete interface. |
| Six datasets x seven persuasive frameworks / six SOTA domains | NOT MET | Three in-domain datasets plus QUVA transfer. Published comparison rows are not same-population reproductions. Current user-authorized scope must be acknowledged, not retroactively passed. |
| Submission assurance/rendering | NOT ASSESSED / PENDING | Existing paper/PAPER_CLAIM_AUDIT.json belongs to old paper/main.tex, not paper/cvpr27. No fresh visual QA in this static review. |

## Prioritized findings

### P0-1. False claim that test was not used for selection

Evidence: paper/cvpr27/sec/4_experiments.tex:39-42 states each model is evaluated once and "no test video is used for training or selection". docs/EXECUTION_LOG_20261108_CLAUDE.md:416-420 says final_v2/wo_llm was selected after test evaluation. Git history confirms 276d6ba reports the revealed test results, then 9294155 freezes the winning no-LLM framework. Appendix sec/6_appendix.tex:74-75 says adaptation hurt test and "we therefore keep the encoder frozen" but does not plainly disclose the post-test method and benchmark choice.

Fix now: distinguish checkpoint selection by original-view validation from framework and main-benchmark selection after observed test results. Keep all already revealed rows and original chronology. New seeds on the same exposed test set do not restore an untouched final evaluation. Use qualified exploratory/retrospective comparisons until an independent evaluation design is approved. This is a reporting correction; no retraining is required to make the paper honest.

### P0-2. Completion and aggregation can silently certify incomplete experiments

Evidence: final_tables_frozen.py:31-33 and :48 filter missing result/eval files; :42 and :61 glob arbitrary seeds; :82-83 prints max seed count across datasets. No expected row IDs, expected per-target populations, finite-value checks, common seed intersection or config equality checks exist. The JSON stores only mean/SD/count, losing source rows and paths.

The problem is reachable through actual orchestration: launch.py:85-87 emits QUEUE_DONE for both done AND partial runs. final_frozen_chain.sh:9 only greps QUEUE_DONE, uses set -u rather than failure propagation, and prints FINAL_FROZEN_DONE at :17 even if QUVA/table commands fail. Thus a partly failed queue can appear finished, and missing rows can silently disappear.

Minimal patch proposal (implementation-only, no method change): explicitly list main runs s0-s4 and every expected ablation s0-s2; require exact files per target and QUVA; reject duplicate/unexpected seed identities or report them separately; require matching target key, metric population, configuration differences restricted to named ablation, n/n_pos, and finite metrics. Emit a source table mapping each statistic to paths/run/seed/checkpoint/selected epoch/config/input population. Fail before writing final table fragments if anything is missing. A fail status must be distinct from successful completion; do not rerun or overwrite training.

### P0-3. Central exact-shift claim fails at normal finite-video boundaries

Evidence: paper/cvpr27/sec/3_method.tex defines V'[i]=V[2i] and claims exact stream identity. extract_features.py:encode_rate samples min(j+s*f, T-1), i.e. copies the last decoded frame for tail padding. For an even-length video, direct encoding of V[::2] pads with original frame T-2, while cached stride-2s encoding pads with T-1. The tokens involving these frames need not be equal. This is a source-level counterexample, not a measured accuracy effect. Dense cells near the start are also not each averages of four clips, contrary to the unqualified text/figure. Cached crops keep encoder context outside their crop; they are not the encoding of freshly cropped media.

Minimal remedy: have the server executor establish the precise finite-length/padding contract using existing assets; qualify exactness to aligned interior sampling / matching boundary convention unless a stronger current-artifact result is proved. Do not silently regenerate frozen features or call a boundary change equivalent without checking its effect. The algebra for interior clips is sound. Report which boundary convention the published results actually used.

### P1-4. No current artifact-to-source linkage; historical audit is for another paper

Evidence: final_tables_frozen.py:21 mixes final_v2/wo_llm_s0..2 and frozen_a/full_s3..4. make_frozen_specs.py copies configs and strips MLLM keys, but does not prove training code/feature/manifest identity across dates. train.py:372 saves cfg/args/params; :449/:487 saves state/cfg/epoch/target, without launch source revision. The 5-seed summary alone cannot establish source identity. Source HEAD contains later train/model edits than the old seed launch dates. Current Git SHA is evidence of source now, not evidence of source used then. Root reports remote GitHub branch is older; that independent observation should be included only with its own receipt, not inferred from this local review.

paper/PAPER_CLAIM_AUDIT.json is explicitly BLOCKED and hashes old paper/main.tex and paper/sections; it does not cover paper/cvpr27. docs/CITATION_AUDIT_20261109.md is a historical check and marks later additions pending verification. .aris/runs/temporac-cvpr-20261104.json records old MultiRep state and cannot serve as this workflow's completion receipt.

No-new-training closure: recover each run's launch log, command, config, selected checkpoint metadata, manifest versions, feature metadata, code snapshot/source commit if recorded, timestamps and file hashes on the server. Compare effective source/config between old and new seeds. If original source-at-launch cannot be recovered, record that uncertainty; never fill it with today's SHA. Hash current files for current inventory only. Bind paper rows to that inventory and keep separate local/GitHub/server sync receipts.

### P1-5. Metric and population comparisons are not harmonized

metrics.py computes positive-count MAE_p with round(pred), denominator gt+0.1; n includes finite GT, n_pos excludes zeros. The paper describes that formula correctly. RepCount's 152 supplied test videos contain 151 positive-count cases; the table must say n=151 for the reported score and retain train143's zero-count prediction/all-video metric. The source comment in tab/main.tex says test 152. docs/LITERATURE_NUMBERS_20261108.md and EXTERNAL_BENCHMARK_PROTOCOL_20261105.md explicitly record differing historical populations, epsilon/rounding/density-target conventions. A published 0.162 is not thereby a same-population/same-formula 0.162. SOTA claims must be bounded by compared published protocols, with paired raw-baseline claims reserved for shared available predictions. ESCounts reproduced checkpoint is the strongest current same-pipeline anchor, subject to confirming that both use annotated-count targets consistently.

Countix: latest paper/README states 4224/1374/2403 of the 4277/1406/2453 release; log's earlier matching stage says 4275/1388/2427, which is not the final population. prepare_countix.py omits missing/short clips and deduplicates; prepare_report.json truncates each ID list to 30 (:68). Paper promises exclusions in supplement but appendix contains no complete exclusion ledger. Countix literature subsets are not verified identical. Export complete IDs plus exclusion reasons and retained official-row IDs; score all available methods on the exact same retained subset when their raw predictions are available. Do not label it a full official corpus merely because no intentional random subsampling occurred.

### P1-6. Data leakage/source overlap remains an unclosed evidence obligation

make_union.py:47-49 checks only feature-key collision (<original split>__<local ID>). Distinct IDs can represent clips from the same source; split prefixes also hide cross-split collisions. Countix prepare_countix.py drops youtube_id from matched metadata although countix_index.py records it. Within-dataset duplicate removal is by (split, video_id), not source-video independence. UCF prepare_ucfrep.py reports class-group train/test overlap, but current report is absent locally. No joint-source train/development/test overlap matrix is produced.

Server-only closure: using manifests and existing matched index, join source IDs (YouTube/Kinetics clip intervals, UCF class+group, RepCount/QUVA recoverable original sources), exact byte hashes, and available source metadata across ALL datasets/splits. Report exact duplicates, same-source different clips, possible matches and unknown IDs separately. Do not declare absence of leakage from an empty exact-hash intersection. If overlap affects joint training vs target test, quantify affected population and obtain a design decision before excluding or re-evaluating. Encoder pretraining coverage (K400 versus Kinetics-derived Countix) should be disclosed separately from supervised train/test leakage.

### P1-7. "Every component contributes" is stronger than current comparisons

Final table compares main s0..4 with each removal s0..2. Reused seed IDs do not make the displayed aggregates paired. No per-seed paired delta, video/source-cluster CI, or multiplicity treatment appears. UCF removing shift improves MAE 0.192 versus 0.194 and OBO 0.778 versus 0.716; RC removing second depth improves OBO 0.728 versus 0.711. These counterexamples are in current table and must remain. The strongest supported statement is lower displayed mean RepCount MAE for full framework than these removals, pending paired evidence. A small mean gap does not establish real component benefit, equivalence or stability.

Additional confound: crop_video uses unit=BASE*max(load_rates). Disabling shift changes loaded max rate from 16 to 8; disabling pyramid changes it again. Identical random seeds therefore do not guarantee identical sampled crop boundaries. The joint-training control uses 80 single-dataset epochs vs 50 balanced joint epochs, so it is a protocol/data addition ablation, not equal per-dataset optimizer exposure. Do not promise perfect matched treatment beyond what configs establish.

No-new-training closure: use common main seeds s0..2 against each existing ablation; report 5-seed main separately. Recover rounded and continuous per-video predictions; compute paired seed deltas and source-cluster bootstrap CIs with prespecified families of comparisons; retain full populations and multiplicity choices. Describe augmentation/exposure differences. If effect remains weak, qualify the claim rather than declaring acceptance.

### P1-8. Mechanism statistic population is mislabeled

rate_knockout2.py:47 bins on v.T / fps / count, NOT on annotated cycle lengths. This is at best average video-duration-per-count; for untrimmed RepCount it includes idle time and for Countix it is a count-derived proxy. Paper sec/4b_analysis.tex calls these tertiles annotated cycle duration and infers the decoder assigns slow to coarse and fast to fine. The script only zeroes pooled streams at inference (distribution shift), which shows sensitivity/dependence, not optimal rate assignment or unique causal specialization. It discards per-video/per-seed output and saves rounded aggregate means (:66-67).

Minimum no-new-training closure: label the existing diagnostic exactly as implemented; reconstruct the intended period metadata from cycle boundaries and source clocks. If the original per-video knockout outputs were not saved, reproducing them requires server inference, though no training. Save them if run, stratify true annotated medians only where such labels exist, show counts/uncertainty, distinguish all three zeroed-rate effects and untrimmed/trimmed populations. Countix has no real cycle boundaries, so do not present its synthetic timing as annotated rhythm.

### P1-9. Efficiency script does not measure the claimed frozen architecture

paper sec/4_experiments.tex:49,54-55 claims 3.9M trainable params, 70 min for all 9.8k videos, 1.7 h per run and 1.7 GB. No current raw efficiency receipt is locally linked. efficiency.py:30 hardcodes 768-channel inputs; final train infers 1536 for b8+b12. Passing the saved full cfg would mismatch dummy inputs; omitting in_dim benchmarks a different model. It also omits domain input required by the final token-conditioned model. :63 counts encoder clips for model rates only, not the extra r16 stream. It times only one view, whereas inference averages original and shifted views. :21 uses a hardcoded approximate 180 GFLOPs/clip. This cannot substantiate complete final-method FLOPs/latency.

No-new-training closure: recover actual config.params and training/log peak/timing first; establish exact decoded population, encoder depth extraction strategy and which rates were included in the 70-min job (both b8/b12 and r16 are required). Report observed device RTX 4090D accurately and concurrent workload. Update profiling to build the saved cfg, use in_dim and domain, profile each/combined view with actual lengths and decoding+encoding+transfer+head boundaries; report token-cache disk/RAM cost. Forward profiling is server-only and requires no new training. Existing historical timing may be reported as historical, not substituted for a measured full pipeline.

### P2-10. Method spec differs from optimizer/target behavior

Paper equation averages one video's valid cells, but compute_loss averages all valid cells across the batch and multiplies density loss by 100 (default); longer videos get more influence than per-video normalization. With w_count=0 this is one active optimization term, satisfying <3 terms, but implementation detail should be stated because it interacts with multi-dataset length distributions and learning rate.

Paper says each full cycle target has unit mass and sum y=c; density_target normalizes over a grid extending beyond video boundaries, then truncates to [0,G). This preserves partial crop mass intentionally, but can lose Gaussian tails even for uncropped boundary-adjacent cycles. State that precisely; do not silently change old targets. Feature cells near the start have fewer than four contributing clips. Decoder actually expands rates to the BASE=2 grid before fusion (models.py to_base), while paper describes fusion on finest s=2 grid then output at 2-frame grid with no intervening upsampling. Resolve the geometry in prose.

### P2-11. Comparator coverage and publication claims need one current inventory

Current main table has RepNet, Context, TransRAC, Full-Res, MFL, DeTRC, D2-STX, ESCounts (+its reproduced checkpoint), CountLLM, PoseRAC and PAMS. ESCounts reproduced is not another independent framework; several entries are WACV/ACCV/IJCV/arXiv/CVPR Findings rather than seven top-main-conference methods. Literature inventory contains omitted IVAC-P2L, SkimFocusNet, HTRM-Net, RACnet, MultiCounter, FCA-RAC, and secondary-only GMFL/SPKDB/BIGC values. No evidence here that any is stronger than the displayed 0.143 under a matched setting, but their omission needs an inclusion/exclusion rationale and latest source review before "best among RGB". Multi-stride collaboration is closest to the central mechanism yet was abstract-only in novelty note; not an accepted novelty clearance. PAMS protocol ambiguity must remain explicit and cannot justify treating every other published row as fully comparable.

Suggested table inventory: method, publication/version, modality, train source(s), test population, zero/rounding/epsilon/target rules, source table/page, reproduced vs author-reported, and reason if omitted. Show mixed-protocol numbers as such. Do not restart discontinued methods or add training merely to fill a quota without a design/budget decision.

## Minimal server analysis package (no new training)

1. Inventory expected 26 new frozen runs + five main model references and target-specific result/prediction/config/checkpoint coverage. Verify all completions, n/n_pos and source/config provenance. Preserve originals.
2. Export complete data retention and source-overlap ledger across the joint union and QUVA. Freeze actual populations; retain zero-count outcomes.
3. Aggregate saved predictions on common seeds s0..2 for matched removals; 5-seed main separately. Supply per-seed point results, paired deltas, source/video uncertainty, secondary metrics and chosen multiplicity handling. Raw external checkpoint predictions only support pairwise claims on shared populations.
4. Export historical efficiency/config/epoch timing evidence and training curves; identify what applies to the final architecture. Optional measured profile needs model forwards but no training and must run on server.
5. Rebuild truthful tables with strict preconditions and direct source-row linkage. Correct post-test selection disclosure, sample sizes, Countix subset and mechanism labels before a fresh manuscript claim audit.
6. Assess final hyperparameter/stability obligation honestly: existing development grids may support documented exploratory behavior, but unrelated old tempogram sweeps cannot be renamed final-framework sensitivity. Epoch traces supply convergence stability, not sensitivity to sigma/rates/lr. New final sweeps require a later approved budget; no-new-training work alone may leave this requirement unmet.

## Review limits and protocol receipt

Read project E:\AGENTS.md, repository AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, CVPR_RESUME and run-state JSON. Full pinned f76ec8d6b77792e2242cca049e08d0cbb7d54e19 Codex skills read: integrity-forensics, experiment-audit, research-review, paper-claim-audit, result-to-claim. Supporting reviewer-backends, reviewer-independence, experiment-integrity and review-tracing read. This does not mean the integrity-forensics launcher ran: it requires its own deterministic server workflow and cannot certify CLEAN. No canonical audit artifacts were overwritten. Applicable scope: fresh static review only; primary raw result verification, source overlap computation, literature refresh, visual PDF QA, and submission assurance remain separate gates.

Overall verdict: FAIL for current submission-facing claim fidelity; PARTIAL evidence readiness. No accusation of fabricated numeric results: the local summaries may be correct, but the source linkage, reporting and incomplete scientific acceptance cannot be upgraded by a compiled PDF or a successful generator.
