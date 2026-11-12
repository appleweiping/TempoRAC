# TempoRAC current evidence and acceptance — 2026-11-11

This checkpoint supersedes the November 4–5 execution snapshots for resuming work. It does not erase those runs or upgrade their scientific acceptance. The active implementation is `temporac_cvpr/`, inherited from commit `489d3ab8fb1b83ffd5f247247bf4d4effe22441c`; the current experiment backend is the isolated AutoDL TempoRAC project. Local experiment execution and new recurring tasks remain prohibited.

## Verified work

- Collected the actual server configs, logs, predictions and results for 29 completed frozen-framework runs: five full-model seeds, fifteen component-removal runs, and nine single-dataset runs. The five main seeds use the same configuration apart from seed/output path.
- Recomputed the stored full-population scores on the server, checked identities and completion, and added matched-seed 0–2 contrasts, video/source-cluster bootstrap intervals and supplementary metrics. Intervals condition on the trained models; sample SD over seeds is separate.
- The current method uses a frozen ST-MAE encoder, blocks 8 and 12, a grid-aligned stride {2,4,8} pyramid, an auxiliary stride-16 cache for shifted views, a 3,878,788-parameter decoder, and dataset-token/balanced joint training. It has one density-MSE objective; the frozen version has no LLM or encoder adaptation.
- The 121 deployed source files with local counterparts matched their local SHA-256 values. Deployment was a copied source tree without a Git checkout, so this is a source-file receipt, not a verified server Git HEAD.

| Historical frozen result | Positive-count MAE | OBO | Seeds |
|---|---:|---:|---:|
| RepCount-A | 0.143115 ± 0.007599 | 0.711258 | 5 |
| UCFRep | 0.194398 ± 0.017183 | 0.716190 | 5 |
| Countix | 0.235985 ± 0.004407 | 0.705035 | 5 |
| QUVA zero-shot | 0.182591 ± 0.011033 | 0.562000 | 5 |

These are retained observations, **not a claim of clean independently selected SOTA**. Method selection followed earlier test inspection. The data findings below add a separate independence problem.

## Data findings and bounded correction

Two RepCount-A train/test pairs have identical actual media bytes and identical cached feature arrays at both depths and all four rates, despite conflicting counts:

- training `stu4_50` (3) / test `stu4_5` (51);
- training `stu3_27` (70) / test `stu4_27` (56).

The corresponding Lance source blobs also collide. Historical receipts from three earlier original-media trees independently confirm the second pair. This is not merely a stale manifest hash. The provenance of the first pair and the conflicting evaluation annotations remains unresolved.

Two Countix train/validation pairs use overlapping segments of the same source videos (`t6Ny1kl7Mxs`, `s5stimPX3h0`). Their raw-file SHA equality does not mean that the cropped model inputs are exactly identical.

All original manifests, predictions, checkpoints and negative results remain intact. The active correction is a versioned **training-only** source-exclusion protocol: preserve all evaluation rows and remove training instances whose known source also appears in validation/test. Reuse all verified encoder/feature caches. Previously exposed decoder weights cannot certify this corrected training protocol. The correction does not resolve upstream test-label quality or make already inspected tests unexposed.

The first three full runs are now released: seeds 0 and 1 started on separate GPUs; seed 2 follows seed 0 in a finite queue. Both initial jobs completed epochs 0–3 with finite training losses by 2026-11-12 05:32 UTC. Intermediate validation scores do not trigger protocol changes. Each has 50 epochs, 5,309 unique training videos and 8,570 balanced entries per epoch. The derived manifest is `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`. Twenty-seven total full/ablation specifications were frozen before new scores; the other 24 have not been released. Existing equivalent clean UCF-only runs may be reused after provenance verification. This is a finite experiment campaign, not a recurring task.

The user's latest execution rule is explicit: collect questions before a new group, then lock its data, metrics, design and matrix. During active experiments perform only runtime monitoring and necessary fault repair. In parallel, inherit the completed literature work and fill missing coverage for a 10–15-framework ledger; that work must not change the running group. Do not invent internal external baselines or delete selection history to create a universal SOTA claim.

## External baseline provenance

| Baseline group | Actual evidence in this project |
|---|---|
| ESCounts, released RepCount checkpoint | Official decoder/encoder weights; inference records on RepCount-A, RepCount→UCFRep transfer, and QUVA. No self-training for these rows. |
| RepNet, Context, TransRAC, Full-Res, MFL, DeTRC, D²-STX, CountLLM, PoseRAC, PAMS in the literature table | Published numbers, including some third-party reproductions reported in other papers. They are not all our own executions. |
| `escounts_style`, `tsm`, `repnet` custom-head runs | Internal controlled implementations. They are not official external-method reproductions. |
| PoseRAC weight asset | Present; asset existence is not a completed qualified evaluation. |
| ESCounts warped-test evaluation | Partial OOM run, 23/152 records; not complete. |

Our released-weight ESCounts scores are RepCount-A 0.215098/0.556291 (151 positive-count videos out of 152), RepCount→UCFRep transfer 0.341636/0.466667 (105 videos), and QUVA 0.299129/0.300000 (100 videos). The UCFRep transfer must not be relabelled as reproducing the published in-domain 0.216 result. Historical run-time weight digests were not located; current weight hashes and code/launch-path evidence are recorded separately.

## Acceptance table

| Requirement | Status | Evidence or remaining work |
|---|---|---|
| Resume actual latest work, reuse caches | Pass | Actual server files and 29 completed runs collected; no cache rebuild. |
| Full formal training/evaluation executed | Partial | Historical runs complete; corrected training protocol and evaluation still pending. |
| Source-disjoint and qualified inputs | Not met | Four known cross-split source groups; media/label conflicts require qualification. |
| Top-conference-scale external performance comparison | Partial | Broad literature table, one actual released-weight comparator; official-method execution coverage incomplete. |
| Every component contributes | Partial | Matched mean effects and uncertainty reported; second depth and joint training on RepCount-A have intervals spanning zero. No blanket significance claim. |
| SOTA on all intended domains | Not met | UCFRep/Countix trail stronger published values; protocol/selection/data issues prevent an unqualified claim. |
| Six independent datasets × seven frameworks preference | Not met | Four evaluated datasets, and no complete common-protocol matrix. |
| Observation, sensitivity, efficiency | Partial | Existing mechanism evidence retained; final-framework sensitivity and full-pipeline efficiency remain incomplete. |
| Manuscript fidelity | In correction | Selection, metric population, source overlap, padding, loss and baseline provenance are being corrected; old PDFs preserved. |
| Final publication-quality assurance | Not met | Independent final claim/citation/proof checks and all-page rendering still required. No submission authorized. |
| Local/GitHub/server synchronization | Partial | Source-file parity verified; current GitHub branch is behind newer local work until the publication receipt is recorded. |

Machine-readable raw analysis is retained under `E:/ICASSP_TempoRAC/aris-work/resume-20261111/server/` and the corresponding server analysis directory. Astra scientific decisions and independent Sol review are retained alongside it. Requested models are recorded; backend identity is not independently attested and same-family semantic review remains provisional.

## Literature and manuscript checkpoint

The inherited plus supplementary ledger now covers 15 framework families, with 13 verified comparison-table counts. See docs/literature/COVERAGE_15_20261111.json and LITERATURE_DELTA_20261111.md. This is not a claim that all 15 full texts or official runnable repositories are available. The two remaining table-count/full-text gaps are explicit. The running matrix is unchanged.

The revised manuscript source is an explicitly historical draft with selection, population, source-overlap, loss, finite-padding and external-baseline provenance corrections. Server-only compilation/rendering is in progress in an isolated paper environment; this source checkpoint does not certify a final PDF or submission readiness.
