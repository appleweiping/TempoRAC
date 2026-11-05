# TempoRAC: external baselines and next complete experiment

Date: 2026-11-05. This is an execution decision and evidence inventory, not a completed benchmark or a SOTA claim. The active original experiment remains bound to source `751059bec69b98a76845a125c574add021cf5453`. A documentation commit does not change that experiment's source.

## Decision

Main comparisons will use published external frameworks and their native benchmark protocols. Internal canonical/control variants are ablations, never additional baselines. The first new complete evaluation uses the first fully ready official checkpoint on the full RepCount-A regular validation split. ESCounts currently has an obtainable decoder/encoder; TransRAC's trained checkpoint must still be recovered or acquired. Neither blocks the other. Then evaluate the corrected official RepNet implementation as its complete inputs become available. Do not substitute debug subsets, old 84-video diagnostics, or reconstructed proxy networks.

The user explicitly cancelled scheduled execution. Automation `temporac-cvpr` was deleted on 2026-11-05; no replacement schedule is authorized. Continue within the active chat. Preserve the existing remote control job and its natural first-pair stopping rule; do not restart canonical or release the remaining seeds.

## What comparable papers actually evaluate

Counts below are distinct external methods in the specified table, excluding the paper's own method and duplicate backbone/reproduction rows.

| Paper | Accurate venue | Main external comparison | Dataset scope |
|---|---|---|---|
| CountLLM | CVPR 2025 main conference | Table 1: 7; Countix Table 2: 3 | RepCount, UCFRep, Countix; its two backbone rows are one framework |
| MFL | WACV 2024 | RepCount Table 1: 8; UCFRep Table 2: 7 rows but 5 distinct methods | RepCount-A and UCFRep; includes action-recognition architectures adapted for counting |
| PAMS | CVPR 2026 Findings, not main conference | Table 1: 10 total; RepCount/Countix/UCFRep: 10/8/10 | Mixes pose, RGB and MLLM methods; input and supervision must be labelled |
| TransRAC | CVPR 2022 main conference | Table 2: 6 | RepCount-A/B and UCFRep transfer; Countix appears as dataset statistics, not a tested result |

Sources: [CountLLM, sections 4.1 and Tables 1–4](https://arxiv.org/html/2503.17690v1); [MFL, Tables 1–2](https://openaccess.thecvf.com/content/WACV2024/papers/Li_Repetitive_Action_Counting_With_Motion_Feature_Learning_WACV_2024_paper.pdf); [PAMS, Table 1](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_Counting_CVPRF_2026_paper.pdf); [TransRAC with supplementary tables](https://arxiv.org/pdf/2204.01018).

The competitive external pool is RepNet, Context, Sight & Sound, TransRAC, MFL, ESCounts and CountLLM. PAMS is an additional close pose-based competitor. This is a candidate pool, not seven reproduced methods and not seven verified CVPR/ICCV/ECCV publications: MFL is WACV, ESCounts is ACCV, and the publication venue of Full Resolution Repetition Counting remains separately qualified. The most competitive unavailable method must remain visible as a gap rather than be replaced by an internal variant.

Use native RGB, RGB+audio/flow, pose, MLLM, supplied-track and test-exemplar settings with explicit labels. Label each actual row, not only the framework family: Sight & Sound includes sight-only and audiovisual results, and the frequently reused UCFRep result is sight-only. ESCounts exemplar-free inference is the relevant ordinary comparison; labelled test exemplars belong in a separate setting. A stronger backbone or extra supervision is not a controlled component comparison.

## Data and metrics

| Dataset | Published size and protocol | Immediate use |
|---|---|---|
| RepCount-A regular | 1,041 videos; 758 train / 131 validation / 152 test, TransRAC supplementary Table 8 | First complete external evaluation; qualify all 131 validation media and original annotations before launch |
| UCFRep | 526 videos; original Context group split is 421 train / 105 validation; ESCounts CSV matches 421/105 | Second domain after exact split/ID verification; use an internal split of the original training portion for tuning |
| Countix | 8,757 videos in the compared literature | Third community domain, subject to complete legal media, native labels and task/input qualification |

The retrieved ESCounts Countix CSVs contain 4,277/1,406/2,453 rows before loader filtering, rather than a verified complete 8,757-video corpus. Record this release and filtering difference explicitly before any full-benchmark comparison.

RepCount-A open-set 655/130/256 is a different protocol. MFL reports 757/130/151 (sum 1,038), so its reported population cannot silently replace the 1,041-video regular release. MFL calls its 105 UCFRep held-out videos test, whereas Context/CountLLM use validation wording; compare exact IDs rather than infer a different count from those names. The early design's proposed 420/106 split was unsupported and is superseded by the verified 421/105 protocol. See [Context, Evaluation Datasets](https://arxiv.org/html/2005.08465). Reuse exact original videos already acquired; MultiRep composites and differently cut clips are not original RepCount videos.

The primary community metric is normalized absolute count error, often named **MAE** in these papers:

`mean_v(abs(pred_v - gt_v) / gt_v)` (lower is better).

The key secondary metric is **OBO**, the fraction with absolute count error at most one (higher is better). Preserve continuous native count outputs unless the exact official contract specifies rounding. Record zero-count handling, strict/non-strict OBO boundary, denominator epsilon, failed inputs and population explicitly. In particular, TransRAC code uses a denominator offset whereas its paper formula does not; save both native and harmonized scores with distinct names. PAMS prose does not state a GT denominator clearly enough to certify direct metric equivalence.

Raw count MAE, RMSE, OBZ, bias and efficiency can supplement the main metrics. Existing MultiRep video/person-macro NMAE is a development metric, not an official RepCount score. Never copy video-level count labels onto each detected person. Automatic primary-subject mapping must be frozen without test labels; supplied GT boxes/identities must be reported separately from automatic end-to-end inference.

The validation CSV contains one zero-count video. Preserve all 131 predictions. For this first adapted ESCounts run, the official `GT+0.1` scorer uses its density target on all 131; the unmodified paper formula is undefined for a zero GT and must be named separately on the 130 positive-count rows, with the excluded zero row's prediction and false-positive outcome still explicitly reported. These are development measurements, not certified published test comparisons. See `ESCOUNTS_VALID131_INTERFACE_20261105.md` for the frozen zero-shot two-offset interface and exact density/rounding rules.

The six-domain quality target remains open. QUVA, YTsegments and VSRep are qualification candidates, with HAC an alternate; they are not six completed benchmarks. Verify complete releases, label units, native metrics and source independence before allocating GPU. Non-human generic periodic videos cannot silently disappear from a full-benchmark claim. RepCount-derived MultiRep and Countix-derived subsets are not independent new source datasets.

## Component decision and bounded improvement

The old NOLA reconstruction is algebraically redundant under uniform routing or a single head: normalizing identical window responses returns the same response. Keep it as an implementation detail, not a third demonstrated scientific contribution. Do not remove the normalization denominator to manufacture a useful ablation.

Astra's proposed coupled structure is (A) local tempo/reliability, (B) learnable multiscale event evidence, and, conditionally, (C) tempo-constrained event-sequence decoding. C would use A's reliable local period and B's event response to discourage multiple inconsistent events per cycle. It is not implemented or validated. First recover existing full response arrays and the canonical/control pair; only proceed if fragmentation and reliable train-side period evidence support this hypothesis. A different observed failure must not be forced into this explanation.

The new component comparison is full, minus A, minus B, minus C, plus a targeted A×C interaction. Keep candidate generation and calibration identical for the C removal (`lambda=0`); distinguish removing a component from replacing its internal design. Use matched-capacity controls where necessary. Existing canonical/control evidence is reused.

Expansion requires useful component effects on complete development data. If a removal improves results or leaves no useful effect, stop expansion and perform one evidence-directed revision. Do not keep an ineffective module merely to preserve three names. Confirm contributions with three seeds and source-cluster uncertainty after the first complete development gate; no automatic release of remaining seeds.

Stop adding GPU work to an unsuitable development domain, retain all existing negative results and the dated selection record, and move to the next qualified domain. Freeze the final dataset set before final evaluation. A poor revealed test result stays in the final evidence; it cannot be relabelled as an abandoned development domain.

## Current evidence and immediate work

The original canonical run completed 10,000 steps and all 486 development identities. Its NMAE is 2.4144492875856693 and OBO is 0.04189435336976319; 466/486 identities overcount. An independent recomputation confirms the score. This remains a negative result. Checkpoint 4,500 was selected by the frozen X0 objective; no development-label replacement checkpoint is substituted.

The last trusted remote control observation was 1,137/10,000 steps at 2026-11-05 13:31 UTC. Later connection failures, including a fresh attempt during this work session, mean its current state is unknown, not that the job failed. Its original controller already includes full prediction/scoring and first-pair stop. Do not interfere with STaR-Body or other users' processes.

The official [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC) and [ESCounts](https://github.com/sinhasaptarshi/EveryShotCounts) repositories have been acquired on E:. TransRAC is fixed at `68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979`; ESCounts at `f18fcf1933abb3d4f199fd12ee3cafc53581576a`. Both are clean detached checkouts. The following 1,574,019,539 bytes were downloaded and verified; source availability alone is not a completed inference.

| Asset | Bytes | SHA256 |
|---|---:|---|
| ESCounts RepCount decoder | 239106669 | `297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559` |
| VideoMAE encoder | 1207498009 | `b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8` |
| Swin backbone, not a trained TransRAC model | 127414861 | `9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc` |

No third-party source, model weights, raw dataset or private transport material is committed with this document. Complete current media, feature tokens and a verified runtime remain missing. The native ESCounts `--only_test` entry point is not assumed to select validation; the exact val-only route must be checked before running the 131-video development set.

The isolated validation entry also has to correct a confirmed upstream encoder-loading defect: the feature script exits its outer parameter loop after composing the first fused qkv weight. The downloaded checkpoint actually has separate q/k/v keys and triggers this branch. The permitted correction uses the same repository's demo mapping and verifies complete encoder coverage; no encoder architecture or preprocessing change is allowed. At zero shot, the official decoder ignores exemplar input, so the adapted entry may uniformly omit unused exemplar extraction and loading, including the empty-period failure for the zero-count video.

Next execution order:

1. Finish full RepCount-A validation ID/media/annotation coverage and exact external checkpoint/dependency manifests. Historical remote inventories list all 131 validation IDs; historical content hashes cover 36, leaving 95 unverified rather than proven missing. These old inventories do not prove that the complete original corpus was migrated to the new server. Verify current locations and reuse before downloading replacements.
2. Run the first ready official ESCounts/TransRAC checkpoint on every qualified validation video, save per-video predictions and both scoring conventions, then the other method and corrected RepNet as ready. No arbitrary sample cap. Preserve each native code path's rounding, density-target construction, denominator offset and eligible population in its separately named native score.
3. Recover the existing original pair and response arrays when the tunnel is reachable; preserve source751 and all caches/checkpoints. Do not begin new MultiRep training.
4. Use the paired evidence for the single conditional component revision, then complete external comparisons and qualifying domains before final writing.

## Implemented validation entry

The isolated entry is now implemented in `scripts/escounts_valid131_common.py`, `scripts/escounts_valid131_extract.py` and `scripts/escounts_valid131_eval.py`. It imports the fixed official model, retains all 131 validation IDs, uses the frozen two-offset zero-shot recipe, records native and separately named paper-formula metrics, and supports verified reuse of completed token/row artifacts. Incomplete or mismatched artifacts are retained and rejected. It restricts output/cache paths to E: or the TempoRAC server project and uses only the explicitly selected single device.

After copying into this repository, the configured CPU suite ran eight checks: seven passed and full official model import was explicitly skipped because `iopath` and other runtime dependencies are unavailable. The original token-window function and all 131 original density targets matched the adapted paths. A separate checkpoint audit matched 152 checkpoint-derived encoder tensors to the official demo mapping; this is not coverage of an instantiated encoder. A fresh Sol review independently checked the numerical paths, mapping and resource boundaries and found no remaining concrete entry-code blocker within that scope; its semantic verdict remains same-family provisional.

Full model strict restoration/forward, actual RGB decode/transforms, GPU/autocast numerics and full VALID131 inference have not passed. Before extraction, the actual encoder's complete state must pass the coverage audit. The implementation has not been deployed to the unavailable server and does not change the original source751 job.

Entry commands require the verified official source directory and existing complete validation media/token paths:

```text
python scripts/escounts_valid131_common.py --source SOURCE --video-root VALID_RGB --output PREFLIGHT_JSON
python scripts/escounts_valid131_extract.py --source SOURCE --video-root VALID_RGB --encoder ENCODER --token-dir VALID_TOKENS --output TOKEN_MANIFEST --device SELECTED_DEVICE
python scripts/escounts_valid131_eval.py --source SOURCE --decoder DECODER --token-dir VALID_TOKENS --output PREDICTIONS_JSON --device SELECTED_DEVICE
```

These are argument templates, not records of an executed full-data command. The source/weight/CSV and per-artifact validation gates run before model work. The relevant configured test is `python -B -m unittest discover -s tests -p test_escounts_valid131.py -v`, with `ESCOUNTS_SOURCE` fixed to the official checkout, `PYTHONPATH` including the E-only einops dependency, and `ESCOUNTS_TEST_TMP`/`TMP` on E. Tests that require the external source explicitly skip when it is not configured; that default skip is not a substitute for the recorded configured run.

## Acceptance at this checkpoint

| Requirement | Status |
|---|---|
| Actual external-paper table counts and community metric definitions | Checked for four named papers; metric and split ambiguities explicitly retained |
| External frameworks replace internal variants in the main comparison | Decision recorded; new full external results pending |
| Official-model validation entry | Implemented; configured CPU interfaces checked and independently reviewed; complete model/data/GPU execution pending |
| Three components each make a real contribution | Not established; NOLA demoted, one evidence-gated C candidate proposed |
| Six real domains, at least seven persuasive frameworks, SOTA results | Not completed; remains the experiment quality target |
| No new schedules | Existing automation deleted; no replacement |
| Local/GitHub/server synchronization | This document needs its own publication receipt; active experiment source751 keeps its existing provenance; remote update pending connectivity |

Design: requested `gpt-6-astra`, effort `ultra`. Extraction/readiness/review: requested `gpt-6.1-sol`; ordinary independent review at `xhigh`. Actual backend identities are not attested, and same-family semantic review is provisional. Raw reports remain in the E-drive research workspace.
