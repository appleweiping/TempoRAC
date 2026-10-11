# TempoRAC all-experiment result ledger — 2026-08-18

This is the append-only human-readable ledger for the controlled
`partial-cache` / `GT-bbox-assisted` / `supplied-track` pilot. A missing,
abstained, failed, or gate-blocked result is recorded explicitly and is never
silently omitted. No entry in this file is main-paper eligible unless its row
states otherwise.

## Dataset and checkpoint binding

| Item | Value |
|---|---|
| Train population | 268 identities / 110 videos |
| Validation population | 134 identities / 51 videos |
| Train SHA-256 | `c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb` |
| Validation SHA-256 | `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251` |
| Selected diagnostic teacher | `teacher_s20260815_v8_naturaldecoder_1000.pt` |
| Teacher SHA-256 | `5e5588cf2439154aff2ef40538e800e3035c6770c64318b623287c7c3b2bee50` |

## Completed entrance and certificate experiments

| Experiment | Split | Result | Status |
|---|---|---:|---|
| Frozen preprocessor | train | 172/268 (64.18%) | FAIL entrance |
| Frozen preprocessor | val | 91/134 (67.91%) | FAIL entrance |
| Resampled v2 preprocessor | train | 268/268 (100%) | diagnostic PASS |
| Resampled v2 preprocessor | val | 134/134 (100%) | diagnostic PASS |
| v6 teacher X0 certificate | tune | 56/56 | diagnostic PASS |
| v8 stable-pair certificate | train first 20 | 15/20 | diagnostic |
| v8 stable-pair certificate | val first 20 | 12/20 | diagnostic |
| v9 structurally filtered stable pair | train all | 155/268 (57.84%) | diagnostic |
| v9 structurally filtered stable pair | val all | 98/134 (73.13%) | conservative diagnostic |
| v10 DTW stable pair | val all | 102/134 (76.12%) | not promoted; 6 phase-class failures |
| Full legal chain availability | train all | 212/268 (79.10%) | feasibility |
| Full-chain retained-event fraction | train all | 90.98% mean | feasibility |
| Full-chain geometry-arc q80<=0.04 | train all | 161/268 (60.07%) | feasibility |
| Full-chain mean-DTW q80<=0.04 | train all | 185/268 (69.03%) | best complete-chain feasibility |
| Full-chain median-DTW q80<=0.04 | train all | 177/268 (66.04%) | not promoted |
| v11 complete-chain entrance | train first 20 | 13/20 | FAIL precheck: 3 landmark, 3 phase-class, 1 reconstruction |
| v11 complete-chain entrance | val first 20 | 7/20 | FAIL precheck: 5 landmark, 4 phase-class, 3 topology, 1 reconstruction |
| v12 DTW reconstruction + geometry-arc phase | train first 20 | 15/20 (75.00%) | diagnostic: 3 landmark, 2 reconstruction |
| v12 DTW reconstruction + geometry-arc phase | val first 20 | 13/20 (65.00%) | diagnostic: 5 landmark, 2 reconstruction |
| v12 DTW reconstruction + geometry-arc phase | train all | 191/268 (71.27%) | FAIL precheck: 56 landmark, 17 reconstruction, 4 collision |
| v12 DTW reconstruction + geometry-arc phase | val all | 96/134 (71.64%) | FAIL precheck: 28 landmark, 9 reconstruction, 1 origin |
| v13 fixed v12-to-v9 cascade | train first 20 | 16/20 (80.00%) | diagnostic: 4 landmark |
| v13 fixed v12-to-v9 cascade | val first 20 | 14/20 (70.00%) | diagnostic: 6 landmark |
| v13 fixed v12-to-v9 cascade | train all | 202/268 (75.37%) | FAIL precheck: 62 landmark, 4 collision |
| v13 fixed v12-to-v9 cascade | val all | 104/134 (77.61%) | FAIL precheck: 30 landmark; 4 identities short of 80% |
| v14 all-return-valid landmark availability | train all | 189/268; 10 recovered beyond v1 | feasibility only; loses 33 v1-available identities |
| v14 all-return-valid landmark availability | val all | 105/134; 5 recovered beyond v1 | feasibility only; cannot replace v1 |
| v14 fixed fallback cascade | train first 20 | 16/20 (80.00%) | no gain; 3 landmark, 1 topology |
| v14 fixed fallback cascade | val first 20 | 14/20 (70.00%) | no gain; 5 landmark, 1 topology |
| v14 fixed fallback cascade | train all | 202/268 (75.37%) | FAIL precheck; zero net gain over v13 |
| v14 fixed fallback cascade | val all | 104/134 (77.61%) | FAIL precheck; 28 landmark, 2 topology; zero net gain |
| v15 certificate-bound fallback | X0 ledger test | 1/1 | PASS; frozen X0 certificate ledger identical |
| v15 certificate-bound fallback | train first 20 | 16/20 (80.00%) | diagnostic; 4 landmark |
| v15 certificate-bound fallback | val first 20 | 14/20 (70.00%) | diagnostic; 6 landmark |
| v15 certificate-bound fallback | train all | 213/268 (79.48%) | diagnostic; +11 over v13; 50 landmark, 1 reconstruction, 4 collision |
| v15 certificate-bound fallback | val all | 109/134 (81.34%) | PASS numeric coverage; +5 over v13; 25 landmark |
| v15 author protocol preflight | bound artifacts | PASS | independent review still missing; zero launch authority |
| A005 Round 9 successor author prechecks | two implementations | PASS | independent reviewer missing; fixed aliases unchanged |
| A005 Round 9 independent review request | exact successor bytes | READY | awaiting independent `ACCEPT` or `REVISE` |

## Existing natural counting diagnostic

| Metric | Value | Eligibility |
|---|---:|---|
| Raw count MAE | 21.604477611940297 | X0-only diagnostic; not TempoRAC result |
| Relative per-identity NMAE | 0.9915196743554954 | same |
| OBO | 0.0 | same |

## Frozen 27-job campaign ledger

All jobs reached terminal `BLOCKED_PRECHECK`: the complete-sequence v11 natural
target runtime failed both train and validation entrance checks before any
formal training. The subsequent v12 candidate removed the observed phase-class
and topology failures, but its complete validation coverage was only 96/134
(71.64%), below the 80% entrance requirement and below v10's 102/134 diagnostic
coverage. Planned jobs are not reported as zero-valued metrics.

The later v13 fixed cascade improved validation coverage to 104/134 (77.61%)
without consulting count labels. This is the best complete-runtime validation
coverage so far, but it remains four identities below the minimum passing count
of 108/134 (80.60%). It therefore does not reopen the frozen campaign.

The v14 landmark sweep found 10 train and 5 validation identities unavailable
to the v1 complete-chain selector. However, the complete v14 fallback certified
none of those additional identities: train and validation stayed exactly at
202/268 and 104/134. Two validation failures moved from LANDMARK to TOPOLOGY.
The internally inconsistent v14 fallback is therefore not promoted.

The v14 zero gain was traced to a protocol binding defect in the candidate:
its runtime generated v2 landmarks while its copied certificate recomputed v1
landmarks. v15 copied the certificate again and bound both sides to v2. Ruff and
the X0 ledger-equivalence test passed. The corrected cascade reached 213/268
train (79.48%) and 109/134 validation (81.34%), exceeding the validation numeric
entrance requirement of 108/134. Because v15 was designed after inspecting the
existing validation diagnostics, this is a diagnostic gate pass only and must
receive independent protocol review before any campaign can be reopened.

| Run ID | Exact job | Steps | Status | Result |
|---|---|---:|---|---|
| TCH-20260815 | `temporac.execution.v4/teacher/seed=20260815` | 20000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| TCH-20260816 | `temporac.execution.v4/teacher/seed=20260816` | 20000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| TCH-20260817 | `temporac.execution.v4/teacher/seed=20260817` | 20000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| CAN-20260815 | `temporac.execution.v4/response/canonical/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| CAN-20260816 | `temporac.execution.v4/response/canonical/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| CAN-20260817 | `temporac.execution.v4/response/canonical/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| CAP-20260815 | `temporac.execution.v4/response/capacity-control/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| CAP-20260816 | `temporac.execution.v4/response/capacity-control/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| CAP-20260817 | `temporac.execution.v4/response/capacity-control/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SNP-20260815 | `temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SNP-20260816 | `temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SNP-20260817 | `temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SNU-20260815 | `temporac.execution.v4/response/shortcut/nuisance-only/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SNU-20260816 | `temporac.execution.v4/response/shortcut/nuisance-only/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SNU-20260817 | `temporac.execution.v4/response/shortcut/nuisance-only/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SPS-20260815 | `temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SPS-20260816 | `temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SPS-20260817 | `temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SSC-20260815 | `temporac.execution.v4/response/shortcut/static-code-only/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SSC-20260816 | `temporac.execution.v4/response/shortcut/static-code-only/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SSC-20260817 | `temporac.execution.v4/response/shortcut/static-code-only/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SWM-20260815 | `temporac.execution.v4/response/shortcut/warp-metadata/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SWM-20260816 | `temporac.execution.v4/response/shortcut/warp-metadata/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| SWM-20260817 | `temporac.execution.v4/response/shortcut/warp-metadata/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| STL-20260815 | `temporac.execution.v4/response/shortcut/track-length/seed=20260815` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| STL-20260816 | `temporac.execution.v4/response/shortcut/track-length/seed=20260816` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |
| STL-20260817 | `temporac.execution.v4/response/shortcut/track-length/seed=20260817` | 10000 | BLOCKED_PRECHECK | v11 natural entrance failed |

## Downstream result inventory

| Block | Status | Result |
|---|---|---|
| Table 1 main comparisons | BLOCKED_UPSTREAM | No eligible TempoRAC predictions |
| Table 2 seven ablations | BLOCKED_UPSTREAM | No eligible three-seed predictions |
| Synthetic 3x3 specialization | BLOCKED_UPSTREAM | Response checkpoints missing |
| Nonstationary-tempo stress | BLOCKED_UPSTREAM | Response checkpoints missing |
| Tracking HOTA/IDF1/IDS | BLOCKED_SCOPE | Supplied-track partial pilot; prediction artifacts missing |
| 10000-draw bootstrap CI | BLOCKED_UPSTREAM | Complete paired prediction population missing |

## Current decision

`HOLD_INDEPENDENT_REVIEW`. The user authorized continued server experiments on
2026-08-18, but no scientific gate was waived. v11 failed the complete-sequence
natural target preflight. v12 then passed Ruff and was executed against the
bound v44 population and selected teacher checkpoint, but achieved only
191/268 train (71.27%) and 96/134 validation (71.64%) certification. It removed
the v11 phase-class/topology failures, yet did not reach the 80% entrance
threshold. The fixed v13 cascade subsequently reached 202/268 train (75.37%)
and 104/134 validation (77.61%); all 30 remaining validation abstentions were
landmark failures. Because the required validation count is at least 108/134,
all 27 formal training jobs and their downstream Table 1, Table 2, stress,
tracking, and bootstrap evaluations remain blocked for this campaign version.
The v14 expanded landmark fallback produced zero net certifications because its
runtime and certificate were bound to different landmark selectors. The v15
certificate-bound correction reached 109/134 validation (81.34%), numerically
clearing the 80% entrance threshold. Nevertheless, v15 was formulated after
inspection of this validation population. The frozen 27-job campaign remains
closed until an independent reviewer accepts v15, its effective contract is
rebound, and a fresh campaign ledger is authorized. No MAE/OBO is assigned to
the still-blocked formal jobs.

The 2026-08-19 author preflight also reconfirmed that A005 fixed aliases remain
the rejected Round 8 bytes and Round 9 remains unfinished. Therefore the hold is
both local (v15 independent/untouched-population review) and global (A005,
effective-contract, native-runtime, P00/P06, and launch authority).

A new timestamp pair `20260819_195424` repairs the four handed-off Round 9
issues and passed Python/Docker structural plus PowerShell semantic/hash author
prechecks. These are not independent reviewer evidence. The pair has not been
copied to fixed aliases, so the global hold remains unchanged.
