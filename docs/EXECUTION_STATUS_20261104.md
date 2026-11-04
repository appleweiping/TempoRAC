# TempoRAC execution checkpoint — 2026-11-04

This checkpoint records actual execution, not a claim of completed research.

| Item | Verified state | Remaining work |
|---|---|---|
| Continuation | Existing TempoRAC branch inherited at `3e9ae828`; new branch `codex/temporac-cvpr-aris-20261104` | Continue the same method, preserve old evidence |
| ARIS | Exact `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`; 46-source read receipt and 20 installed stage skills | Read conditional child skills immediately before using them |
| Design/code review | Fresh Astra review; same-family provisional; no blocking timebase-audit finding | Does not establish model validity |
| CPU validation | Timebase tests: 13 pass on Python 3.11.2 / NumPy 1.26.4; Ruff passes; separate reviewer ran 13 tests and fixed-coordinate checks | No natural performance or GPU result implied |
| New server | SSH authenticated; 4 x RTX 4090, 24 GiB each; all occupied at startup | Reserve resources only once a complete runnable job is ready |
| Resource separation | TempoRAC and STaR-Body are equally high priority; existing STaR and other jobs retained | This chat changes only TempoRAC; no unknown job is killed |
| MultiRep candidate | Larger source has 794/183/180 video directories with corresponding MP4/JSON pairs | Release identity, source-video split overlap, media integrity and full pose extraction still need verification |
| Misleading small copies | Two `data/MultiRep` copies each contain only 14/13/14 videos | Excluded from claims of full-data experiments |
| RepCount candidate | Existing 758/131/152 videos and train/validation annotation rows | Verify official release, hashes and input protocol before use |
| Full CPU data audit | Started all 794 train + 183 validation MP4 files; streams SHA-256 and checks population/header consistency | No test annotation or test metric is read; full decoding and source-video overlap remain separate checks |
| GitHub | First checkpoint pushed at `4afb45a615677f740c61fcfcdd3859851228f11c` | Further literature/status changes are additive checkpoints |
| Server source | Dedicated owned root; initial source synchronized at `4afb45a615677f740c61fcfcdd3859851228f11c`, clean remote checkout, bundle SHA verified | Later checkpoints must each be synchronized; source/config sync is distinct from complete data/results |
| Recurring continuation | App heartbeat `temporac-cvpr`, every thirty minutes | A scheduled wakeup is not an experiment result |

The startup storage inventory showed approximately 81 GiB free on the selected SSD mount and 41 GiB free on the shared data mount. Full raw-frame duplication is not authorized by this observation: reuse existing read-only media and estimate pose/cache size first. No cleanup of either project was performed.

Private infrastructure coordinates, encrypted credentials, full filesystem inventories and transport logs stay outside Git. The media audit emits only engineering evidence. Exact file duplicates cannot exclude source-video leakage across different composite videos, and an MP4 header cannot establish successful full decoding.

The next scientific decision is the frozen natural target/timebase protocol on complete inputs, informed by the current literature and Astra review. Canonical full runs, strong controls, multi-seed ablations, uncertainty, efficiency and final CVPR writing remain pending. Existing PAMS/MotionBERT/partial-cache numbers are not TempoRAC results.
