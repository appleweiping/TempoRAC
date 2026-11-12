# First completed ablation evidence acceptance

Observation: 2026-11-12T14:59:38.971574+00:00. Actual run_complete event: 2026-11-12T14:58:15.642033+00:00. One of21 new arms completed: wo_pyramid_s0.

| Requirement | Status | Evidence |
|---|---|---|
| Preserve live jobs and frozen scope | Pass | COMPLETION_RECEIPT.json process_control=false; worker435832/trainer482574 on wo_pyramid_s1,1logged epoch; worker435833/trainer435853 on wo_shift_s0,24logged epochs; both alive |
| Completed arm only, exact full training | Pass | raw/wo_pyramid_s0/complete_known_source.json;50contiguous log entries, epochs0..49; no nonfinite steps, finite loss fields |
| Exact frozen config and provenance | Pass | Raw config, provenance and first_checkpoint_receipt; matching contract dc60e3d..., worker d23bdd79..., source489dc77; cfg9bbb203e... |
| All fixed evaluation populations | Pass | RC152/positive151, CX2403, UCF105; both saved prediction-view keysets exactly match all corresponding manifest keys; finite predictions |
| Completion artifact integrity | Pass | All9completion-listed files match SHA, including three checkpoints hashed on server; no checkpoint/media/cache copied |
| Durable compact saved evidence | Pass for server; local transfer verified separately | COLLECTION_SHA256.json, raw small files, COMPLETION_RECEIPT.json; LOCAL_TRANSFER_VERIFICATION.json records final copy status |
| Component contribution and ranking | Pending | Paired three-seed groups incomplete; no interim cross-component ranking or contribution conclusion |
| Executor identity | Partial | Requested gpt-6.1-sol/high; actual backend not independently attested; same-family semantic assurance provisional |

Only saved-file parsing and hashing ran on the server. No new inference/training/tests, model loads, dependencies, HOLD files, signals, start/restart/resume/migration, repository/Git/status/source_sync edits, or C:/D: output writes occurred in this collection pass. Existing finite worker continuation was observed, not initiated by this collector.
