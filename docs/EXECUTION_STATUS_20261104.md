# TempoRAC execution checkpoint — 2026-11-04

This is an engineering/data checkpoint. Formal model experiments, final scientific review and writing remain pending. Current private state and server run logs take precedence over this snapshot.

| Item | State | Evidence / next concrete work |
|---|---|---|
| Continuation / ARIS | Passed mechanically | Existing `3e9ae828` method retained; exact ARIS `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`; stage full-read receipts; no method restart |
| Literature / novelty | Provisional | Closest-work evidence and Astra caution preserved; MultiCounter+/TWCRAC full reading and original result reconciliation still incomplete |
| New server / separation | Passed infrastructure | Authenticated isolated TempoRAC root; all GPUs occupied at startup; STaR and others untouched |
| Complete candidate population | Preserved | 794 train / 183 development / 180 A candidate test; release/recipe identity not certified; no test labels read |
| Composite byte audit | Passed mechanically | 977/977, 4.62GB, zero failures; four train duplicate pairs, no train/val byte duplicate |
| Original-source byte audit | Passed mechanically, independence partial | 906/906, 8.41GB, 301 byte IDs; A180 retained, B78 candidate only, 102 overlaps; 9 B source components |
| Conflicting train labels | Observed; unresolved | Four exact-media pairs disagree 58/48 on aliased person count/period; bbox/periodicity agree; both labels retained |
| Full decoded frames / clocks | Running | PID1543699, 977-video loop, single CPU decoder, all raw integer clocks; never substitute header or initial receipts for full completion |
| Fresh input audit review | No remaining engineering blocker; same-family provisional | Fresh Sol/xhigh caught decoded-counter defect, fixed/rechecked; 16 CPU tests pass. Raw original finding retained in INPUT_AUDIT_REVIEW_20261104.md |
| Native pose frontend / contract | Assets and design ready; implementation in progress | Offline COCO17 snapshot hashes pinned; Astra native-N+bbox projection contract; full population CPU projection and implementation review next |
| Learner / natural targets / scorer | Pending qualification | Dynamic cache separate from fixed320; metric contract and geometry-target validity next; do not start old27 jobs blindly |
| Main results / 3seed / baselines / ablations | Not met | No natural full experiment result; six-dataset/seven-framework quality rule remains unmet |
| Observation / hyperparameters / efficiency / loss<3 | Not met | Existing method components retained; full mechanism/efficiency evidence and loss contract still required |
| CVPR final paper | Not met | Preserve prior drafts/numbers with reconciliation; experiments precede final writing, all-page render required for final PDF changes |
| Source synchronization | Pre-change common HEAD acc2bbf3000a7596336e7cff9864e295173fd888 verified local/GitHub/server | This checkpoint must be committed, pushed and fast-forwarded on server before claiming new common source; data/result receipts separate |

Storage startup: approximately81GiB free on project SSD and41GiB shared-data mount. Existing videos are reused read-only; no raw-frame duplication or cleanup. Detect-only ARIS watchdog uses E-disk state, stale threshold6h; it cannot create scientific verdicts or restart jobs.

Operational deviation: one author pytest invocation omitted TEMP/TMP/basetemp and created four fixture files (261 payload bytes) in a D-disk pytest temp directory. Subsequent checks explicitly use E for temp/cache/basetemp. Automatic policy rejected recursive cleanup and then a fixed-four-file cleanup before execution, giving only `blocked by policy`; files remain and no bypass was attempted. The all-E artifact requirement is therefore partial for that invocation. Research execution continues on E/server.

No Word/PDF was changed at this checkpoint. Private credentials, original inventories, restricted media/weights and transport logs remain outside Git.


Latest code checkpoint: full977/2364 bbox projection complete; fresh Sol/xhigh extractor review has no remaining concrete blocker after two resume corrections,12CPU tests pass. `NATIVE_POSE_CACHE_20261104.md` contains exact full-generation command; no cap. Raw generation is pending only the source-bound GPU allocation launch at this snapshot; no full CPU or GPU micro inference was used. Astra metric contract is documented, while official count parity and final primary freeze remain pending. Actual later source sync and job PID/log state are recorded in private current_status.json/server source_sync.json, rather than editing prior snapshots into completion claims.
