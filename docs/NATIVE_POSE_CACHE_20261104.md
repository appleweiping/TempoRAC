# Native pose cache implementation — 2026-11-04

## SPEC (recorded before code)

Target: independent `temporac.native-pose.v1` raw cache, preserving all candidate train/val videos/identities, native xy+confidence, original source frame indices and observation masks. Inputs: exact media manifest, trusted train/val annotation root, existing decode audit, local ViTPose assets. Outputs: strict bbox-only projection, variable-length raw NPZ/JSON plus failure/attempt ledgers under a caller-provided project output. No existing fixed-320 model/feature/loss changes. Base commit: `acc2bbf3000a7596336e7cff9864e295173fd888`. Scope excludes training, metrics, test access, PTS reconstruction, GPU/full CPU inference launch. Success command: scoped pytest then full projection/validation only. Root owns independent review/commit; this worker does not commit.

## Assumption ledger (ASK=never; before dependent code)

| ID | Underdetermined point | Choice | Class | Source |
|---|---|---|---|---|
| A001 | Complete population contract | Production config requires train794/1878 persons and val183/486; fixture APIs may receive explicit expected census | semantic | user |
| A002 | File layout / corpus metadata interface | JSONL projection records plus separate provenance audit; raw per-video JSON+NPZ; config references existing manifest/audit paths | interface | default |
| A003 | Decode audit unavailable or incomplete | validate reports pending; extraction may use its own native source-frame decode count while recording missing sidecar/source_frame_only; never fabricates PTS or audit completion | semantic | user |
| A004 | L differs from decoded D | Preserve independent L annotation and D clock/decode axes, all L pose rows unsupported and alignment_unresolved; never map mismatched rows | semantic | user |
| A005 | Failed/absent/nonfinite pose | Zero invalid numeric payload, false masks with reasons; keep all slots/frames; low finite scores remain observations | semantic | user |
| A006 | Timestamp sidecar format | Reuse integer pts and rational time base when actually supplied; missing or only float timestamps recorded unavailable/source_frame_only, duplicates preserved | interface | default |
| A007 | Resume policy | Reuse only validated complete artifact matching full input/config/script/assets binding; preserve failed attempt metadata and retry | semantic | user |
| A008 | Complete raw vs inference failure | All videos get artifacts and full failure ledger; failed/alignment_unresolved artifacts do not qualify as completed successful pose extraction or learner input | semantic | user |
| A009 | Native frame decoder | Lazy OpenCV, original RGB, explicit thread count1; record backend/version; no frame dropping/resampling/interpolation | interface | user |
| A010 | Scientific bridge | audit only: reasons include fixed320 shape, missing support, old preprocessing restrictions; bridge not implemented | semantic | user |

## Ladder and predeclared checks

| Rung | Feature | Check | Tier | Initial status |
|---|---|---|---|---|
| F0 | CLI config -> honest plan JSON artifact | `python -B scripts/build_temporac_native_pose_cache.py plan --config fixture` | MUST | pending |
| F1 | strict bbox-only projection / population validation | scoped pytest projection cases | MUST | pending |
| F2 | raw assembly/masks/clocks/failures with injected fixtures | accumulated scoped pytest | MUST | pending |
| F3 | real offline frontend/decoder extraction, atomic/resume/validate_raw | accumulated scoped pytest; no real inference launched | MUST | pending |
| F4 | full projection census + Type-B review | full project/validate; review by root | MUST | pending |

## Run record

Initial ledger written before script/tests. Complete Astra input contract and skill read. Current GPU/all-CPU inference explicitly not authorized. No permission question.

## Deferred / blockers

Fixed-320 bridge is intentionally not implemented. Independent silent-assumption/code review delegated to root as explicitly instructed. Real GPU inference and complete raw generation wait for resources and review.

F0: plan CLI -> plan.json exit0; first PowerShell heredoc attempt had parse error before writes, fixed on attempt2. F0 artifact explicitly implementation_pending, not a pose result.


## Implemented interface and actual verification

F0 completed before the implementation. F1–F3 code and CPU fixtures are implemented; tests use explicitly synthetic frontend outputs in the test file only. No production stub emits a pose. `project` and `validate` import no NumPy/torch/transformers; `extract` lazily loads an exact local FP32/eval ViTPose and OpenCV, and requires explicit device, OMP/MKL=1. `validate_raw` loads numeric NPZ with allow_pickle=False and checks numeric schemas/masks/ownership; `audit_bridge` always returns inadmissible because a versioned learner bridge is not implemented.

Actual E-only CPython3.11 CPU test runs: first9 passed2.80s; after video preflight failure ledger10 passed2.39s; after formatting10 passed2.04s; after one additional real resume-path fixture11 passed2.97s. Ruff0.14.4 `format` reformatted the two owned Python files; `check --no-cache` passed before the final test-only resume case. The final test case was added after the root started review; root was notified. TEMP/TMP, pytest basetemp, PYTHONDONTWRITEBYTECODE/-B were explicit. The first local lint capture used the system GBK decoder and failed to display UTF-8 diagnostics; rerun with explicit UTF-8 captured diagnostics, fixed them and passed. No model forward was performed.

The full projection was generated on the server using `/usr/bin/python3 -B`, standard-library only, against the completed media manifest. The reviewed-script destination was not overwritten: execution used staging `/mnt/4tssd/lilei/TempoRAC/work/native-pose-implementation/build_temporac_native_pose_cache.py`. Projection provenance records that staging script's actual SHA; local formatting later changed script bytes, so do not misidentify the provenance SHA as the later final file SHA. Root owns final source synchronization.

Commands `project` then `validate` both completed REMOTE_EXIT0, using `/mnt/4tssd/lilei/TempoRAC/config/native_pose_v1.json`.

| Population | Videos | Fixed slots | Native annotation frames (sum L) | Person-frames (sum P×L) |
|---|---:|---:|---:|---:|
| train | 794 | 1878 | 523223 | 1314434 |
| development val | 183 | 486 | 172402 | 443600 |
| total | 977 | 2364 | 695625 | 1758034 |

All 1758034 supplied bbox rows are non-null (`bbox_present`), which does not itself prove positive crop validity or pose success. Media aliases remain separate records. Final validation snapshot had 157 decoded-audit rows available, 820 pending, zero L/D mismatch among available rows. This is not complete decode validation, and the file can change as the independent decode audit progresses. No missing timestamp or future audit record was fabricated.

Artifacts under `/mnt/4tssd/lilei/TempoRAC/native_pose_v1`:

| Artifact | Bytes if measured | SHA256 |
|---|---:|---|
| projection.jsonl | 40091158 | b53569b09c643ecdb041df8d4109fddf7d1bdeadfb2163a045f301b68b9761da |
| projection_provenance.json | — | effc4bbb11812217e4bb83128c765b68f499e5fca7a0e1eab9ec618ef99ff970 |
| projection_validation.json | — | 1cf6a58e4b4ae2e394d761cadb7befe7ca4bc129a2ccdead45f83950ea4b0705 |

Private E command/stdout receipts: `E:\ICASSP_TempoRAC\aris-work\tmp\native_pose_full_projection.sh/.txt`, `native_pose_projection_totals.sh/.txt`; config template `native_pose_v1.json` in the same directory. No source labels or projection raw arrays belong in Git.

Raw uncompressed numeric payload budget from declared axes is approximately `274×1758034 + 27×695625 + 8×2364 = 500502103 bytes` (477.3MiB), excluding JSON metadata, ZIP headers and compression effects. This is a cache-size calculation, not measured inference memory/runtime/ETA. Actual CUDA throughput and full extraction runtime are unmeasured; no full CPU inference was launched.

### Exact full-generation command after root review and explicit GPU allocation

The root must first synchronize the reviewed implementation into `$R/scripts/build_temporac_native_pose_cache.py`; the command below is **not a launch receipt**. The current config has source_root=`/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/MultiCounterPlus1`, completed media manifest=`$R/runs/media-integrity-20261104/media.jsonl`, existing decode_audit=`$R/runs/media-decode-20261104`, exact local a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45 snapshot, output=`$R/native_pose_v1`, batch_size=8. It contains no sample cap. An explicit assigned GPU is required; do not substitute a busy STaR-Body card.

```sh
R=/mnt/4tssd/lilei/TempoRAC
PY=/home/lilei/PoseRAC_Experimental_Code_20260828/.venv/bin/python
cd "$R"
mkdir -p "$R/work/native-pose-runtime" "$R/runs/native-pose-v1"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$R/work/native-pose-runtime"
export HF_HOME="$R/work/native-pose-runtime/huggingface"
export TORCH_HOME="$R/work/native-pose-runtime/torch"
export XDG_CACHE_HOME="$R/work/native-pose-runtime/cache"
export CUDA_CACHE_PATH="$R/work/native-pose-runtime/cuda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" "$PY" -B scripts/build_temporac_native_pose_cache.py extract \
  --config "$R/config/native_pose_v1.json" --device cuda:0 --resume
CUDA_VISIBLE_DEVICES='' "$PY" -B scripts/build_temporac_native_pose_cache.py validate_raw \
  --config "$R/config/native_pose_v1.json"
```

Root should wrap the authorized full command in its standard project-local PID/log launcher. If launch is from Windows, its background root must be Hidden/CreateNoWindow; the script's own git metadata subprocess explicitly uses CREATE_NO_WINDOW where available. This worker did not create a GPU/background inference job.

### Acceptance status at handoff

| Rung | Actual status and evidence |
|---|---|
| F0 spine | PASS: plan CLI exit0, actual plan.json with prepared/pending state |
| F1 projection | PASS engineering: full977/2364 projection and validate exit0; exact whitelist fixture rejects unknown/privileged fields |
| F2 raw contracts | PASS engineering fixtures: nativeN, batch ownership, bbox/missing/nonfinite/low scores, L/D mismatch, missing/duplicate PTS, all-slot failure |
| F3 extract/resume/validate_raw | PARTIAL: implemented, synthetic frontend/decoder fixture and actual artifact resume tests pass; real model forward/device integration not run |
| F4 population/review | Population PASS; root's fresh Sol review in progress, no Type-B acquittal claimed |
| Complete pose cache / learner input | NOT COMPLETE: no extraction launched; old320 bridge deliberately absent |

No live production stubs. Unknown checkpoint-specific training recipe remains a disclosure limitation. Complete pose validity, GPU runtime, full decoder population, source-disjoint independent evaluation and learner bridge remain separate requirements. Semantic decisions A001/A003/A004/A005/A007/A008/A010 retain their full ledger statements above; none is represented as a scientific-quality PASS. This bounded implementation does not change loss, model, target generation, benchmark labels, or the frozen training feature schema.

## Narrow independent review corrections (before dependent fix)

Fresh Sol/xhigh review (owned by root) found two concrete blocking resume defects, not an acquittal: completed reuse preceded current media SHA validation; failed artifacts enforced completed-style binding and could not retry with a repaired batch size. A011 (semantic, Source=sweep): validate current source path/SHA before any completed reuse; strict binding applies only to completed reuse, while failed attempts retain their existing attempt JSON and may retry under a new binding. Only these fixes and regressions are authorized in this correction.


Narrow review fix verification: final scoped suite **12 passed in 5.06s**, Ruff check **All checks passed**. Regressions explicitly prove current media changes cannot reuse or overwrite a completed cache; a failed batch2 attempt can retry at batch1 and old attempt JSON bytes remain intact. Source SHA256 `d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1`; test SHA256 `05f5a8628df3d42d53d02d4632a5ba3f9d33389782359bc3a19b9d3b11b523e8`. Root was sent these exact SHAs for fresh Sol narrow recheck. No post-review correctness/acquittal or inference success is claimed by this worker. Completed artifacts whose effective configuration changes remain non-reusable; failed attempts can retry under the current binding.

Local full-population receipts are `E:\ICASSP_TempoRAC\aris-work\tmp\native_pose_full_projection.txt` and `native_pose_projection_totals.txt`; they include both real commands' success, full population totals and three artifact hashes. Small JSON receipts are copied locally as `native_pose_projection_validation.json` and `native_pose_projection_provenance.json`; raw bbox projection stays on the designated server, outside Git. The staging projection generator predates the narrow extraction-only fixes; no projection semantics changed, and its own recorded script SHA remains the truthful generator identity.
