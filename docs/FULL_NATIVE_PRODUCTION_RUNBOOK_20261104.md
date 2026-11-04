# Full native production handoff — 2026-11-04

Prepared commands only; none executed by this task. Reuse completed decode. The 12:21 UTC receipt has fullraw at 300/977, child PID1567156/wrapper1567147 on GPU1. A later executor must obtain actual terminal/completion evidence before running this sequence. Current baseline is ae00d2f7dd7b7cbf347b29639757108408d4928e; a later runtime-only checkpoint may advance HEAD. Record that exact new HEAD and confirm the source hashes below remain identical. This document does not certify the pending six-file training runtime, a full build, target quality, official metrics or science completion.

## Acceptance contract before execution

| Requirement | Required evidence / gate | Preparation status |
|---|---|---|
| Preserve healthy extractor | Do not restart, signal or modify extractor/raw originals; extractor SHA d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1 | Bound in commands/source receipt |
| Full input population | train794/1878 identities + val183/486, total977/2364; 695625 video frames,1758034 person frames; no caps, partial build or subset success | Existing census identified; actual fullraw pending |
| Reuse decode | final977/977,695625 frames,zero findings; exact existing CSV/JSONL/status bindings | Completion receipt already exists |
| Actual PTS overlay | Full977 immutable manifest, no rejected video; preserve missing/nonmonotone actual PTS, never substitute best-effort ticks | Existing checked CLI identified; production pending |
| Qualified NB1 | Full2364 ledger and validation; no artifact failures; retain partial/unusable response states, clocks, reasons and support distributions | Existing checked CLI identified; production pending |
| NT1 | Full2364 target ledger/validation and label-free input census; rejected certificates remain rows; hybrid unavailable | Existing checked CLI identified; production pending |
| Isolated diagnosis | Complete train/dev identities; raw count/unit/event/support records and both full/conditional summaries; no test labels or feedback to producer | Existing checked evaluator identified; field names evidenced; endpoint policy unresolved |
| Scientific follow-through | Once-only evidence-grounded continue-or-semantic-fix decision; then NTR1 plan/ledger/seeds/budget/selection/metric/resource lease freeze | Pending; no training command authorized by this document |

## Existing inputs and proposed new outputs

All paths below refer to the already verified server project root R=/mnt/4tssd/lilei/TempoRAC, with source checkout CODE=$R/repo. New paths are proposals to create later, not existence claims. Never place outputs beneath input roots.

| Role | Exact path relative to R | Provenance |
|---|---|---|
| Existing raw/projection root | native_pose_v1 | Existing extraction output |
| Complete bbox-only census | native_pose_v1/projection.jsonl | SHA b53569b09c643ecdb041df8d4109fddf7d1bdeadfb2163a045f301b68b9761da |
| Original annotation-to-slot provenance | native_pose_v1/projection_provenance.json | SHA effc4bbb11812217e4bb83128c765b68f499e5fca7a0e1eab9ec618ef99ff970 |
| Existing extractor configuration | config/native_pose_v1.json | Exact saved nonsecret config supplies source_root/media_manifest/snapshot; do not replace |
| Existing decode root | runs/media-decode-20261104 | status SHA5556f2e5a693f02adb81256ea600cc6d07d1aed1f111f98214752f6d4c88317b; decode.jsonl SHA f7a97582f4c946c94b382d4a853fc6053791bd187b025d48257b3d8aefb02782 |
| Frozen media inventory | runs/media-integrity-20261104/media.jsonl | SHA8ac8ec586326a3868b6cc2710cddf8ed24f9e9fa85eb0969031567492924576b |
| Existing conservative source graph | runs/canonical-sources-20261104/full/source_components.json | SHA231a83494a2d4004450a1443e8185f9149952e60d25b33c9cf7660991c8c3ee2; path from source inventory and saved close.sh |
| New execution work/log/config root | work/full-native-production-20261104 | Keep raw logs/configs/exits here |
| New immutable overlay root | native_clock_overlay_v1_20261104 | overlay_manifest.json + validation.json + per-video JSON/NPZ |
| New qualified NB1 root | qualified_native_bridge_v1_20261104 | identities/<split>/<media_key>_<slot>.npz/.derived.npz/.json; population_ledger.json/validation.json |
| New NT1 root | native_targets_nt1_20261104 | identities/<split>/<media_key>_<slot>.npz/.json; population_ledger.json/validation.json/input_diagnostics.json |
| New diagnosis root | runs/native-target-diagnostics-full-20261104-unresolved | Must not exist before evaluator; identity_rows.jsonl/diagnostics.json/receipt.json |

The annotation root is the existing config's source_root, saved as /mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/MultiCounterPlus1. It is read-only input to the later isolated evaluator, never an output location. No annotation was opened for this runbook. Source graph membership contains test name metadata; this does not authorize test annotation access. The graph is conservative byte/token evidence, not certified recording independence.

## Execution environment and source freeze

Run the following only after the existing fullraw job has terminal evidence, all977 raw records exist, and the parent confirms its recovery/allocation receipt. Do not equate an unreachable host or absent wrapper with job failure. Do not launch another extractor or decode. The existing wrapper owns FastSAM recovery; check saved PID/start/CWD identity before any recovery action. STaR-Body and other jobs stay untouched.

Use the existing environment from the allocation/review receipts, without shared installs. No environment readiness claim is made here; this is a reuse plan. Required NumPy/SciPy/Torch versions were recorded in the NT1 review; the checked pure helper path below was already used. If unavailable, preserve the exact failure and resolve the environment ledger rather than installing globally. No new GPU witness is needed for these CPU-only production steps.

```bash
set -euo pipefail
R=/mnt/4tssd/lilei/TempoRAC
CODE="$R/repo"
PY=/home/lilei/PoseRAC_Experimental_Code_20260828/.venv/bin/python
W="$R/work/full-native-production-20261104"
export R CODE PY W
cd "$CODE"
test -x "$PY"
test ! -e "$W"
mkdir -p "$W/config" "$W/logs" "$W/tmp" "$W/cache"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TMPDIR="$W/tmp" TMP="$W/tmp" TEMP="$W/tmp"
export XDG_CACHE_HOME="$W/cache/xdg" HF_HOME="$W/cache/hf" TORCH_HOME="$W/cache/torch"
export MPLCONFIGDIR="$W/cache/mpl" NUMBA_CACHE_DIR="$W/cache/numba"
export PYTHONPATH="$CODE/src:$R/work/native-target-implementation/vendor_pure"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
run_step() {
  local name="$1" rc=0
  shift
  if "$@" > "$W/logs/$name.log" 2>&1; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" > "$W/logs/$name.exit"
  return "$rc"
}
git rev-parse HEAD > "$W/source_head.txt"
git status --porcelain > "$W/source_status.txt"
```

Inspect source_status.txt before proceeding: a later intentional checkpoint is acceptable with an exact receipt; unexplained modifications are not. Do not reset/revert another worker. Verify required checked source hashes (small source files only; no MP4 rehash):

```bash
sha256sum -c <<'SHA'
d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1  scripts/build_temporac_native_pose_cache.py
6026a51e4eebdb546fabdd23d61e2afe8721dca3c5a03bed7332862f54bdccf5  src/pams/temporac/native_clock_overlay_v1.py
2cf9f164619df63f5d6ee85bda5e0c405870c2ab8dd68ee10dbbd1572c3be398  scripts/build_temporac_native_clock_overlay.py
219c9077584a5923c65d6f96ab44ceef534218cb5519dc94461a545b5bec9a39  scripts/build_temporac_qualified_native_bridge.py
50aecf64e85667570b656553789e255763b755d3f252d82b3c21f7585071c4f3  src/pams/temporac/native_bridge_v1.py
1822d120a2d1f6e00fb031dae7e535b75cde28a9b99fd18fd0afea1b5460806a  scripts/build_temporac_native_learner_bridge.py
a1c84ef76830bddeaf558579f7d8c02c592b8349c1eeef44d72f6461fa251390  src/pams/temporac/native_target_nt1.py
95b8686caa725d3c220e68fb74e13dab7b822f41fe3d3dc1e12c10155ab3fee5  src/pams/temporac/native_target_diagnostics_v1.py
00e320893718d1d8bdf24372d6685fcc92ae770c82351a3b536f92d50676de15  scripts/build_temporac_native_targets.py
add1421442fcc5e9ca69cb289f061a8b2f3052f7760b39965d21eb6b0e51a913  scripts/evaluate_temporac_native_targets.py
SHA
```

The builder SHA starting1822 identifies a read-only input helper, not the legacy builder to run for production. Exact hashes for all inspected entry points/modules are in the private runbook read receipt. A source mismatch stops this handoff; it does not authorize editing the active extractor.

Generate only the three label-free configs. Each CLI enforces exact keys and full production census; there are no caps or debug flags.

```bash
"$PY" -B - <<'PY'
import json, os
from pathlib import Path
r,w=Path(os.environ['R']),Path(os.environ['W'])
raw=r/'native_pose_v1'; census=raw/'projection.jsonl'
configs={
 'overlay.json':dict(raw_root=str(raw),census=str(census),decode_root=str(r/'runs/media-decode-20261104'),media_inventory=str(r/'runs/media-integrity-20261104/media.jsonl'),output=str(r/'native_clock_overlay_v1_20261104')),
 'nb1.json':dict(raw_root=str(raw),census=str(census),overlay_manifest=str(r/'native_clock_overlay_v1_20261104/overlay_manifest.json'),output=str(r/'qualified_native_bridge_v1_20261104')),
 'nt1.json':dict(bridge_root=str(r/'qualified_native_bridge_v1_20261104'),census=str(census),output=str(r/'native_targets_nt1_20261104'),hybrid_mode='unavailable')}
for name,cfg in configs.items():
    with (w/'config'/name).open('x') as f: json.dump(cfg,f,sort_keys=True,indent=2)
PY
```

## Ordered full production sequence

Execute blocks separately and inspect their receipts before the next block. Every command redirects to its own log; retain actual exit codes. A shell command finishing successfully is an engineering completion gate, not quality acceptance. Do not auto-advance on a stale file from a previous attempt; use a new output root if binding/content changes.

1. Fullraw validator, not project/validate/extract. validate_raw loads every raw JSON/NPZ, verifies payload hashes/schema/projection and writes raw_validation.json; it does not rerun inference or open GT.

```bash
run_step raw_validate "$PY" -B scripts/build_temporac_native_pose_cache.py validate_raw --config "$R/config/native_pose_v1.json"
```

Require977 videos/2364 persons and all977 status complete, declared/decoded lengths agreeing; retain the raw bridge-support diagnostics. An exit0 alone does not assert raw status complete: the CLI returns recorded statuses, so inspect all rows. Preserve failures and stop advancement if any raw is unfinished/failed or mismatched. Archive the existing population ledger and new raw_validation receipt by hash. Do not rewrite pose arrays or original metadata.

2. Freeze complete actual-PTS overlay, then validate the immutable manifest:

```bash
run_step overlay_build "$PY" -B scripts/build_temporac_native_clock_overlay.py build --config "$W/config/overlay.json"
run_step overlay_validate "$PY" -B scripts/build_temporac_native_clock_overlay.py validate --config "$W/config/overlay.json"
```

Require overlay_manifest status frozen_complete_census, videos977/identities2364, validation status validated_population and no rejected selection. Read selection counts including raw-retained/overlay-attached/unavailable, not just attached count. The CLI binds exact decode row bytes, CSV SHA, decoder source/status, frozen media inventory, raw JSON/NPZ and implementation sources. It preserves actual PTS defects and raw bytes; five measured rational clocks are not replaced by global FPS. Prior media SHA plus stable stat is the disclosed assurance, not fresh media rehash or pixel equivalence.

3. Build only the qualified NB1 entry point, then validate:

```bash
run_step nb1_build "$PY" -B scripts/build_temporac_qualified_native_bridge.py build --config "$W/config/nb1.json"
run_step nb1_validate "$PY" -B scripts/build_temporac_qualified_native_bridge.py validate --config "$W/config/nb1.json"
```

Require both population_ledger.json and validation.json status validated_population,977/2364, empty failures, full key/slot census. Clock-qualified2364 does not mean2364 executable/certified identities. Keep all status, clock-mode/reason counts, source/observed/geometry/response edges, runs and known seconds. NB1 binding must name the frozen overlay manifest SHA and census SHA; old fallback NB1 artifacts cannot resume as qualified artifacts.

4. Full NT1 build, validation and label-free input census:

```bash
run_step nt1_build "$PY" -B scripts/build_temporac_native_targets.py build --config "$W/config/nt1.json"
run_step nt1_validate "$PY" -B scripts/build_temporac_native_targets.py validate --config "$W/config/nt1.json"
run_step nt1_inputs "$PY" -B scripts/build_temporac_native_targets.py diagnose-inputs --config "$W/config/nt1.json"
```

Require population_ledger.json/validation.json complete_population,977/2364,empty failures,natural_GT_accessed=false; input_diagnostics.json diagnosed_inputs,977/2364. Certificate rejection is a legitimate zero/abstaining row, not an artifact error or reason to discard identities. NT1 policy binds exact NB1 feature, raw/bridge binding, producer dependency hashes, config and hybrid availability. Preserve all attempts/branches/traversal bounds/support funnels/training-ineligible reasons/direct pseudo-event counts. Join by original complete census and source components; no sampling or dropping short/unsupported identities. NT1 does not establish a traversal equals one true repetition.

## Isolated full train/dev target diagnosis

The original per-object keys are count and period. This is established without a new GT read by the saved duplicate_fields_0943.sh parser (a.get('count') and a.get('period')) and duplicate_annotations_20261104.json output (non-null counts and count/period disagreements). The parser was inspected only, never executed. Field schema is resolved; endpoint semantics remain unresolved. Dynamic NB1/NT1 SHA values must come from the completed production files, never placeholders or this preparation date.

Endpoint policy remains unresolved. With endpoint_policy=unresolved the evaluator can report full/conditional counts, raw periods and support, but unit/event-in-period coverage is explicitly unavailable. e_minus_s_v1 is a proposed diagnostic convention only: raw [s,e) source-edge overlap divided by e-s, pulse positions s<=p<e,0<=s<e<=N. It is not official MultiRep semantics. A complete unit/event diagnostic requires explicitly freezing that proposed policy (with limitations) or resolving authoritative endpoints; do not silently call unresolved output full unit acceptance. Do not try both and select whichever looks better.

After the endpoint decision is recorded, generate frozen_diagnostics.json under W/config with the exact schema below. Set DIAGNOSTIC_ENDPOINT_POLICY explicitly to unresolved for a count/support-only result, or to e_minus_s_v1 only after the proposed convention has been frozen with its limitations. The output name records this choice. There is no implicit policy default or selection based on results. Bootstrap0 is the fixed descriptive first diagnosis; it does not imply statistical acceptance.

```bash
: "${DIAGNOSTIC_ENDPOINT_POLICY:?record and set explicit endpoint decision}"
export DIAGNOSTIC_ENDPOINT_POLICY
"$PY" -B - <<'PY'
import hashlib, json, os
from pathlib import Path
r,w=Path(os.environ['R']),Path(os.environ['W'])
policy=os.environ['DIAGNOSTIC_ENDPOINT_POLICY']
if policy not in ('unresolved','e_minus_s_v1'): raise ValueError('unfrozen endpoint policy')
def item(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
inputs={
 'census':item(r/'native_pose_v1/projection.jsonl'),
 'provenance':item(r/'native_pose_v1/projection_provenance.json'),
 'source_components':item(r/'runs/canonical-sources-20261104/full/source_components.json'),
 'nb1_manifest':item(r/'qualified_native_bridge_v1_20261104/validation.json'),
 'nt1_manifest':item(r/'native_targets_nt1_20261104/population_ledger.json')}
source=json.loads((r/'config/native_pose_v1.json').read_text())['source_root']
cfg=dict(protocol='temporac.native-target-diagnostics.v1',inputs=inputs,
 annotation_root=source,bridge_root=str(r/'qualified_native_bridge_v1_20261104'),
 target_root=str(r/'native_targets_nt1_20261104'),
 output=str(r/('runs/native-target-diagnostics-full-20261104-'+policy)),
 count_field='count',periods_field='period',endpoint_policy=policy,bootstrap_replicates=0)
with (w/'config/frozen_diagnostics.json').open('x') as f: json.dump(cfg,f,sort_keys=True,indent=2)
PY
```
Run as a separate CPU process only after the four label-free blocks have exited. No producer/trainer runs in this process. Use the environment above; CPUthreads1/CUDA empty/offline/cache roots fixed. Provide only complete immutable NB1/NT1/census/provenance/source graph plus the original train/val annotation root. The checked evaluator validates every complete artifact before opening any label; annotation paths are derived solely from train/val census, constrained inside that split, checked against original SHA and complete original-object-to-slot mapping. These are implemented dataflow/read-boundary protections; this command does not claim an OS sandbox or inaccessible GT filesystem.

```bash
run_step diagnostics "$PY" -B scripts/evaluate_temporac_native_targets.py --config "$W/config/frozen_diagnostics.json"
```

Require identity_rows.jsonl exactly2364 identities (train1878/dev486), retaining all794/183 videos and four exact-media train alias groups. Report original object/count/periods, count-period disagreements (including existing58/48 conflicts), source component/branch/clock/reasons, full/conditional video→person macro proposed NMAE/raw MAE/OBO/bias/ratio, frame/edge/known-duration support and unknown-time edges. Under a frozen valid proposed endpoint convention also report all GT-period denominators,0/1/>=2 pulse bins, outside-period pulses and full/partial/unsupported events. Do not scale partial counts; failed prediction contributes count0 plus abstain. bootstrap_replicates0 makes no CI/significance claim; freeze any later bootstrap plan separately (implementation seed20261104,source-component clusters). PeriodAP is unavailable without a legal period decoder; official_metric_status remains not_certified and target_quality_acceptance remains not_evaluated_no_tolerance_frozen.

Do not use natural GT to tune thresholds, choose target branch, filter population, select checkpoints or alter sampler. Preserve X0 NMAE.9915197/OBO0, v15/old80% coverage failures and other negatives. Once the complete diagnosis exists, record one evidence-grounded continue-or-semantic-fix decision, distinguishing semantic errors from merely poor empirical quality. Quality tolerance, official count parity, release/exposure independence, strong comparisons and scientific acceptance are outstanding. Full NTR1 canonical/control seed20260815 then20260816/17 requires its separate full population/paired ledger/budget/selection/metric/live lease freeze; this runbook adds no training launch.

## Receipts, failure handling and scope

Save command/config/source HEAD+hashes, immutable input/output manifest hashes, original process/allocation completion evidence, stdout/stderr and actual exit for each block. The run_step wrapper records both zero and nonzero exits before returning; set-e stops the sequence on a nonzero return. Stop on a nonzero exit, retain artifacts/logs, and resume only exact matching bindings; changed content requires a new output root. Do not re-run completed decode or silently overwrite prior negative evidence. Archive this complete evidence before any eventual cleanup; no cleanup is authorized now.

Checked interfaces: scripts/build_temporac_native_pose_cache.py main/validate_raw; scripts/build_temporac_native_clock_overlay.py config/execute/main; scripts/build_temporac_qualified_native_bridge.py execute/main; scripts/build_temporac_native_targets.py execute/main; scripts/evaluate_temporac_native_targets.py and native_target_diagnostics_v1.py run/unit_diagnostics/evaluate_rows. Existing contracts, full-decode completion/source inventory and nonsecret config/helper receipts establish paths. CODE is the source checkout under R/repo: cd, relative scripts, source SHA checks and git commands all resolve there; R remains the data/work/output root. The six training runtime files were not reviewed or changed. No build/test/MP4/GT/evaluator/GPU/server-job/commit/push/state operation occurred in preparation.
