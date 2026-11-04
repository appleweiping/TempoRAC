# Independent native clock overlay / qualified NB1 review — 2026-11-04

**Substantive verdict: no concrete engineering blocker found in the reviewed five-file scope.** The result is OpenAI **same-family provisional** semantic assurance. It does not establish production coverage, target quality, training readiness, scientific merit, or paper acceptance.

## Identity, scope and primary-source receipt

- Fresh native reviewer session: `/root/native_clock_overlay_review`; parent `/root`.
- Requested reviewer/model effort: `gpt-6.1-sol / xhigh`. Actual backend identity and effort: not independently attested. Requested identity is not substituted for actual evidence.
- Executor/reviewer relation: same-family; acceptance status `provisional`. No cross-family approval is claimed.
- Primary clock decision read in full before implementation access: `docs/NATIVE_CLOCK_OVERLAY_DECISION_20261104.md`, independently reproduced SHA256 `955b056834e0c32f288129c91d0d5380740cb6d74de8180b5bab9fdd64417b45`.
- NB1 contract read in full: `docs/NATIVE_LEARNER_BRIDGE_CONTRACT_20261104.md`, SHA256 `f0622e664ed6ed53f123cfc7f7151a71f04b2d15f347e84114e548ddbaa9487a`; existing checked NB1 review, including its final three-finding recheck, was read in full. That prior result was not reclassified as fresh review here.
- Full prerequisite reads: `E:/AGENTS.md`, project `AGENTS.md`, `ARIS_LOCK.md`, `docs/aris/WORKFLOW.md`, `docs/CVPR_RESUME_20261104.md`, current `.aris/runs/temporac-cvpr-20261104.json`.
- Full pinned `research-review/SKILL.md` and required references were read from exact pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`; independent source/copy hashes were identical. References: reviewer-backends, reviewer-routing, review-tracing, integration-contract, output-composition, output-versioning, output-manifest, output-language; reviewer-independence was additionally read. The skill SHA is `17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5`.
- No new implementation file was opened before the parent's STABLE dispatch. All four Python hashes were independently reproduced before full reading and again after verification. The documentation was allowed to receive actual receipts during review; its final reviewed hash is recorded below.

Full reads covered the four new Python files and final documentation. Targeted existing-source reads covered raw validation/load/projection, extractor media hashing/default OpenCV/native-row path, complete decode audit, checked NB1 builder helpers, NativeFeatureV1 and native_clock, and the actual NT1 builder consumption path. This was not a generic redo of NB1/NT1 scientific or numerical reviews.

| Reviewed file | Final independently reproduced SHA256 |
|---|---|
| src/pams/temporac/native_clock_overlay_v1.py | 6026a51e4eebdb546fabdd23d61e2afe8721dca3c5a03bed7332862f54bdccf5 |
| scripts/build_temporac_native_clock_overlay.py | 2cf9f164619df63f5d6ee85bda5e0c405870c2ab8dd68ee10dbbd1572c3be398 |
| scripts/build_temporac_qualified_native_bridge.py | 219c9077584a5923c65d6f96ab44ceef534218cb5519dc94461a545b5bec9a39 |
| tests/temporac/test_native_clock_overlay_v1.py | dfbd3c97638dc5e11c4665497726727673ead139635d56442293898b16f5083f |
| docs/NATIVE_CLOCK_OVERLAY_20261104.md | 7cae0469567be61f13b315acbfd98b37db9ed83c45a591de93e245515e23edc9 |

## Checklist prepared before implementation review

The primary checklist was delivered to the parent while waiting for STABLE. Final dispositions follow; “pass” below is an engineering result within the stated fixture/source scope.

| Requirement | Final status and evidence |
|---|---|
| Exact actual int64 PTS, positive rational time base, original-present PTS consistency/no-op | Pass: integer grammar/range checks; no float PTS parse; equivalent rational units accepted; raw-present rows cannot disappear or change. Partial raw clocks backfill from the whole CSV view. Complete matching raw clocks retain their original bytes, including unreduced equivalent rational units. Large ticks are preserved and NB1 performs integer subtraction first. |
| Bind raw/row/CSV/media/N/L/D/index/dimensions/stream/order/read set | Pass for the frozen source contract: raw NPZ digest, complete raw status and L=D=N, every native index, selected absolute stream and per-frame dimensions, decoded count/parse/status, inventory/projection membership/SHA/size, old stream if present, and raw/CSV read-set stability are checked. Exact JSONL row and CSV hashes are calculated in this freeze. Binding errors reject rather than become source-frame-qualified fallback. |
| Honest media-SHA limitation | Pass: raw extractor actually hashes each media before extraction; decode `prior_sha256` is copied from inventory, with before/after stat and size checks. Overlay retains that restriction explicitly; no decode-time fresh rehash or pixel-equivalence claim. |
| Missing/duplicate/reversed PTS remain and break NB1 support | Pass: six authored clock classes plus independent all-duplicate/all-reversed checks. Present invalid clocks remain `pts` with `clock_invalid` and no response support; they do not downgrade to source-frame mode. Only entirely absent true PTS permits explicit `source_frame_only`. |
| No best-effort/FPS invention | Pass: only CSV `pts` is read into the clock; best-effort fields and decode `physical_clock` are not used. Independent CSV containing only best-effort without a `pts` column rejects. |
| Raw, nonclock pose/masks/identity bytes stay unchanged | Pass: authored byte comparisons cover every nonclock NativeFeatureV1 array and both raw files, storage/idempotence/no-op. Source only replaces `pts`, `pts_present`, `time_base`; unchanged old six files and extractor independently verified below. |
| Freeze only after complete population; retain all videos/slots/failures | Pass engineering: production defaults fix 977 videos/2364 identities plus full native/person-frame totals; no CLI cap/population override. Missing/in-progress decode is pending; unfinished raw preflight freezes no overlays. Terminal failed raw is a retained rejected row. Synthetic rejected/corrupt-artifact ledgers preserve all four slots. Actual full population remains pending. |
| Resolve clock before NB1 preprocessing and output exact consumer artifacts | Pass source and actual consumer fixture: qualified builder calls resolve before preprocess and retains checked feature/derived/sidecar layout and schema. Actual checked NT1 `diagnose-inputs` consumed exact generated two-video/four-identity artifacts, as documented below. This does not validate target production or true repetition units. |
| Atomic artifacts/resume bind overlay and invalidate old fallback | Pass: existing checked NB1 atomic helpers, full manifest/source/content bindings and new build ID. Old fallback sidecar or changed manifest cannot resume; differing immutable overlay rejects. Independent truncated resumed feature NPZ is caught before parsing and all four ledger slots remain. |
| No training/model/GT/GPU/healthy-job effects | Pass within actual execution: CLIs expose plan/build/validate only; local tests synthetic CPU. One separately authorized server scratch interoperability probe used existing CPU dependencies, empty CUDA visibility, one thread, no Torch import and no production data. No main-repository/server-job/shared-environment mutation. |

## Stream selection and presentation-order evidence

Raw OpenCV metadata does **not** itself record the selected stream index. Decode metadata invokes `-select_streams v:0` before checking that one selected stream is returned; that check is not proof the entire container has only one video stream.

The accepted alignment here instead follows the exact default decoder source path: frozen extractor opens `cv2.VideoCapture(path)`, repeatedly calls `read`, performs no seeking/stream/raw-mode override, and records OpenCV4.11.0/FFMPEG. The versioned [official OpenCV4.11.0 source](https://github.com/opencv/opencv/blob/4.11.0/modules/videoio/src/cap_ffmpeg_impl.hpp) initializes `video_stream=-1` and `rawMode=false`, iterates stream indices in the default decoded-frame path, initializes the first video stream, assigns its index, and breaks. Its separate raw-packet path uses best-stream selection; that path is not enabled by the frozen extractor.

[Official FFmpeg n6.1 stream-specifier documentation](https://github.com/FFmpeg/FFmpeg/blob/n6.1/doc/fftools-common-opts.texi#L30-L45) defines video type filtering and matching-stream numbering in libavformat order. I infer from these two primary sources and the actual extractor call path that default OpenCV first-video selection matches audit `v:0`; this inference is not based on matching frame counts or an invented unique-stream requirement. Every CSV row still must match audit's selected absolute stream index and dimensions. Different recorded versions/backends and conflicts reject.

The complete author-retained OpenCV source at `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/opencv411_cap_ffmpeg_impl.hpp` was independently hashed to `f10f6987757d3da421e31cd529f99a4cd2bb955f771e116855644a61cc365354`; relevant selection, initialization and sequential decoding code was read, and the official versioned source was separately opened. This establishes a source-level selection/presentation contract, not framewise pixel equality across decoders or fresh media-byte equivalence during historical decode. No actual stream ambiguity was found in the supplied source/provenance requiring a new proof framework or mass rehash.

## Independently executed local verification

Environment: existing `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe`, NumPy/pytest and Ruff. TEMP/TMP were `E:/ICASSP_TempoRAC/aris-work/tmp`, PYTHONPATH was repo/src, PYTHONDONTWRITEBYTECODE=1. Python `-B`, pytest `-p no:cacheprovider`, E-only basetemp; Ruff `--no-cache`. No installation.

```text
python -B -m pytest -p no:cacheprovider tests/temporac/test_native_clock_overlay_v1.py tests/temporac/test_native_bridge_v1.py tests/temporac/test_native_time_cue_v1.py --basetemp E:/ICASSP_TempoRAC/aris-work/tmp/clock-review-baseline -q
```

Actual raw completion output, exit0:

```text
.............................................................            [100%]
61 passed in 46.30s
```

Independent Ruff on the four scoped Python files, exit0:

```text
All checks passed!
```

Additional independent reviewer-owned fixture probe: `E:/ICASSP_TempoRAC/aris-work/tmp/native_clock_overlay_review_extra.py`; only its own synthetic outputs were mutated. Final raw output, exit0:

```text
all_duplicate pts ('clock_invalid',) unusable
all_reversed pts ('clock_invalid',) unusable
best_effort_without_pts_column rejected clock CSV missing or duplicate columns
actual NT1 diagnose-inputs unavailable: preexisting local scipy dependency absent; no install/stub/server call
qualified NB1 feature-only loader and PTS preprocessing roundtrip consumed all four slots
truncated resumed qualified NPZ retained all slots 4 {'artifact_failure': 2, 'complete': 2}
failed raw retained full manifest {'overlay-attached': 1, 'rejected': 1}
unfinished raw preflight froze no per-video overlays
```

The first local actual NT1 import attempt ended with `ModuleNotFoundError: No module named 'scipy'` via `native_target_nt1 -> certify`. This was an existing audit-environment dependency limit, not an implementation finding. No mock/stub or install was used to manufacture a consumption pass. The parent then explicitly expanded only the necessary server CPU fixture boundary; the successful actual consumer check below supersedes the executable-compatibility limitation.

## Narrow actual NT1 interoperability probe

Authorization was limited to the reviewer's own `/mnt/4tssd/lilei/TempoRAC/work/native-clock-review` scratch, existing PoseRAC CPU dependencies, synthetic qualified products, and checked NT1 `execute(...,'diagnose-inputs',expected=fixture)`. No 18/31-test server rerun, formal/full build, model, natural GT, GPU, main-repository edit, shared-environment mutation, job inspection/signal or training occurred.

The reviewer generated fresh synthetic qualified outputs locally using the stable overlay and qualified builder. Archive `E:/ICASSP_TempoRAC/aris-work/tmp/clock-review-interop/interop.tar`, SHA256 `3c25b99e3ac11b7857cea135767cbb496fe9ff32acd198907ab40a4454a002de`, contains the two-video/four-person census, exact generated NB1 feature/derived/sidecar products and all four stable new Python sources. The existing server source was copied read-only into the reviewer's own snapshot and the four new stable files copied there. Ten source SHA values were asserted before consumer execution, including unchanged NB1/cue/extractor, NT1 module `a1c84ef76830bddeaf558579f7d8c02c592b8349c1eeef44d72f6461fa251390` and checked NT1 CLI `00e320893718d1d8bdf24372d6685fcc92ae770c82351a3b536f92d50676de15`.

Actual result: checked NT1 `diagnose-inputs` returned0, `status=diagnosed_inputs`, videos2, identities4. Each exact qualified feature artifact was additionally loaded and preprocessed as PTS with all40 timestamps present. Python3.10.12, NumPy1.24.4, SciPy1.15.3; installed Torch2.0.1+cu118 was **not imported**. CUDA_VISIBLE_DEVICES was empty; OMP/BLAS/MKL/NUMEXPR threads1; all cache/TMP paths were under own scratch. This proves fixture artifact consumption, not numerical target quality or full population readiness.

Raw final summary:

```text
qualified artifacts consumed by checked NT1: 2 videos / 4 identities; no model, natural GT, GPU or production build
REMOTE_EXIT 0
```

Reproducible E-side scripts: `tmp/native_clock_review_prepare_interop.py`, `tmp/native-clock-review-prepare.sh`, `tmp/native-clock-review-consume.sh`. Structured actual-environment/source result downloaded as `E:/ICASSP_TempoRAC/aris-work/tmp/native-clock-review-interop-receipt.json`, SHA256 `e3bd77f7c64030f29f2bb32efcf122ce78c263ed4a94855f94170aa8e83138e2`. Actual diagnostic SHA256 `a852c7979f7e9a1037053a35049c11ec90f0d6f0442915dc3443d983346dd64e`.

## Frozen existing files and final disposition

The following hashes matched the prior checked NB1 final snapshot both before and after this review:

| Existing frozen file | SHA256 |
|---|---|
| src/pams/temporac/native_bridge_v1.py | 50aecf64e85667570b656553789e255763b755d3f252d82b3c21f7585071c4f3 |
| src/pams/temporac/native_time_cue_v1.py | 97d4c537813757ca03f83e3c695abf63418a9465fcba481ec1d21b159252a880 |
| scripts/build_temporac_native_learner_bridge.py | 1822d120a2d1f6e00fb031dae7e535b75cde28a9b99fd18fd0afea1b5460806a |
| tests/temporac/test_native_bridge_v1.py | 1ed1647397279792b52eb3b24db301b38f20821e71e2bb980f5025f3f54a4e8d |
| tests/temporac/test_native_time_cue_v1.py | d4d9750fb1fb33f181dbb471f5bea9cc8df6cbab03e66dfe1c380b5d53bcaf16 |
| docs/NATIVE_LEARNER_BRIDGE_20261104.md | 25e998690a517d9b76aee97f56519650fd6062c5fe88b0927d0249a5e49eb1c2 |
| scripts/build_temporac_native_pose_cache.py | d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1 |

`git diff --name-only` and tracked-only `git status --short --untracked-files=no` remained empty. The reviewer made no implementation fix: only this report and reviewer-owned synthetic probes/outputs were written, including the separately authorized isolated server scratch. No commit/push or cleanup occurred. Parent owns shared trace/meta/manifest integration; this is the requested raw reviewer response and evidence.

Review completed: **true**. Review scope complete: **true for the five supplied files and bounded qualified-NB1/NT1-consumption interface**. Substantive engineering verdict: **no concrete blocker found**. Acceptance: **same-family provisional**, requested Sol/xhigh, actual backend/effort unattested.

Actual full977 overlay freeze and full2364 qualified build/validate/census, real physical-time/response coverage, NT1 target construction/quality and true-unit diagnostics, metric/failure protocol, complete training and scientific acceptance remain **pending/separately gated**. Fixture success does not satisfy those obligations and does not authorize training.
