# Native clock overlay implementation — 2026-11-04

## SPEC (recorded before code)

Target: immutable `temporac.native-clock-overlay.v1` and a separate qualified NB1 builder. Inputs: complete raw JSON/NPZ, exact decode JSONL row bytes, actual PTS CSV, complete projection and media inventories. Outputs: independently bound overlay JSON/NPZ, full video manifest, NB1 identity artifacts and full failure ledger. Base commit: `2458eb2906f39dde42f3fcd3119b2acefa11bcc3`. Success command: E-only Python 3.11 synthetic CPU pytest for the new test module, followed by Ruff. No actual partial/full cache build, training, labels, models, GPUs, process changes, commit or push in this implementation task. Existing six NB1 files and extractor remain byte frozen.

## ASSUMPTIONS (ASK=never; pre-code)

| ID | Gap | Choice | Class | Source |
|---|---|---|---|---|
| A001 | Output organization | Overlay manifest contains all census videos; paths relative to manifest directory; rejected videos retain every identity in qualified ledger | interface | default |
| A002 | Full build gate | Require complete raw/decode population before any overlay artifact freeze; reject bindings independently, freeze failure rows too | semantic | user |
| A003 | Presentation order | Require recorded OpenCV 4.11.0 / FFMPEG and unchanged extractor SHA; default non-raw VideoCapture selects first video stream, aligned with ffprobe v:0. This does not establish a unique video stream or pixel equivalence; reject other versions/backends | semantic | default |
| A004 | PTS defects vs audit findings | Allow only explicit clock missing/nonmonotonic findings; all other decode findings reject. Preserve missing/duplicate/regressing actual PTS | semantic | user |
| A005 | Media SHA assurance | Compare raw SHA, copied decode prior SHA and frozen inventory SHA; disclose prior SHA plus stable-stat limitation; no fresh media rehash claim | semantic | user |
| A006 | Immutability/resume | Manifest and artifacts bind content and source hashes; no overwrite on differing content, reuse only exact reconstructed data and bindings | interface | default |
| A007 | Fixture population | Internal execute(expected=...) exists only for synthetic CPU fixtures; CLI exposes no cap/population override | interface | user |
| A008 | Qualified definition | Clock binding accepted; qualified count is separate from complete/partial/unusable NB1 response status and physical-clock coverage | semantic | default |

## BUILD_NOTE

| Rung | Feature | Check | Tier | Status |
|---|---|---|---|---|
| F0 | CLI plan entry to JSON artifact | synthetic plan test | MUST | passed, exit 0 |
| F1 | strict immutable overlay API + nonclock identity | new overlay fixture tests | MUST | passed, exit 0 |
| F2 | full census freeze + qualified NB1 composition | synthetic complete census build/resume/validate | MUST | passed, exit 0 |
| F3 | corruption/old artifact invalidation | new failure fixture tests + existing impacted CPU tests | MUST | passed, exit 0 |

Pinned research-implement-feature and mandatory shared references read before implementation. Sole-file documentation is the parent-authorized replacement for extra implement-stage files. Source decision: `955b056834e0c32f288129c91d0d5380740cb6d74de8180b5bab9fdd64417b45`; NB1 contract: `f0622e664ed6ed53f123cfc7f7151a71f04b2d15f347e84114e548ddbaa9487a`.

## Run record / deferred / blockers

Pre-code SPEC/ledger/ladder were recorded before source creation. Procedural deviation: F0's first execution happened after F1/F2 code was written, rather than before those rungs. First F0 attempt failed because the E-only pytest parent directory did not exist; created that directory, then F0 passed (1 passed, 26 deselected in 1.44s). This is not evidence of independently green earlier rungs. No live stubs.

CPU evidence, all using `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe`, NumPy and no Torch:

- New initial suite: 27 passed in 18.50s, exit 0, `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/pytest1.txt`.
- Impacted suite: 57 passed in 41.10s (27 overlay + 30 unchanged NB1/cue), exit 0, `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/pytest_final.txt`.
- Stable new suite after source-selection/read-set/no-op fixes and four added focused fixtures: **31 passed in 21.20s**, exit 0, `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/pytest_stable.txt`.
- Ruff check: **All checks passed**, exit 0, `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/ruff_final.txt`. Four owned Python files formatted.
- `TEMP=TMP=E:/ICASSP_TempoRAC/aris-work/tmp`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=E:/ICASSP_TempoRAC/TempoRAC-CVPR/src`, Python `-B`, pytest `-p no:cacheprovider`; final basetemp `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/tests_stable`. No new process launcher or background process was introduced.

Stable command:

```text
E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/temporac/test_native_clock_overlay_v1.py --basetemp E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/tests_stable -q
```

| File | SHA256 |
|---|---|
| `src/pams/temporac/native_clock_overlay_v1.py` | `6026a51e4eebdb546fabdd23d61e2afe8721dca3c5a03bed7332862f54bdccf5` |
| `scripts/build_temporac_native_clock_overlay.py` | `2cf9f164619df63f5d6ee85bda5e0c405870c2ab8dd68ee10dbbd1572c3be398` |
| `scripts/build_temporac_qualified_native_bridge.py` | `219c9077584a5923c65d6f96ab44ceef534218cb5519dc94461a545b5bec9a39` |
| `tests/temporac/test_native_clock_overlay_v1.py` | `dfbd3c97638dc5e11c4665497726727673ead139635d56442293898b16f5083f` |

Pinned/installed research-implement-feature SHA are equal: `33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8`. Shared references read: reviewer-backends, acceptance-gate, reviewer-independence, reviewer-routing, review-scope-limits, effort-contract, output-language, capture-antipatterns and injection-hygiene. Some batched output was initially truncated; standalone rereads completed the affected references before final handoff. No existing tracked files changed (`git diff --name-only` empty); NB1, extractor and the other writer's NT1/LF1 files untouched.

## Stream selection evidence

The [OpenCV 4.11.0 official source](https://github.com/opencv/opencv/blob/4.11.0/modules/videoio/src/cap_ffmpeg_impl.hpp#L1164-L1310) loops stream indices and initializes the first video stream in the default decoded-frame path. The separate raw-packet path at lines 1148–1162 uses best-stream selection; the frozen extractor does not enable that path. Its `opencv_frames` opens `cv2.VideoCapture(path)` then repeatedly calls `read`, with no seeking or stream override. [FFmpeg's official stream-specifier documentation](https://github.com/FFmpeg/FFmpeg/blob/n6.1/doc/fftools-common-opts.texi#L30-L45) makes v:0 the first video stream in libavformat order. This is the documented alignment contract, not an inference from equal frame counts or the selected stream count. CSV rows still must match the actual selected absolute stream index and every frame's dimensions.

The complete primary OpenCV source was saved read-only under `E:/ICASSP_TempoRAC/aris-work/clock-overlay-cpu/opencv411_cap_ffmpeg_impl.hpp`, SHA256 `f10f6987757d3da421e31cd529f99a4cd2bb955f771e116855644a61cc365354`. Overlay JSON records these primary URLs/version/source SHA. Other recorded OpenCV versions/backend, changed extractor source, conflicting original clock/stream, failed decode or ambiguous input are rejected. No media was reprobed or decoded in this task.

## Entry points and artifacts

Both CLIs provide `plan|build|validate --config <json>` only, no production population override/cap and no training launcher.

Overlay config requires exactly `raw_root`, `census`, `decode_root`, `media_inventory` (the frozen media JSONL), and `output`, all on E on Windows. Complete census and final decode status are mandatory; missing/in-progress input only writes `population_pending.json`, without per-video overlays. Full manifest is `output/overlay_manifest.json`, with every census video and all slots, raw-retained/overlay-attached/unavailable/rejected selection counts, exact CSV/JSONL-row byte hashes and source bindings. Actual missing/duplicate/regressing PTS survive unchanged; best-effort timestamps are never used as PTS. Manifest clock fields expose PTS frames, adjacent supported time edges and known seconds; executable NB1 coverage is reported by the next builder.

Qualified NB1 config requires exactly `raw_root`, `census`, `overlay_manifest`, `output`. It imports the checked NB1 census/raw/projection/preprocessing/derived-array/summary/atomic-write helpers and resolves clock before preprocessing. It retains NB1 feature NPZ, `.derived.npz`, `.json` sidecar and `identities/<split>/<key>_<slot>` layout accepted by the NT1 CLI. Binding adds a new build ID, manifest SHA, overlay content and source provenance; old fallback receipts cannot resume. All slots remain in `population_ledger.json`/`validation.json`, including rejected slots; status, clock qualification, clock-mode/reason counts, support distributions, executable runs and known seconds are separate. Rejected video slots never produce fallback-qualified identity artifacts.

Read-set stability is checked over raw JSON/NPZ/CSV construction as well as individual reads. CSV and exact JSONL-row hashes are computed now. Decode prior media SHA remains copied from the frozen inventory plus stable pre/post stat; it is **not** a decode-time fresh rehash or framewise pixel-equivalence claim. Overlay attachment changes only the effective PTS/present/time-base view; raw originals, original clock metadata/hashes, pose bytes, masks and source-frame indices remain preserved. Complete matching raw clocks are no-ops, including equivalent unreduced rational units.

## Acceptance evidence

| Requirement | Status | Evidence |
|---|---|---|
| Independent APIs and two CLIs | passed mechanically | 31-test stable suite and Ruff |
| Six clock classes, NB1 segmentation, best-effort exclusion | passed mechanically | all-missing, partial-missing, duplicate, regression, large integer, raw-complete fixtures |
| Nonclock bytes/immutable raw, no-op/idempotence | passed mechanically | byte comparisons plus storage roundtrip and rational no-op fixture |
| Binding/read-change/source/CSV corruption rejects | passed mechanically | 16 corruption classes, decoder-version/read-set/duplicate-row fixtures |
| Complete census gate, resume, old-artifact invalidation, all failed slots retained | passed mechanically | two-video/four-slot complete synthetic population, pending and rejection fixtures |
| NT1 artifact compatibility | partial | exactly NB1 output layout/schema and feature-only loader exercised; no real NT1 run or target-quality acceptance |
| Production full977 overlay / full2364 NB1 census | pending | authorized only after complete raw/decode; none built here |
| Fresh semantic/interface review | pending | parent-owned fresh Sol review; backend identity unattested, same-family provisional |

Production freeze/build, training, target/loss quality, scientific claims and independent-test/main-metric gates remain outstanding. No actual partial cache build, natural GT, models, GPUs, server process changes, commit, push, manifest state edits or cleanup occurred.
