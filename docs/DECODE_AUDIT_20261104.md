# Decode audit — 2026-11-04

## Specification (before implementation)

Base commit: `acc2bbf3000a7596336e7cff9864e295173fd888`. ASK=never, balanced/draft; the requested three-file ownership combines SPEC, assumptions and build note here. No commit/push. Implement a read-only, single-worker CPU full-frame decode of all 794 train + 183 validation MP4s. Reuse the completed byte inventory; do not rehash media, access test, train, or change shared environments. Read matching train/val JSON solely for top-level length/width/height. Outputs in an isolated server run: status.json, decode.jsonl, clocks/*.csv and logs/*.stderr. All anomalies remain in receipts. Success command is the CLI below; exit 0 only after all videos decoded without findings, 1 after a complete run with findings, 2 for input/setup failure. This is media evidence, never model efficacy or release certification.

## Assumptions recorded before code

| ID | Decision | Class | Source |
|---|---|---|---|
| A-001 | Decode the exact completed inventory population, require 794/183 and matching current MP4 membership; test is never enumerated or opened. | semantic | user |
| A-002 | Use ffprobe software decoder `-threads 1 -show_frames`, with no hardware acceleration and serial files. Frame metadata is emitted from decoded AVFrames; a stream metadata-only probe is not decoding. | semantic | default |
| A-003 | Preserve per-frame PTS, packet DTS, best-effort timestamp and packet duration ticks. Physical units come only from selected stream time_base. Inspect strict monotonicity of PTS and best-effort presentation ticks separately. Missing/duplicate/reversed presentation ticks are findings; DTS is preserved/diagnosed, not substituted for presentation order. No index/FPS clock synthesis. | semantic | default |
| A-004 | Store original reported avg_frame_rate/r_frame_rate as metadata; observed frame interval range/rate comes from best-effort ticks times time_base. Neither a nominal rate nor an annotation implies a valid physical clock. | semantic | default |
| A-005 | Compare decoded frame count and every decoded dimension to JSON length/width/height, and count to stream nb_frames when available; differing dimensions or declarations are findings, not deleted samples. | semantic | default |
| A-006 | Prior inventory lacks mtime/inode. Copy its SHA and verify current byte size; snapshot dev/inode/size/mtime_ns before/after decoding. This binds the receipt to the prior inventory identifier plus stable observed stat, but cannot exclude same-size changes between the earlier SHA pass and this pass. No claim of a freshly verified SHA binding. | semantic | default |
| A-007 | Any nonempty decoder stderr is a retained finding (warnings may be benign), with full text in per-video logs; nonzero return/zero frames and JSON/read failures remain per-video failures and do not stop later videos. | semantic | default |
| A-008 | CLI schema temporac.decode-audit.v1, one CSV per inventory row indexed deterministically, atomic status every video and flushed JSONL. Output must be outside the dataset and new; no silent resume/overwrite. | interface | default |
| A-009 | Only top-level annotation length/width/height leave the JSON reader. Python parses JSON into memory transiently; no object/count/period values are read by code, logged, or exported. | semantic | user |
| A-010 | No whole-file timeout: legitimate long videos must finish; external interruption leaves last completed status and partial per-frame evidence, not a complete result. | interface | default |
| A-011 | `decoded` counts completed nonempty rc0 frame decodes with no frame parse failures and stable media stat. Partial/nonzero/zero/unstable-source attempts are failed, retain all evidence, and never increment decoded. Clock/declaration findings remain separately reported. | interface | sweep |

## Acceptance ladder

| Rung | MUST check | Status |
|---|---|---|
| F0 | Empty complete-inventory fixture creates status/decode artifacts end-to-end (engineering fixture only, no dataset claim). | passed: test_f0_empty_fixture_end_to_end |
| F1 | CPU tests: strict inventory scope/population, safe annotations/stat binding and findings retained. | passed: accumulated 10-test suite |
| F2 | CPU tests: streamed decoded frame clocks/count/dimensions, missing/reversed timestamps and decoder failure retention. | passed: accumulated 10-test suite |
| F3 | Actual available ffprobe decodes CPU-generated video and records original timestamps; independent technical review; then launch all 977 real videos. | real CPU fixture passed; independent review/full launch pending |

## Execution record

- Full prior byte audit reported complete: 977 files, 4,621,403,934 bytes, 1,251.508 seconds, 0 failures; 4 train duplicate groups, no train/val byte duplicates. Not repeated here.
- Remote system has no command-path ffmpeg/ffprobe or cv2/av/imageio_ffmpeg. Conda cached ffprobe exists but initial run lacks libtesseract.so.5. Dependency resolution stays in isolated TempoRAC root if required.
- Prior inventory has no stat timestamps; A-006 documents the unavoidable provenance limitation without rehashing.

## Final acceptance evidence

Pending. Code checks, real full decode completion, findings, technical review and limitations will be recorded separately.

## Engineering evidence (before natural full run)

- CPython 3.11.2 in E-only audit env: `python -B -m pytest tests/temporac/test_decode_audit.py -q -p no:cacheprovider --basetemp=E:/ICASSP_TempoRAC/aris-work/tmp/pytest-decode-311` → 10 passed in 6.73 s. Initial check exposed Windows path-separator mismatch and zero-population fixture elapsed-time omission; corrected and suite rerun. These synthetic tests are engineering checks only.
- F0 output/status and empty JSONL exist in the pytest fixture. Feature implementation was delivered in one pass before the accumulated F0/F1/F2 check, rather than claiming a separately executed staged ladder.
- Available cached ffprobe 6.1.1 was made runnable via this process's LD_LIBRARY_PATH, preferring cached pcre2 10.48 before cached lib directories. `ffprobe -version` exited 0 with empty stderr. No package installed or shared environment changed.
- Actual remote CPU interface fixture: ffmpeg generated five 32x16 MPEG4 frames at 25 Hz, ffprobe decoded all five with no stderr; stream time_base 1/12800, original PTS/DTS/best-effort ticks 0,512,1024,1536,2048; observed interval .04 s. Audit exit 0, 0 findings, .081 s. Artifacts: isolated `runs/decode-interface-fixture-20261104/output` under server TempoRAC root. Fixture prior SHA is explicitly a synthetic `aaaa...` placeholder, never evidence about natural media.
- Annotation exceptions and metadata exceptions do not block the full decode attempt. Per-video decode exceptions retain logs/stat and the run continues to remaining inventory rows.
- Independent technical review requested from parent; natural 977-video run not yet launched at this checkpoint.

- Final accumulated check after import ordering fix: Ruff 0.14.4 all checks passed; CPython 3.11 suite 10 passed in 3.54 s, exit 0.


## Independent technical-review correction

The fresh reviewer reproduced that zero-frame rc0 and partial-frame nonzero-rc receipts were incorrectly labelled decoded, inflating summary decoded. Corrected status semantics before natural execution: decoded requires decoder exit 0, at least one frame, no frame parse errors, and stable before/after source stat. Otherwise failed, while all partial counts, clocks, findings and logs remain. Timestamp/declaration findings remain distinct from whether the full decoder ran successfully. Added CLI regression with zero-frame rc0, one-frame rc7 and a succeeding third video: completion must be 3, successful decoded only 1, findings 2. Original reviewer recheck pending; no natural full run started.

- After reviewer correction, Ruff all checks passed and CPython 3.11 suite 11 passed in 4.38 s, exit 0. Revised source SHA256: `74696b26e826ef1da56fdfbf9b49b564c19363a93199271c5f3e83c73e558a7a`.

## Full CPU run launched (not completed)

After the original independent reviewer rechecked the correction and reported no remaining blocker, the complete 977-video job was launched as PID 1543699. Source SHA256 exactly matched `74696b26e826ef1da56fdfbf9b49b564c19363a93199271c5f3e83c73e558a7a` before launch. Actual process inspection showed nice 19, orphaned parent 1 and ionice best-effort priority 7. GPU visibility is empty; one worker and decoder thread 1. No media hashes repeated, no shared package changes, no STaR action.

Server isolated receipts: `runs/media-decode-20261104.launch.json` (exact command/PID/CPU environment/library paths), `.launch.log`, and `.runtime.json` (ffprobe SHA256 `3984b415612635368073220da8b1406488573f3d674b34beebbb57ec7c91b53f`, ldd output and 56 resolved library files with hashes). Hashes in runtime receipt concern tool/dependencies, not media. Output directory is `runs/media-decode-20261104`; full current train/val population preflight was still performing read-only filesystem I/O at the 32-second inspection. First decode receipts and complete-run evidence remain pending; launch is not a dataset PASS.

## Initial natural-media receipts and acceptance state

At process elapsed 1m29s, `status.json` reported running, completed 2/977, decoded 2 and 0 videos with findings. First video decoded 216 frames at stream time_base 1/12288 (tick step 512, 24 Hz); second decoded 900 frames at time_base 1/11988 (tick step 400, 29.97 Hz). Both agreed with technical annotations, had constant expected dimensions, strict PTS/best-effort presentation clocks, stable source stat and no stderr findings. Every frame's PTS/DTS/best-effort/duration ticks remain in separate CSVs; no equal-spacing clock was synthesized. These initial receipts demonstrate execution only; the remaining 975 videos were pending at this inspection.

| Requirement | State | Evidence |
|---|---|---|
| Scoped implementation and CPU interfaces | passed (mechanical) | Only script/test/this document changed by implementer; Ruff pass; 11 CPython 3.11 tests pass; actual CPU-generated frame decode pass |
| Failed/zero/partial decodes not counted successful | passed (regression) | CLI regression preserves three attempts and reports successful decoded=1; corrected source hash recorded above |
| Fresh technical review before full launch | reported no blocking issue by parent; semantic assurance at most provisional | Parent retained original review and recheck; no cross-family acceptance claimed here |
| Complete 794 train + 183 val job launched | passed (launch only) | PID/command/source SHA/tool+dependency SHA/CPU and IO priority receipts; initial 2 natural receipts |
| All 977 videos decoded and examined | pending | Running; 2/977 at last scoped inspection, not a complete dataset result |
| Per-video stat + prior SHA provenance | partial by design | Prior SHA copied; bytes compared and before/after stat saved. Prior audit lacks stat, so no proof excluding same-size changes since old hash pass |
| Test untouched, no labels exported, no GPU/shared-env/STaR changes | passed within this task's scope | CLI restricts train/val, reader returns only length/width/height; CPU process and cached tools; no such changes issued |

Private launch/runtime receipts were also downloaded to E-only aris-work for recovery. No dataset cleanup or sample exclusion was performed. Parent owns monitoring/completion aggregation of the active full run.

## Parent continuation snapshot

At approximately09:58UTC the downloaded running receipt reported94/977 completed,94 successful decodes,0 findings. PID1543699 was observed live. `evidence/decode_progress_20261104.json` binds this snapshot; it does not claim completion. The original reviewer report and narrow correction recheck are now retained in `INPUT_AUDIT_REVIEW_20261104.md` (requestedSol/xhigh, exact backend not attested, same-family provisional).
