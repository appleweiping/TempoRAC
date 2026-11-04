# Full 977 CPU decode closure — 2026-11-04

The existing run is complete: 794 train + 183 validation videos, 977 distinct inventory paths, 695,625 decoded frames, decoder rc0 for all 977, zero row findings and zero closure discrepancies. Original PID 1543699 is absent; launch log and terminal status agree. Original process exit code was not separately captured. Runtime was 7,639.016 seconds (127.317 minutes).

| Acceptance requirement | Result | Evidence |
|---|---|---|
| Existing full run completed without restart | passed | restored/decode/status.json and restored/launch/launch.log; PID inspection in aggregate.json |
| Exact 794/183 membership and 977 distinct paths | passed | all JSONL paths/splits/prior SHA/bytes compared with restored/inventory/media.jsonl |
| Full decode counts/dimensions match declarations | passed | every row status decoded, rc0, no parse errors, frame count equals annotation length and stream nb_frames; every existing CSV independently parsed and matched |
| Presentation clocks | passed | PTS and best-effort: zero missing, duplicated or reversed ticks across 695,625 frames; original raw CSV preserved |
| DTS diagnostics retained | passed with diagnostic | 1,954 missing packet DTS ticks; zero reversed/duplicate valid DTS. DTS is not substituted for presentation time |
| Physical durations and modes | passed | five rational time bases; 977 constant within-video adjacent-step clocks. This was measured, not assumed; per-video metrics in duration_metrics.json |
| Raw evidence archival | passed | 977 clock CSV + 1,954 stderr files and all metadata preserved; all stderr files empty |
| E + server recoverability | passed | same archive SHA; restored manifest verifies 2942 artifacts, 30800644 bytes with zero mismatch |
| Fresh decode-time media SHA binding | partial by design | prior media SHA copied; current byte size and before/after stat agree. Same-size changes since prior SHA pass cannot be excluded |
| Scientific/release/source-independence acceptance | not established | this is deterministic media-job completion only |

Measured time bases (videos): {"1/12288": 3, "1/11988": 348, "1/15360": 438, "1/12800": 111, "1/19184": 77}. Observed rates (videos): {"24.0": 3, "29.97": 348, "30.0": 438, "25.0": 111, "23.98": 77}. No global FPS was used. Best-effort timestamps may be reconstructed by the decoder; raw PTS/DTS are preserved separately.

Presentation span (last best-effort minus first, scaled by time_base): total 23,912.825060759 s, range 7.005838198–87.966666667 s. Span plus positive final packet duration: total 23,946.812681544 s, range 7.047539616–88 s; valid for all 977. Sum of positive per-frame packet durations independently has the same total. These quantities have different definitions and are retained separately.

Archive: /mnt/4tssd/lilei/TempoRAC/work/decode-closure-20261104/decode-evidence.tar.gz (server) and decode-evidence.tar.gz (E), 5612293 bytes. SHA256 `da485e71553e816bdfec950208f4f6846391260f87850cb5c0baa5ea621f5769`. Restored artifact manifest SHA256 `fded16a617072846d263b4a1d65194c38c1e86347f2441c43f650aa8a30954cb`. Machine receipts: completion_receipt.json and restore_verification.json. Small exact status/inventory/aggregate copies are in small/.

Exact source/raw SHA256:

| Artifact | SHA256 |
|---|---|
| decode/status.json | `5556f2e5a693f02adb81256ea600cc6d07d1aed1f111f98214752f6d4c88317b` |
| decode/decode.jsonl | `f7a97582f4c946c94b382d4a853fc6053791bd187b025d48257b3d8aefb02782` |
| inventory/status.json | `9cc21492cf73ce1949a2b25ee564f37afba83d3188b5290747dd968282682773` |
| inventory/media.jsonl | `8ac8ec586326a3868b6cc2710cddf8ed24f9e9fa85eb0969031567492924576b` |
| source/audit_temporac_decode.py | `74696b26e826ef1da56fdfbf9b49b564c19363a93199271c5f3e83c73e558a7a` |
| closure/aggregate.json | `481a6019619fd4714e57eb70b7bc5d4e392339b42a8bc292fb2485e7ea9486b0` |

The pinned Codex monitor-experiment SKILL was read in full; source and installed hashes both 66a42b21a7ec5a9c4ee2d4da6babfd533764d6c4188dcc178d5bcbd1d8a31072. Project lock/workflow/resume/run-state, decode audit, external-cadence and acceptance-gate support were read. No cadence or notification was created. Ordinary Sol requested, backend identity unattested; same-family assurance remains provisional.

No MP4 was read, rehashed or decoded by this closure, no annotation was opened, and no test/person-label/model/GPU/shared-environment/other-job operation, cleanup, commit, push, or repository mutation was issued. Healthy raw extraction and STaR jobs were left untouched. Prior inventory duplicate evidence and limitations remain archived. All closure artifacts live under the explicitly owned E/server closure directories.
