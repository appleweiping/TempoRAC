Parent routing receipt: requested `gpt-6.1-sol`, reasoning `xhigh`, fresh `fork_turns:none`, task `/root/input_audit_review`. The tool routing request is known; backend attestation is unavailable. The raw reviewer report follows unchanged. Same-family semantic assurance remains provisional.

# Fresh engineering input-audit review — 2026-11-04

Final disposition after the same-session narrow correction check: no remaining concrete engineering blocker. The full 977-video CPU decode may launch with the corrected script and the already bound full inventory. Scientific acceptance remains provisional. The original blocker and raw reproduction are preserved below.

Scope: read-only review of the six new artifacts below, directly from disk. Repository base and current HEAD at review: acc2bbf3000a7596336e7cff9864e295173fd888. All six were untracked at review start; this review did not edit repository artifacts. No executor summary was used. Read E:/AGENTS.md, project AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, docs/CVPR_RESUME_20261104.md and the active run state. Also inspected existing scripts/audit_temporac_media.py and the local canonical-source config for interface context. Did not access GPUs, server state, models, natural annotations or natural media.

Review identity: fresh native OpenAI agent session; exact runtime model identity is not exposed through this agent's tools. Requested workflow role is routine Sol technical reviewer. This is same-family review and does not confer scientific acceptance. The scientific protocol remains provisional. These scripts are engineering preparation, not model validation or release certification.

## Initial reviewed input SHA256

| File | SHA256 |
|---|---|
| scripts/audit_temporac_decode.py | 73da4c8b00fe5beb4da3a3174938a194b823d40d61d3df58764735023ffc728b |
| tests/temporac/test_decode_audit.py | 5c5ef4ae4b347ac2aab7bb3b66002de756e361d9b5ed3766266e364d028646a3 |
| docs/DECODE_AUDIT_20261104.md | bdde7c3462e70acd0e7ad5d92c1a29de1f7e2e5b20752f2fff4f0de0218f4858 |
| scripts/audit_temporac_sources.py | 77e6168cf664dd78c4a36d9b3f44bec2e9c58b08d9552dd09ce386efaf23daf5 |
| tests/temporac/test_source_inventory.py | 4bf6ac7ae5ba15f3607df8d92a628cf881ff2b67ba91267f430818a119fffc02 |
| docs/SOURCE_INVENTORY_20261104.md | 8a16945a048855400d50052957fdf24a96c02b6ecc1da70cb8f73d311b790061 |

## Initial concrete blocker — resolved by the narrow correction below

### P2 — A failed/empty decode is counted as decoded

Location: scripts/audit_temporac_decode.py:231 (unconditional record["status"] = "decoded"), consumed at :292 (increments status["decoded"]). The failure observations are already detected at :200–203.

Input SHA256: 73da4c8b00fe5beb4da3a3174938a194b823d40d61d3df58764735023ffc728b.

After decode_frames returns zero frames, or returns a nonzero decoder exit after only partial frames, audit_one still sets status to decoded. main therefore increments the decoded counter for those failures. This contradicts docs/DECODE_AUDIT_20261104.md:17, which specifies that nonzero-return/zero-frame cases remain per-video failures. A complete audit with one failed video could consequently summarize completed=977, decoded=977 even though one video did not complete decoding successfully.

Reproduction used only the existing synthetic pytest fixture on E: and patched subprocess objects; no natural inputs or server were accessed. probe_metadata returned nb_frames=3/time_base=1/20. Two independent decoder responses produced:

```json
{"case":"zero_frames","status":"decoded","returncode":0,"frames":0,"findings":["zero_decoded_frames","annotation_frame_count_mismatch","annotation_or_variable_dimension_mismatch","stream_declared_frame_count_mismatch"]}
{"case":"partial_nonzero","status":"decoded","returncode":7,"frames":1,"findings":["decoder_nonzero_exit","annotation_frame_count_mismatch","stream_declared_frame_count_mismatch"]}
```

Action: retain a failed decode state (or distinguish attempted/partial from successful decoding) for nonzero-return and zero-frame cases, and ensure the summary decoded counter counts only the intended successful state. Add a focused assertion for these two cases. Preserve their raw CSV, findings, logs and full-population continuation. The existing findings and final nonzero process exit already disclose anomalies; this is a bounded reporting defect, not evidence that the code hides findings or returns a clean audit success.

No concrete blocker found in scripts/audit_temporac_sources.py or the other reviewed behavior.

## Initial acceptance checks and evidence

| Requirement | State | Evidence |
|---|---|---|
| Test annotations untouched | Passed for reviewed implementation and supplied fixture scope | Decode main fixes expected splits to train/val (:262); load_inventory rejects other splits before media access (:43–48), checks media resolution stays under that split, and current enumeration is limited to those splits (:58). Annotation reader returns only length/width/height (:64–69). Source script opens explicit original-media candidates and receipts; it does not open composite media/annotations. Its CLI fixture succeeds with invalid annotation text and absent composite media. No new security framework is required by this review. |
| Full population/no dropping | Passed mechanically | Decode verifies expected count, duplicate paths and exact live train/val MP4 membership (:34–61), then iterates all rows and catches per-video audit exceptions. Source verifies full inventory counts/unique identities (:31–48), live directory membership (:168–172), exact reused development identities (:137–150), and retains every composite row in membership (:93–108, :215–216). Actual natural-run completion is not claimed. |
| Full decode vs header-only probe | Passed implementation check | Decode uses a separate ffprobe -show_frames stream (:126–170), drains stdout, waits for exit and preserves all returned video-frame rows. Its metadata-only -show_streams call is distinct (:112–123). The supplied tests mock this subprocess interface; the documented remote real ffprobe fixture was read but not independently rerun on the server. |
| Timestamp/coordinate units | Passed mechanically | CSV retains raw pts, pkt_dts, best_effort_timestamp and pkt_duration ticks (:15, :136, :150–155). Summary converts using selected stream time_base (:218–228), never index/FPS as a substitute clock. PTS and best-effort monotonicity findings are separate; missing timestamps do not create a synthetic adjacent interval. Reconstructed best-effort timestamps and prior-hash limitations are disclosed. |
| Source aliases/multiple/unknown candidates | Passed mechanically | All exact case-sensitive stem candidates survive (:51–65). All realpaths are hashed; byte-identical paths collapse only by SHA while distinct bytes remain ambiguous (:68–84). Missing and read errors remain explicit; unresolved test constituents yield unknown unless overlap evidence already exists (:95–107). |
| Exposure and conservative graph | Passed mechanically within declared scope | Both full train and val are current exposure (:88–91). All possible byte IDs of ambiguous exposure tokens participate. Same-token and possible-byte overlap are separately recorded; components retain both kinds of edges (:113–134). Historical exposure, synthesis recipes and transcoded recording identity remain explicitly unresolved; B is candidate only. |
| Truthful completion/failure states | Partial — blocker above | Source errors remain rows and yield complete_with_errors/exit 1 (:194–218, :232); decode logs/findings survive and final findings cause exit 1 (:243–245, :293–302), but zero/nonzero decoder outcomes are incorrectly counted as decoded. |
| CPU checks | Passed | 15 supplied tests passed in 5.17 s; command/output below. |

## Raw CPU check

Environment: E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe. Set TEMP=TMP=E:/ICASSP_TempoRAC/aris-work/tmp, PYTHONDONTWRITEBYTECODE=1, PYTHONUTF8=1. Exact command:

```text
python.exe -B -m pytest tests/temporac/test_decode_audit.py tests/temporac/test_source_inventory.py -q --basetemp E:/ICASSP_TempoRAC/aris-work/review_pytest_input_audits -p no:cacheprovider
```

Raw output:

```text
...............                                                          [100%]
15 passed in 5.17s
```

The additional two-case reproduction wrote only CSV/stderr fixture outputs beneath the same E: pytest base. No repository mutations, C:/D: artifacts, server operations, GPUs or models were part of this review. No claim of full natural-media audit completion, model validity, scientific approval or source-recording independence is made.
## Same-session narrow correction recheck

The parent requested only the status/counter correction recheck; this is not a new timed or expanded review. The source script was unchanged. Read revised decode source :199–248 and the new CLI regression at tests/temporac/test_decode_audit.py:175–197, then read the resulting actual fixture status and all three decode JSONL rows.

Revised input SHA256:

| File | SHA256 |
|---|---|
| scripts/audit_temporac_decode.py | 74696b26e826ef1da56fdfbf9b49b564c19363a93199271c5f3e83c73e558a7a |
| tests/temporac/test_decode_audit.py | a1424d4d382644fc7b3e22c0f457798052b9d4e4ba8fc997f4f8c52226ea20b6 |
| docs/DECODE_AUDIT_20261104.md | c2f780b8c887840011968133b670eff51747b018f576c1df8757ed602cfa2e78 |
| scripts/audit_temporac_sources.py — unchanged | 77e6168cf664dd78c4a36d9b3f44bec2e9c58b08d9552dd09ce386efaf23daf5 |
| tests/temporac/test_source_inventory.py — unchanged | 4bf6ac7ae5ba15f3607df8d92a628cf881ff2b67ba91267f430818a119fffc02 |
| docs/SOURCE_INVENTORY_20261104.md | 24aef52a78267c8e414f343f1b937166b2ab459f77ef4f2f04a286a441ba7e90 |

The docs were updated concurrently by their owners. Their initial hashes are preserved above, and both revised docs were read. The updated source inventory documentation reports server closure; independent server or archive verification was outside this code review and was not performed here.

At scripts/audit_temporac_decode.py:231, decoded now requires returncode == 0, decoded_frames > 0, and no parse_errors. Source-stat changes (:238–240) or unavailable after-stat (:241–244) force failed while retaining partial receipts. main's existing decoded-state counter at :294 therefore counts only this intended successful state. The new regression exercises zero-frame rc0, one-frame rc7, and a succeeding third video in one CLI call, preserving full-population continuation and exit 1.

Recheck command used the same E-only environment and exact command recorded above. Raw output:

```text
................                                                         [100%]
16 passed in 8.19s
```

Read-back from E:/ICASSP_TempoRAC/aris-work/review_pytest_input_audits/test_cli_does_not_count_zero_f0/output:

```json
{"status":"complete_with_findings","completed":3,"decoded":1,"findings_count":2,"script_sha256":"74696b26e826ef1da56fdfbf9b49b564c19363a93199271c5f3e83c73e558a7a"}
{"path":"train/clip0.mp4","status":"failed","decoded_frames":0,"returncode":0,"findings":["zero_decoded_frames","annotation_frame_count_mismatch","annotation_or_variable_dimension_mismatch","stream_declared_frame_count_mismatch"]}
{"path":"train/clip1.mp4","status":"failed","decoded_frames":1,"returncode":7,"findings":["decoder_nonzero_exit","annotation_frame_count_mismatch","stream_declared_frame_count_mismatch"]}
{"path":"train/clip2.mp4","status":"decoded","decoded_frames":3,"returncode":0,"findings":[]}
```

Final requirement disposition: the truthful decode-state blocker is resolved; all other initial mechanical checks stand. No remaining concrete blocker was found in these two engineering audit implementations. Launch of the complete 794 train + 183 validation CPU decode is supported within this scope. This does not declare that the natural files will all decode cleanly: findings must be retained and interpreted after the full run. Existing declared limits remain: the prior byte SHA is reused without a fresh rehash, stable stat cannot rule out an earlier same-size change, best-effort timestamps may be reconstructed, and source byte/path graph candidates do not settle synthesis recipes, recording-level identity or historical exposure. None is newly manufactured as a launch gate, and no model effectiveness is inferred.
