# Exact rebind proposal — root only; not applied

Production remains stopped after attempt2. Candidate runtime/control/state files exist only in resilience-v2/. No third acquisition, production overwrite, shared source_sync/status change, Git write, new network probe or model work was performed.

Final frozen identities:
- acquire_assets.py:733fcbdeb28c98b3dc624e947b551a6c9a1d0408bf1fdcb20c20d18e85ff87b6
- range_client.py:3778fac8e44a728f9727dd45e0b15fef3975c923ed9791da9d89dd02b9e567f7
- schema_audit.py unchanged:2f11f050d39ca189f2b52e09b30ed0101cc1e3fde455ffbd0a82376f22145758
- CONTROL.candidate.json:682c9c9cd411d1f953273b1a3e165b7b4cea8656b4e2937bde708ff7137cf2de
- STATE.rebound.candidate.json:e79ba344da3f03a140dbaea79a86e656893c3a36cb46bf137d01b06d9b5318c4

Production prerequisites, all must still match immediately before promotion:
- STATE.json:c66238255c04985977323a96847154d712f1e0847b80c11ac3ae9e6d21046e0c
- CONTROL.json:b7d6e8dea8ce857e172ea3d80f450c50b6b4607b3dfc56051267a9c9f97bc46a
- FAILURE_001/002 and original logs preserved in production-preserved/ with original private journals; PRODUCTION_PRESERVATION_MANIFEST.json records exact identities. No object bytes were copied/reset/changed.

After root's focused code inspection and three-way publication, promote only under the existing WRITER.lock with the actual prior worker dead. Validate candidate manifest and exact production state hash again. Replacing runtime/control first while holding this exclusive lock prevents an intermediate worker from opening a mixed bundle. Root must generate a NEW deployment gate explicitly binding the amended control/code/policy/publication; the old root gate authorizes the previous zero-retry code and must not be reused for v2.

The proposed state was created by deep-copying the exact stopped production state, replacing only control_sha256 and adding transport_retry={total_retries:0,consecutive_failures:0} plus an appended operational_amendments history entry. Zero means no automatic retries have yet been spent under this new policy; it does not erase the two manually authorized attempts. All existing fields including status, pid, attempts2, started_utc/epoch, original deadline1794673524.2427886, every object's coverage/verified/SHA, G1 reused77936474B, and network received=durable1177829376B remain identical.

For atomic state replacement, write the exact staged STATE.rebound.candidate.json bytes to a same-directory W/STATE.json.tmp opened exclusively, flush/fsync, verify its SHA is e79ba344da3f03a140dbaea79a86e656893c3a36cb46bf137d01b06d9b5318c4, then os.replace(tmp,W/STATE.json) and fsync W. Do not reset attempts/deadline, rerun --prepare, recreate object files, or reconstruct coverage from assumed transfer lengths. Assert the final runtime/control/state/gate identities before releasing the lock and using one actual --run --resume continuation under CPU26/nice10/numericthreads1/CUDA empty and the original resource limits. Root owns this action; no executable promotion was run here.

Any further transient failure reserves and persists a retry slot before its2/5/10s backoff, then recomputes current missing ranges. A4th consecutive transient failure or a request needing65th automatic retry records a terminal reason; reload is rejected before any new request. Success clears consecutive count only. Total retries and terminal stop reason survive resume. Original deadline, disk/reserve, certificate, host/status/range/length, hash/schema and checkpoint/metadata faults never release another retry. If even checkpoint or request/state persistence fails after a transport timeout, that failure is terminal and original receipt/error remains available.

Known limits: seven targeted CPU fixture groups pass, including actual partial timeout and journal accounting with mocked TLS/socket. They do not prove real transport availability, full archive content/schema, resource sufficiency over48h, scientific quality, baseline qualification, or training readiness. No new semantic review loop was invoked; root focused inspection remains pending, same-family assurance remains provisional. Candidate CONTROL and preserved journals are private; do not copy them into public Git.
