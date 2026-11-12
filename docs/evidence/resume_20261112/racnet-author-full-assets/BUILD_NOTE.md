# Build and recovery acceptance

Original packet is retained at preserved_pre_repair_r1_r3/. The initial independent negative review at ../racnet-author-full-assets-review/REVIEW.md identified R1–R3; its verdict is not erased by executor checks.

| Rung / requirement | Evidence | Result |
|---|---|---|
| F0 inherited transport spine | CPU_TRANSPORT_REGRESSION_R1_R3.log / CPU_TRANSPORT_FIXTURE.json, server command exit 0 | Pass within engineering fixture scope |
| R1 rename-before-state adoption with/without journal and wrong identity rejection | CPU_RECOVERY_R1_R3_FIXTURE.json first 3 cases; zero requests, original deadline123456 retained | Pass |
| R2 deadline/reserve terminal flush | Cases R2_deadline / R2_reserve: received2, durable2, one commit, original error preserved, only one body read | Pass |
| R2 failed checkpoint accounting | R2_checkpoint_failure: attempted8/received4/durable0; R2_deadline_checkpoint_failure: attempted8/received2/durable0 and timeout retained; one commit each | Pass |
| R3 actual RCReader nested UCF stop/replay | Distinct progress files; RC prefix3/UCF prefix2, successful member replay, zero requests | Pass |
| Unstarted prepared state/G1 preservation | RECOVERY_REBIND_RECEIPT.json: attempts0, no deadline, original coverage unchanged, unique reuse77,936,474, new acquisition bytes0 | Pass |
| Independent targeted closure | Same saved reviewer session owns closure against exact refreshed packet | Pending; same-family at most provisional |
| Formal acquisition | launch_once.sh remains staged | Not launched |

Server-only command: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= taskset -c 26 nice -n 10 /root/autodl-tmp/TempoRAC/env/bin/python cpu_recovery_r1_r3_fixture.py

Actual run record: first recovery fixture exit1 at a synthetic UCF replay stream whose chunk boundaries differed from its interrupted stream; raw CPU_RECOVERY_R1_R3_ATTEMPT1.log retained. Fixture-only correction used the same chunk contract. Second recovery fixture exit0 with8cases; original transport fixture regression exit0. No additional scientific validation, probes, downloads, arrays, model, training or environment installation occurred. Do not broaden this evidence to full acquisition, real input schema/population, fair baseline qualification, SOTA or training readiness.

No live stubs, scientific changes, or new semantic assumptions. Existing semantic rows A01/A07/A08/A09 remain unchanged in ASSUMPTIONS.md. New operational interface rows A13–A15 are tied to the saved review. Standalone private packet: no Git commit, shared status, source_sync or other-project edits.

Pinned skill source/installed SKILL SHA256:33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8; pinned checkout HEAD:f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Full research-implement-feature SKILL read including Phase4; reviewer-independence/backends and scope/effort/acceptance/injection/output references consulted. Parent retains same-session reviewer route and identity limitation; executor does not certify reviewer model/effort.
