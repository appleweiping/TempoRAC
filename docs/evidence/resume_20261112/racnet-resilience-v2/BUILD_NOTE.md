# Bounded operational build evidence

| Requirement | Actual server evidence | Result |
|---|---|---|
| Only transient network timeout/reset retries | CPU_RESILIENCE_V2_FIXTURE.json network-origin marker/filter case: timeout/reset/abort eligible; certificate and non-marker validation/disk/schema/deadline errors propagate | Pass within fixture scope |
| Preserve partially received bytes and resume durable frontier | Actual Engine.request + mocked RangeResponse: failed[0,7] received=durable2; next request[2,7] received=durable6; final payload8; one2s retry slot | Pass |
| Persistence fault is terminal despite timeout | Injected segment-journal disk fault after timeout; received2/durable0, one request, zero retries; no repeated commit | Pass |
| Consecutive budget | Delays2/5/10,3slots;4th failure terminal; reload performs zero additional request | Pass |
| Total budget | Initial63; one slot reaches64; next failure terminal; reload performs zero additional request | Pass |
| Original total deadline | Before backoff: no slot/no second request; deadline reached during reserved backoff: one consumed slot/no second request; original absolute deadline retained | Pass |
| Production preserved | Nine critical live file hashes unchanged across fixture; production-preserved snapshot includes code/control/state/failure1+2/log1+2 and private journals | Pass |
| Atomic rebind reviewed/deployed/full acquisition | Exact candidate state proposal and root-only procedure prepared; root focused inspection/publication/promotion/continuation remains pending | Not performed |

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= taskset -c26 nice -n10 /root/autodl-tmp/TempoRAC/env/bin/python -B cpu_resilience_v2_fixture.py

Two CPU executions of this one targeted fixture were necessary. First receipt at16:57:02UTC passed its checks, including zero additional retry reservations after cap reload; packet finalization identified that a resumed exhausted job could still make an uncharged initial network call. The final candidate persists the terminal cap reason and checks it before the network loop. The reload assertion was strengthened to zero additional requests; final fixture exit0 at16:58:31UTC. First candidate/receipt/log are retained in pre-terminal-cap-guard/. No real requests or timer/training/pilot occurred in either fixture.

Policy is frozen once for root's upcoming restart: connect30/idle120/request300 seconds; request chunk<=128MiB; checkpoint1MiB;3consecutive automatic retries;64lifetime automatic retries;2/5/10second backoff. It changes operations only. Exact61objects128629792634B,384GiB asset cap,128GiB reserve, fixed .00/UCF/.01..59 order, original48h deadline and all raw results/negative history remain unchanged. Protocol originally had zero automatic retries; this explicit amendment supersedes that operational setting only when root promotes it. Training and all scientific inputs/contracts remain unchanged.

Pinned research-implement-feature SKILL was fully read earlier in this same session (33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8); same-family positive assurance remains provisional. Root explicitly requested this single amendment and owns focused inspection rather than a generic review loop. All files E:/verified server project root only. No new background Windows processes, local validation, environment installation, source_sync, Git or shared progress edits.
