# One explicitly authorized manual continuation

Root authorized exactly one resume after the first full acquisition stopped on an SSL handshake timeout. No new probe, code/config/launcher change, environment install, automatic retry, shared-state update, Git action or GPU work occurred here.

| Acceptance | Actual evidence | Result |
|---|---|---|
| Prior process exited | PID491938 absent; PRE_RESUME_RECEIPT.json | Pass |
| Exact runtime/control/root gate | Three runtime hashes equal frozen CONTROL/gate; control b7d6e8dea8ce857e172ea3d80f450c50b6b4607b3dfc56051267a9c9f97bc46a; gate09c730a10037c70d2bf5440e400872878a75618647f3f54181369bc050891e1d | Pass |
| Original evidence preserved | FAILURE_001 SHA f163ccaac5ad613e2cd56733587196e7e310bb9dbcce09f83babbd7245887eb6; attempt1 log SHA48fdafa322ec19219746ae3411562a4122495f367e52cc70b563991d06548c9b | Pass |
| No reset | attempts1→2; original deadline1794673524.2427886 unchanged; G1 reuse77936474B unchanged | Pass |
| Missing-range continuation | First attempt2 request begins771751936, exactly original rc00 coverage frontier | Pass |
| Durable progress | At16:40:21UTC four complete32MiB requests, additional134217728B; total new received=durable838860800B; rc00 coverage[0,905969664] | Pass |
| Resource/process binding | PID493360 alive, CPU26, nice10; launch environment numericthreads1, CUDA empty, cache/TMP under verified project root | Pass |
| Other work untouched | Parent TransRAC490003 and ablation488635 observed alive; no control action against them | Pass |
| Full acquisition and downstream qualification | Ongoing finite process; no completion or training release claimed | Pending |

Launch used original acquire_assets.py --control CONTROL.json --run --resume --review-file ROOT_DEPLOYMENT_GATE.json --review-sha25609c730a10037c70d2bf5440e400872878a75618647f3f54181369bc050891e1d under nohup/env/taskset26/nice10, with exclusive new ACQUISITION_ATTEMPT2.log. No scheduler or automatic continuation. On another failure stop and report; no further retry is authorized by this task.

These are time-stamped operational observations, not a guarantee the process remains alive later. Safe receipts include no HTTP URLs/headers. Signed transport evidence remains private on the server. Snapshot helper only reads acquisition state and emits receipts; no local computational tests ran. Local artifacts remain on E:.
