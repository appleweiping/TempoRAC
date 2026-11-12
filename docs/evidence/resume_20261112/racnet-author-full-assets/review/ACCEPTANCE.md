| Acceptance obligation | Result | Evidence |
|---|---|---|
| Fixed 61 objects, source revision and byte/SHA/URL identity | Pass static | CONTROL independently matched all 61 original manifest records, 128629792634 bytes; no mismatch |
| Exact HTTP206, Content-Range/body size and TLS host allowlist | Pass within inspected transport | range_client.py; existing fixture only, no new network |
| Receive accounting and durable explicit resume | Block | R1 and R2 in REVIEW.md |
| Consistent extraction resume across RC→UCF boundary | Block | R3 in REVIEW.md |
| Resource/deadline scope including extraction | Partial | Bounded resource checks present; failure checkpoint issue R2 |
| Safe TAR/ZIP/NPY, no pickle/code execution | Pass static, full input pending | acquire_assets.py and schema_audit.py; existing UCF sample receipt |
| Original arrays, required gt_tsm, declared interface gaps | Pass implementation policy | schema_audit.py preserves bytes and records literal key gaps; full population not acquired |
| No training/install/GPU/scheduler/other-process control | Pass static | launch_once.sh and three Python runtime files |
| Independent review identity/assurance | Provisional only | Requested Sol/high; actual model and effort unattested; same-family |
| Full assets acquired | Not run | Prepared STATE attempts=0; downloader not launched by reviewer |
