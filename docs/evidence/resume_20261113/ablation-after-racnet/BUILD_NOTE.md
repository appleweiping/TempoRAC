# Build record

|Rung|Required acceptance|Evidence|Status|
|--|--|--|--|
|F0|Metadata entry reaches durable fixture artifact; dispatch replaced|METADATA_FIXTURE.json|PASS|
|F1|Canonical105/1243/full200/9600attempts/189 native validations/full357 and exact best-MAE identity|metadata fixture positive + negatives|PASS (metadata only)|
|F2|Live/occupied dependency waits; explicit release; original hold archive; lock/once claim; one unchanged worker --resume|metadata fixture lifecycle cases|PASS (fake processes/dispatch)|
|F3|Bounded fresh routine Sol interface review|Root dispatches using REVIEW_READY.json|PENDING|

No actual --check or --execute, GPU query, model import, checkpoint/array load, GPU job, original hold/lane modification, Git/status/source_sync publication was performed. Completed7 and correctedfull3 are untouched. Original worker and scientific contract unchanged. Metadata fixture used one AMP overflow skip, verifying successful updates need not equal9600.

Implementation borrows the reviewed GPU0 one-shot structure (6116d44c3bff0cdeaf22854eb123c537f06a9aaf5333edcd9504406d109aa0cb); new dependency checks are RAC-specific metadata only. The unchanged original worker validates old completion payload digests at actual --resume; this entry does not repeat them. RAC checkpoint identity is the authoritative COMMIT/ref/COMPLETE binding plus file existence and nonzero bytes, no duplicate payload hash/load.

Process exit checks match PID plus exact recorded command to avoid treating unrelated PID reuse as owned work. Both existing RAC writer locks must be free. Future resource check queries only physicalGPU1, <500MiB and no processes, with shared128GiB free reserve. It never signals occupants. Failure after durable claim is manual recovery only, no automatic repeat.
