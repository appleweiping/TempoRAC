# One-shot TransRAC → frozen GPU0 ablation successor

Base source publication e0bc6b0; no live runtime/contract/worker changes. ASK=never. Root explicitly authorized preparation only, default --check and future explicit --execute. No watcher, poll loop, scheduler, training/model test, environment build, process signal or current HOLD/worker/lane mutation is authorized now. Root owns the one necessary independent engineering check and publication/deployment decision.

Reuse: existing ablation_worker.py (SHA d23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144) exports checked_plans/validate_residual and already runs a finite lane with complete-prefix validation, strict saved provenance/output hashes and fail-on-first-child-error. No equivalent TransRAC dependency gate exists there. Preserve its exact file and training commit489dc77; invoke it once with --gpu0 --resume and original contractdc60e3d4.

| Requirement | Acceptance | Initial state |
|---|---|---|
| Fixed project/source/contract/held residual/complete prefix | --check reads hashes/JSON and reuses original helper functions, no torch import | pending |
| TransRAC full200+357 receipt/prediction member identity and exit | Completion, aggregate hashes,357exact member keys+record identities,200/4800 accounting; no inference/checkpoint unpickle | prepared path, future receipt unavailable |
| GPU0 genuinely idle | Only after complete+exit: nvidia-smi physicalindex/UUID/process XML and memory<500MiB; any process/error fails closed. Original worker rechecks at actual child start | future only |
| One attempt, no failed-run restart | Exclusive claim, new success/failure state/event/log; pending incomplete run directory or failed-after-hold history blocks; no retry loop | pending static inspection |
| Preserve historical HOLD/events/checkpoints | Future gate passes: rename old HOLD bytes to a new history filename, SHA/event record; then originalworker --resume. No current mutation | pending read-only check |
| Current observation WAIT with no mutations | Server --check returnsWAIT whileTransRACalive; before/after identity hashes forHOLD/lane/events/worker PID | pending |

Declared operational interfaces: --check is read-only and exits once with WAIT/READY/BLOCKED; --execute also exitsWAIT before claim if dependencies unfinished. Only explicit --execute with all gatesREADY consumes the one-shot claim. The future HOLD rename is root-authorized, retaining raw bytes/SHA and events. A crash after claim, archive or spawn is manual recovery only; never start a second worker automatically. No scientific config,seed,input,cost budget orevaluation change. Reuse actual environment/spec and existing first-run receipts; do not rebuild/revalidate an unchanged environment or treat historical Aug15 ledger as current authorization.
