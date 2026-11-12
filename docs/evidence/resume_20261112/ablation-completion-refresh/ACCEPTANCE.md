# Acceptance contract — ablation completion refresh

Scope: finite saved-artifact collection only, excluding already archived wo_pyramid_s0. No experiment, inference, model load, process control, scope expansion, shared status or Git updates.

| Requirement | Required evidence | Initial state |
|---|---|---|
| Identify newly completed arms | completion records minus wo_pyramid_s0 | pending |
| Exact 50 epochs and finite losses | raw log.jsonl plus server assertions | pending |
| Frozen config and provenance | contract/spec/source hashes, exact config and seed assertions | pending |
| Full evaluation memberships | exact both-view manifest keys: RC152 / Countix2403 / UCF105 | pending |
| Checkpoint and bound artifact integrity | all completion-listed SHA256 verified server-side; no checkpoint copies | pending |
| Small raw evidence and transfer | self-contained receipt, raw logs/predictions/metrics, manifest; matching local SHA | pending |
| Actual lane status | timestamped events, /proc, GPU snapshot | pending |
| Component contribution verdict | deferred to paired3seed group closure | pending by design |
