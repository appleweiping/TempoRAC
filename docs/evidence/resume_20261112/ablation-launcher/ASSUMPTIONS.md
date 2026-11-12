# Assumption ledger

ASK=never.

| ID | Gap | Chosen | Class | Source |
|---|---|---|---|---|
| A1 | Released matrix/evaluation | Exactly contract21 new runs/1320epochs; three singleUCF reused; union arms evaluate RC/CX/UCF, singleRC/CX corresponding test only, no QUVA ablation. | semantic | user/root/Astra contract |
| A2 | Arm protocol | Exact frozen cfg digest, seed, feature roots, splits,50/80epochs; multi best_target.pt vs single best.pt. Existing train489 and eval_saved unchanged. | interface | user/root |
| A3 | Output namespace | Fresh group frozen_known_source_v1_ablation20261112 and control resume20261112/ablation-continuation, preserving all old outputs. | interface | default |
| A4 | Explicit continuation | --resume only; verify same provenance, skip completed prefix, recover existing latest.pt for partial training, reuse valid completed evaluation files. Lane order fixed from contract. | interface | default |
| A5 | Manual hold | Root creates HOLD_AFTER_CURRENT_GPU0/1. Check only after current training+all frozen evaluations+completion receipt. Save exact remaining list and exit. Root clears flag before explicitly resuming. No metric-driven trigger, timer, retry, kill or migration. | interface | user/root |
| A6 | Source role | --source/--commit must name clean immutable training489; launcher itself bound by file SHA plus contract SHA in each provenance/receipt. | interface | default |
| A7 | Fixture scope | Pure configuration mapping and synthetic queue/hold/resume/failure transitions on server CPU, no reduced scientific training or checkpoint fixture rerun. | interface | user/root |
| A8 | Hard interruption after held-lane resume | A stale held/failed remaining list may advance only to its canonical suffix after every skipped arm passes current complete-provenance/file-hash verification. Missing or out-of-order completions reject; no arm/seed omitted. | interface | sweep |
