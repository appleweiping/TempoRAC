# Seventh-arm acceptance

| Requirement | Result | Evidence |
|---|---|---|
| Preserve workers and frozen scope | pass | Read-only /proc and saved artifacts; collector writes confined to collection directory; no HOLD, start, signal, restart or migration |
| New completed arms exist and exact50 epochs | pass | arms[].logged_epochs and exact range0..49 |
| Config/provenance/source/contract exact | pass | arms[].config_and_provenance_exact plus saved raw configuration and provenance |
| Full fixed populations and prediction keys | pass | arms[].evaluations; exact manifest key sets for both views |
| Artifact SHA integrity | pass | arms[].artifact_hashes; all completion-listed artifacts hashed server-only |
| Component contribution / performance ranking | pending | Intentionally no scientific conclusion before paired3seed closure |
| Primary/all/zero metrics preserved and recomputed server-only | pass | PRIMARY_ALL_ZERO_METRICS.json; every stored metric matches within1e-10; all views/zero predictions retained |
| GPU0 held remainder and GPU1 continuation | pass | gpu0_hold_receipt exact10; workers latest metadata and next_contract_arm; active raw log |
| Whole-arm handoff boundary | pass | Original worker held_after_complete and exited; exact4residual preserved |
| Previous6immutable | pass | Exact prior receipt SHA50f85e98; no old collection overwritten |
| Token3scientificjudgment | pending | Only2/3token seeds complete; no bootstrap or groupacceptance |
| Transferintegrity | pending | LOCAL_TRANSFER_VERIFICATION.json appended after transfer |
