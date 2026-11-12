# Bounded launcher review acceptance contract

Review session: /root/ablation_launcher_review_20261112. Requested reviewer: gpt-6.1-sol, xhigh minimum. Actual backend model/effort: unattested/null. Same-family assurance remains provisional. This table defines the launcher-only obligations; it does not reopen Astra science or completed checkpoint durability review.

| Obligation | Evidence to inspect | Initial status |
|---|---|---|
| Exact frozen21/1320epoch matrix, seed and 50/80epoch budgets | RUN_CONTRACT.json, frozen_27_spec.json, checked_plans, server ENGINEERING_REPORT | pending |
| Unchanged training/evaluation semantics and source489 | worker diff, source gates, immutable train/eval source | pending |
| Correct manifest, feature/aux roots, splits and population mapping | frozen spec, evaluation role commands, full prediction-key checks | pending |
| Validation-only checkpoint roles; fixed2view eval; no new QUVA | source489 train/eval and all21 mapped jobs | pending |
| Explicit identity-checked partial/completed-prefix resume | provenance/receipt checks and lane state transition code | pending |
| Failure prevents successor; hold only after complete arm; lane isolation | run_lane and existing server CPU fixture | pending |
| No local experiment/production dispatch, no core change or matrix expansion | review action record and candidate artifacts only | pending |

Initial candidate worker SHA256: 77572af3665c4f7632484d171b5fff415c04b53c44fb88de0fdbc38ba0df2183.
Initial fixture SHA256: fbc042dedb2da3103a2baf731be9df0b5d4132bae822a759d7ecea56ccb738b1.
Contract SHA256: dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167.
