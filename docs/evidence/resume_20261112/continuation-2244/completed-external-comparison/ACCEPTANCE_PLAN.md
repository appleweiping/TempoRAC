# Acceptance plan: aggregate-only completed external comparison

Frozen scope: corrected TempoRAC full3, inherited complete ESCounts released RC checkpoint evaluations, RepNet updated2025 ckpt70 full357, newly complete DeTRC official TSN/R50 full80/full357. TransRAC/RACnet stay pending. No new literature, scoring, inference, model/config/protocol, experiments, queue edits, Git or shared-status edits.

| Check | Required evidence |
|---|---|
| Actual DeTRC completion | Full80 receipt, selected checkpoint SHA, all phase exits0 and exact357 coverage |
| Common primary metric aligned | Frozen scorer source SHA and explicit positive +0.1 denominator/NumPy rounding |
| Full useful metrics | All/positive NMAE, /gt diagnostic, OBO, OBZ, RMSE, continuous diagnostics; explicit unavailable fields |
| Native distinctions retained | DeTRC common vs common_ori vs native_RC, labels/populations/selection differences |
| Weight and input provenance | Author release vs own training,3seed aggregation, frozen ROI and known-source/label qualifications |
| Negative evidence visible | RC zero/all-population counterexample, all domains, stronger inherited literature caveat |
| Claims bounded | No SOTA/significance/equal-budget claim;6datasets x7external target remains unmet |
| No payload recomputation | Only existing aggregate JSON and small source/receipt hashes; server Python -B arithmetic |
| Private safe immutable artifact | No per-video rows/labels/credentials; new E/server packet with source bindings and SHA transfer proof |

Requested executor gpt-6.1-sol; actual backend identity unattested. This is deterministic evidence integration, not a fresh scientific review.
