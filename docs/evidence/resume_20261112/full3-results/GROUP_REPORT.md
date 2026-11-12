# Frozen corrected-training group closure

All three predeclared seeds completed50 epochs and fixed original-population evaluations. No model execution was performed for this report.

| Dataset | All/positive N | Primary NMAE mean ± sample SD | Positive OBO mean ± sample SD |
|---|---:|---:|---:|
| RepCount-A | 152/151 | 0.141996913 ± 0.016629308 | 0.693156733 ± 0.061176187 |
| Countix | 2403/2403 | 0.243386496 ± 0.007086311 | 0.695242059 ± 0.004105604 |
| UCFRep | 105/105 | 0.194626165 ± 0.012612571 | 0.704761905 ± 0.025197632 |
| QUVA | 100/100 | 0.190626881 ± 0.008050887 | 0.530000000 ± 0.060827625 |

The primary inference is fixed per-seed mean of original and shifted counts, followed by NumPy rounding. SD uses ddof=1 across the three fitted seeds. No ensemble, significance or SOTA claim.

Full seed-level/all-population/zero-count/original/shift/fixed-mean metrics and selected epochs are in group_summary.json. Every saved prediction and negative/disputed row remains in the evidence and per_video_all_seeds.jsonl.

Shared source_sync.json still records completedseeds[0] and group_complete=false from08:57UTC. Actual publication49b30a3 and training489dc77 snapshots are clean; this closure records the newer actual completion without modifying shared state.

Inherited test-informed method choice, evaluation-label conflicts, missing corrected ablations/external comparison/coverage remain unresolved. This is execution/statistical closure only.
