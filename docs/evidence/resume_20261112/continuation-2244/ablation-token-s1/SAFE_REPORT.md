# Seventh frozen ablation completion

wo_token_s1 completed50epochs and fullRC152/Countix2403/UCF105 at 2026-11-12T23:18:42.254592+00:00. Total new completions7/21. Earlier6arm receipt remains unchanged.

| Benchmark | All/positive N | Primary NMAE | Positive OBO | All-population NMAE | All-population OBO |
|---|---:|---:|---:|---:|---:|
| rc_test | 152/151 | 0.179541087 | 0.682119205 | 0.507307264 | 0.677631579 |
| cx_test | 2403/2403 | 0.237133877 | 0.706200583 | 0.237133877 | 0.706200583 |
| ucf_test | 105/105 | 0.195743970 | 0.714285714 | 0.195743970 | 0.714285714 |

Fixed inference is arithmetic mean of saved original/shift counts before NumPy ties-to-even rounding. All original/shift/fixed-mean, all/positive/zero metrics are retained in the public summary; raw predictions/zero rows remain private. No training, inference, cache construction or old-group rescoring was performed.

The original worker recorded held_after_complete and exited after full evaluations. The exact residual is wo_token_s2,wo_joint_cx_s0,wo_joint_cx_s1,wo_joint_cx_s2. Root owns the ensuing GPU release; this collector changes no job or control.

Token seeds0/1 are complete but seed2 remains unstarted in the residual. No token3 bootstrap, component effectiveness, equivalence, SOTA or independent scientific acceptance follows. Existing test-informed selection and label ambiguity persist. Requested Sol; actual identity unattested; same-family provisional.
