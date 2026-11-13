# RC zero-count aggregate supplement

Derived only from existing aggregate identities on the server, without rescoring. RC all152/positive151 leaves one zero-count evaluation case. TempoRAC values average the3 fitted models; sample dispersion remains in the original JSON zero_aggregate. No identities, predictions or labels are included.

| Method | Zero N | Zero normalized error (denominator0.1) | Zero mean absolute rounded error | Zero OBO | Zero OBZ | Zero RMSE |
|---|---:|---:|---:|---:|---:|---:|
| TempoRAC corrected full3 | 1 | 43.333333 | 4.333333 | 0.000000 | 0.000000 | 4.333333 |
| ESCounts released RC weights | 1 | 10.000000 | 1.000000 | 1.000000 | NA | NA |
| RepNet released updated2025 ckpt70 | 1 | 30.000000 | 3.000000 | 0.000000 | 0.000000 | 3.000000 |
| DeTRC TSN/R50 full80 seed0 | 1 | 40.000000 | 4.000000 | 0.000000 | 0.000000 | 4.000000 |

TempoRAC zero mean absolute rounded error4.333333 is worse than ESCounts1.000000, RepNet3.000000 and DeTRC4.000000 in these exact evaluated settings. Positive-only superiority does not extend to zeros. ESCounts zero OBZ/RMSE are NA because its inherited RC aggregate lacks positive OBZ/RMSE; deriving missing fields from per-video evidence is outside this aggregate-only task.
