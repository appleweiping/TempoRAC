# Frozen ablation live collection — 2026-11-12 17:55 UTC

The existing GPU1 finite worker completed wo_shift_s1 at 17:49:31UTC with 50 training epochs and all three full evaluations. The new-run completion count is 4/21 (wo_pyramid_s0/s1 and wo_shift_s0/s1); 3 historical single-UCF runs remain reused under the existing qualification. No previously archived arm was downloaded again.

At 17:55:33UTC, GPU1 worker435833 and trainer500464 were alive on wo_shift_s2, with 2/50 completed log epochs and an existing epoch 0 durable receipt. Its next frozen arm is wo_token_s0. GPU0's ablation worker remains held after wo_pyramid_s1, with the exact 10 pending arms retained. TransRAC PID490003 continued on GPU0; this collection changed no worker, queue, training or baseline.

For wo_shift_s1, 50 continuous epochs 0–49 and zero nonfinite steps were verified against the frozen config/provenance/source/manifest. All completion-listed artifacts, including selected checkpoints, matched their SHA256. The original and shifted prediction keys exactly match the frozen full populations. Validation-only checkpoint selection was verified. All stored original/shift/fixed-mean metrics were recomputed from existing predictions on the server and matched within 1e-10.

| Full test population | Positive count n | Primary MAE_p↓ | Positive OBO↑ | All-population MAE↓ | All-population RMSE↓ |
|---|---:|---:|---:|---:|---:|
| RepCount-A 152 |151|0.182645|0.635762|0.444602|4.761413|
| Countix 2403 |2403|0.248519|0.690387|0.248519|3.732972|
| UCFRep 105 |105|0.212576|0.685714|0.212576|2.550443|

These are raw single-arm results using the frozen arithmetic mean of original and shifted counts before ties-to-even rounding. The RepCount-A zero-count row is retained: fixed-mean continuous prediction 4.1979999542, rounded false count 4. All raw predictions, original/shift metrics, negative cases and logs remain in private evidence. The 3 matched shift seeds and complete ablation group are unfinished; no component effectiveness, significance or ranking judgment is made.

Private collection receipt SHA256: 6c4baadab84608bbe64cbc29a6d3d946c917574ea05083907ae72cd4cfbb2926. All 42 manifest files were transferred to E: and independently matched in byte size and SHA256. No model was loaded and no training/inference was launched. This safe summary contains no raw dataset, checkpoint or prediction file.

ARIS pin: f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Full pinned monitor-experiment and analyze-results skills were read and installed hashes matched. Requested executor: gpt-6.1-sol; actual backend unattested. Same-family semantic assurance remains provisional.

