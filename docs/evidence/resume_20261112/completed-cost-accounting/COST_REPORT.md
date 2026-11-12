# Completed-run saved-log cost accounting

All numeric aggregation ran on server CPU using existing small logs/configs. No model/checkpoint loading, inference, training or cache construction occurred. No active-ablation performance was read.

| Completed run | Epochs | Logged train-loop sum (s) | Training process wall (s) | Four full-eval processes (s) | Logged PyTorch peak (GiB) |
|---|---:|---:|---:|---:|---:|
| full_s0 |50|5715.4|6191.682014|63.062295|1.71|
| full_s1 |50|5656.5|6155.423253|63.065095|1.71|
| full_s2 |50|5645.5|6137.632229|60.060378|1.71|

Total:150 epochs; logged training-loop 17017.4s; three training processes 18484.737496s (5.134649 GPU-assigned wall-hours);12 full-eval processes 186.187768s. These are decoder/cache-based costs, excluding upstream RGB/feature extraction.

| RepNet saved timing scope | N | Decode sum (s) | Official inference + trace sum (s) |
|---|---:|---:|---:|
| all357 |357|154.071177|307.231418|
| reused196 |196|111.456832|177.818738|
| new161 |161|42.614345|129.412680|

RepNet v4 evaluation child wall 186.527697s covers only161 new inputs plus196-record adoption/setup/IO. Full357 sums combine retained source-process timings; no fresh357 benchmark was run. TF allocator peak 4898042112bytes. All failed attempts and witness intervals are retained in cost_table.json.

ESCounts has rounded progress markers only: RC validation+test resumed segments, UCF last101/105 at4.2min, QUVA last81/100 at7.3min. Full job and per-video times are unavailable; no exact full-population total is inferred. Historical block8 cache progress and source shell commands are retained; partial rounded progress is not a complete preprocessing bill.

Timing, memory and storage qualifications are explicit in cost_table.json. Decoder parameter count is3,878,788 from saved config; checkpoint sizes are stat-only and their existing receipt hashes are reused. Full backbone parameters/FLOPs were not recomputed. No monetary cost is stated without a billing rate.

The single remaining efficiency matrix need: on one documented GPU/software environment, all actually compared full methods on the same complete357 RC152/UCF105/QUVA100 inputs, retain each official frozen inference view/window setting; report RGB decode/preprocess, feature/backbone, decoder/postprocess, total cold-input end-to-end wall and steady-state wall separately, per-dataset N and frame workloads, identical I/O inclusion and warm-up rule, measured peak process/allocator memory, parameter/FLOP scope, and one-time cache-build/storage cost with an explicit reuse count. TempoRAC requires both final-block and block8 features plus fixed two-view decoder; RepNet requires all official stride searches; ESCounts requires all official windows. Existing logs cannot provide this matched comparison; no new benchmark is authorized by this accounting.

Acceptance: saved timing/epoch/process/adoption evidence PASS; cache/end-to-end/fair conference efficiency PARTIAL; no speedup claim supported. This is evidence gathering, not a fresh semantic/integrity review.
