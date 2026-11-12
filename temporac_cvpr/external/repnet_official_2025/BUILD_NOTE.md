# Build note — corrected review draft

Commands run from /root/autodl-tmp/TempoRAC.

| Rung | Exact acceptance command | Status |
|---|---|---|
| F0 CPU population/source check | env/bin/python external-baselines-20261111/repnet-runner/evaluate_repnet.py --config external-baselines-20261111/repnet-runner/evaluation_config.json --output external-baselines-20261111/runs/preflight-v2 --mode preflight | Server preflight exit0: PREFLIGHT_OK357; all selected media and frozen metadata checked |
| F1 source identity | Static exact-definition comparison to immutable notebook; server py_compile | Prior independent literal-definition check passed; corrected three scripts py_compile exit0 |
| F2 restored model/witness | envs/repnet/bin/python external-baselines-20261111/repnet-runner/evaluate_repnet.py --config external-baselines-20261111/repnet-runner/evaluation_config.json --output external-baselines-20261111/runs/witness-v2 --mode witness | After recheck+GPU1 predecessor |
| F3 complete357 | envs/repnet/bin/python external-baselines-20261111/repnet-runner/evaluate_repnet.py --config external-baselines-20261111/repnet-runner/evaluation_config.json --output external-baselines-20261111/runs/repnet-full357-v2 --mode evaluate | After F2 |

R1 coherent TF2.16.1/Keras3.3.3 keeps literal core. R2 exact predecessor receipt is complete_known_source.json, matching worker_gpu1.events.jsonl queue_complete and PID387997 absence; GPU0 forbidden. R3 frozen inclusive UCF crop. R4 qualified Decord full/input metadata checked against OpenCV decode. R5 commands now match argparse.
Explicit --resume reuses only exact contract/inputs plus atomic row/trace integrity; no automatic restart or skipped failed row. No GPU computation has run yet. Same-family review remains provisional.
