# RepNet official full-population adapter specification

Target: author ckpt-70 updated2025, literal Keras3 core from google-research commit60d48fc1b71b6126462136cad441dd76da7460d0.
Inputs: UNION rc_test152 +ucf_test105; QUVA test100; all357 records including zero and ambiguous RepCount inputs. UCF zero-based inclusive crop/end clamp matches frozen feature metadata. Full decode and crop lengths must agree with qualified metadata.
Outputs: resumable atomic per-video JSON+traces; raw continuous/native-derived and frozen harmonized metrics; complete restoration/source/input/runtime receipts.
Base: official notebook above; adapter directory has no Git repository. No model changes, tuning, training or publication.
Runtime: dedicated server envs/repnet, TF2.16.1/Keras3.3.3/Python3.10. GPU1 only after full_s1 completion receipt, queue_complete, worker387997 gone and no compute owner.
Exact command from project root:
```sh
envs/repnet/bin/python external-baselines-20261111/repnet-runner/evaluate_repnet.py --config external-baselines-20261111/repnet-runner/evaluation_config.json --output external-baselines-20261111/runs/repnet-full357-v2 --mode evaluate
```
Add --resume only to continue the same exact frozen output, after diagnosing the prior failure. Fresh Sol recheck precedes GPU witness/evaluation.
