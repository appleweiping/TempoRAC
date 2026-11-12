# RepNet scheduling-only amendment after explicit user resume

Source authority: exact root resume_authorization.json (2026-11-12; user excerpt “继续吧”), SHA256 8c564c92701879d8edf9ad6487c080176a74609f3199c7855d032d2bc59c67db. Prior user-stop receipts remain immutable and state full_s0=12/full_s1=13 epochs, incomplete; they are not completion receipts.

Only the resource dependency changes: validate exact resume and historical stop receipts, six terminated PIDs absent, no full_s1/GPU1 worker alive, GPU1 compute ownership empty and at least10GiB free. GPU0 remains reserved for ours. One immediate launcher replaces the impossible full_s1-completion wait. No timer, polling, auto retry or ours training.

Unchanged scientific obligations: literal official core SHAaf46180e5f7671efbd2e95b73630592802ec75752382800f4ba8ad903abc6475; frozen metrics SHA fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0; 152RC+105UCF+100QUVA, same labels/UCFinclusivecrop, zeros/ambiguousinputs retained, official checkpoint SHA6caf32eb28e089987ebc3d9ddf5e1e41c45d1c051eed6fd9635f97845956183d and inference settings unchanged.

Runtime: reuse isolated R/envs/repnet; CPU repair/install can proceed. GPU launch requires bounded fresh Sol closure of this amendment. Then one complete226-variable restore/finite kernel witness followed by full357; no proxy or Countix expansion. Output paths change to R/external-baselines-20261112/runs/ to preserve prior artifacts.

Success command from R:
```sh
env/bin/python -u -B external-baselines-20261112/repnet-runner/run_official_repnet.py --review PATH_TO_COMPLETED_GATE_REVIEW
```
Environment remains not ready until real installed/restore/kernel evidence. Base is reviewed 20261111 candidate confige5e255.../worker46c473...; no paper/Git/shared status modification.

