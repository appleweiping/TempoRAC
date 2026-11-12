# TransRAC native executor implementation

Target: implement frozen RUN_CONTRACT SHA 1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a without changing official 68bdd4d model/target. Inputs: asset executor's immutable1244-row manifest/CSVs, RGB, exact Swin and canonical scorer SHA fce39c7. Outputs: input/CPU receipts, atomic epochs, validation selection, native/harmonized full357 predictions/metrics.

Success: use BUILD_NOTE's complete scoped temp/cache invocation of `/root/autodl-tmp/TempoRAC/envs/transrac-native310/bin/python /root/autodl-tmp/TempoRAC/baselines/transrac_native_v1/run.py --phase train-evaluate --gpu N` after root's manual lane handoff and review receipt.

Base: separate non-Git directory; inherited publication4c5cc763ce9c33148a118ee574cca3baacc5db05. Official source unmodified. Python3.10 overlay reads existing exact Torch/CUDA/NumPy; own pinned packages; no base writes. Explicit full RGB prepare runs once/reuses verified entries. Root owns publication/review/GPU launch. No pilot/Countix/newseed/model substitute/retry. Every failure preserves its attempt. Science remains frozen by contract/ledger.
