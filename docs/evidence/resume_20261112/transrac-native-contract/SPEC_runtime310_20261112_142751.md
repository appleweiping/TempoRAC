# TransRAC official native training specification

Target: add a genuine official-code whole-method baseline beyond completed RepNet and ESCounts. Base commit: 68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979. Train fresh counting weights with the official pretrained Swin-T on corrected RC756, seed 1, 200 epochs and logical batch 32. Preserve the native model, target, MSE, AMP, optimizer and schedule. One 4090 simulates four native replica shards, including BatchNorm semantics.

Inputs: official checkout, Swin weight SHA 9950e3be...76acc, existing RepCount CSVs, corrected union manifest, original RGB and fixed UCF ROI/QUVA population. Outputs: prepared input identities, CPU adapter checks, complete epoch/checkpoint receipts, validation-selected best, 357 raw predictions and native/harmonized metrics.

Command to implement: `R/envs/transrac-native310/bin/python R/baselines/transrac_native_v1/run.py --contract R/baselines/transrac_native_v1/RUN_CONTRACT.json --phase train-evaluate --gpu <released-lane>`.

Success requires 200 complete epochs, 4,800 logical optimizer opportunities with AMP skips separately recorded, and all 357 predictions. Implementation and acceptance details follow in IMPLEMENTATION_CONTRACT.md. Runtime readiness is not yet claimed. Root owns implementation review, Git and manual GPU handoff; the existing 21-arm contract is unchanged. No extra GPU pilot, hyperparameter search, Countix run or publication.
Runtime amendment: isolated Python3.10.22 venv inherits existing Torch/CUDA/NumPy read-only; pinned TransRAC dependency overlays belong only to the new prefix. This avoids redundant Torch downloads; all scientific choices remain unchanged.

