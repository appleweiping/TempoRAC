# SPEC — receipt-only RACnet input refresh

Target: eliminate the demonstrated prepare→GenerateGT/decoder repetition after new author assets. Add refresh-inputs and its run.py dispatch; leave initial prepare and all science untouched.
Inputs: inherited CPU_CHECK/READINESS candidate identity;1243 input rows;886 train/valid targets;152 RC-test native_gt within357 scorer rows;100/48030 rawQUVA rows; committed author indices and future QUVA sidecars.
Outputs: updated inputs/prediction_inputs/READINESS plus INPUT_REFRESH.json inheritance receipt. targets.json, score_labels.json and quva_raw_inputs.json retain exact bytes; target NPZ files are never opened. No feature payload SHA/CRC/finite rerun.
Base commit: be033236f0b9d4b0c5ff1903846c94dc8e0f162d (pre-existing unrelated dirty files retained).
Success command: one server CPU-only metadata fixture, then affected-interface fresh Sol review owned by root. Real refresh-inputs runs once only after61 author objects formally complete, and later once on genuinely new completedQUVA outputs.
Scope: no prepare/cpu-check execution, GPU/forward/new downloads, worker/control/Git/global-status writes, or other experiment review.
