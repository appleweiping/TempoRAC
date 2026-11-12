# Bounded single-UCF identity inventory — 2026-11-12

The three existing `runs/frozen_a/wo_joint_ucf_s0,s1,s2` artifacts match the observed effective identity of the already frozen single-UCF arm. They are **potential reuse candidates with explicit historical provenance qualifications**. This inventory neither releases an arm nor orders retraining. Root retains the evidence for one group-end decision. It is not a component-effectiveness or scientific acceptance verdict.

Server observation: **2026-11-12 08:32:03 UTC**, `/root/autodl-tmp/TempoRAC`. Requested model `gpt-6.1-sol`; actual backend/effort unattested; session `/root/ucf_reuse_inventory_20261112`. No new ARIS stage or reviewer invoked. Inherited15-framework/29-run evidence was reused, not regenerated.

## Acceptance and evidence

The pre-inventory checklist is preserved in `ACCEPTANCE_BEFORE.md`.

| Acceptance item | Result | Evidence |
|---|---|---|
| Preserve old assets and current group | Pass within this inventory | Read-only server script; all new outputs confined to this E: directory; no source, config, matrix, shared status or source_sync mutation. |
| Identify three complete runs | Pass for saved-artifact completeness | Each has80 contiguous epoch records0–79, no logged nonfinite steps, checkpoint/config/results/predictions. Full file hashes in `evidence.json`. |
| Match optimization/validation/evaluation population | Pass for recorded inputs | Exact ordered record equality across old/corrected manifests: UCF331/90/105; same media SHA fields, labels, bounds, frame ranges, split/group/action fields and feature keys. |
| Match frozen configuration and relevant method semantics | Pass for observed configuration/source | All three complete cfg objects, seeds, feature roots and role strings match their frozen targets without any cfg difference. Source details below. |
| Establish complete contemporaneous historical provenance | Partial | Current saved hashes match inherited Nov11 provenance; run-start source/runtime/cache binding was not saved in the examined run files. Concrete limitations below. |
| Preserve release and scientific boundaries | Pass | No training, inference, metric recomputation, GPU operation or checkpoint load; no effect/SOTA verdict, release or forced-retrain decision. |

## Exact identities and complete populations

Old manifest SHA256: `bd171452197b74435e6fecfd19332f3e0e7c428e48b1f3a8aeb2b77c35894a13`.
Corrected manifest SHA256: `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`.
Frozen27 specification SHA256: `5e6750b0ff6a894be85c171e5ae31c150e9cb5288a3cb9ab4df8f0f5984babe4`.

Only the two declared RC and two CX optimization keys differ; no UCF row changes. All seeds retain90 validation and105 test prediction keys in both original and shifted views. Each candidate config/checkpoint/prediction/result hash matches the existing `resume20261111analysis/provenance.json` receipt. Current log/stdout hashes are additionally recorded; those files are not individual entries in that inherited provenance map.

| Seed | Completed epochs | Saved best epoch index | Checkpoint SHA256 |
|---|---:|---:|---|
| 0 | 80 | 67 | `2ddcbe5d70f44b1f6ace34ac68f2c69c1841b16877633f4dba094c5334aeb49d` |
| 1 | 80 | 36 | `bc87921d4f01f3a1e72c80a9180f1d25666f9fdf6953abd227954d059a3326f9` |
| 2 | 80 | 32 | `a4e00b2648736e4b52a668fd997bfcbc485fa1e98810572fad058c19caa488ca` |

These are saved selection identities, not newly evaluated outcomes. No performance table or cross-arm comparison was computed.

## Effective training and source semantics

The comparison target is each corresponding record of `resume20261111analysis/known_source_v1/frozen_27_spec.json`, whose single-UCF entries were copied from these saved configs. This proves consistency with the frozen copied specification; it does not constitute an independent method-design endorsement.

The recorded single-UCF model uses concat rates2/4/8, block8 plus block12 features, pool4, width256, three temporal layers,3878020 parameters; batch8,80 epochs, AdamW lr0.0002/wd0.05, warmup3/cosine, gradient clip1, crop probability0.5/min0.6, shift probability0.5/period threshold8 and two-view inference. Its active objective is density MSE scaled100 with count weight0; inactive default period/tempogram fields do not establish active components. There is no decoder init_from or MLLM configuration. The source seeds Python/NumPy/Torch and independent crop/shift/flip streams before fresh model construction;80×ceil(331/8)=3360 nominal optimizer updates, with no logged skipped nonfinite steps.

**The frozen single-UCF arm has neither dataset-token domains/n_domains nor balanced repeat.** It samples331 entries per epoch once. The full union arm includes a dataset token and repetition and has3878788 parameters. This inventory establishes equivalence to the existing frozen single-UCF arm, not a claim that joint population is the only difference from full. It does not alter either specification.

The shared manifest is filtered by role at the beginning of `data.load_videos`; non-UCF rows are skipped before feature/metadata loading. Therefore removal of RC/CX rows does not alter this arm's UCF order, metadata selection, per-video label scaling or feature keys. There is no mixed-dataset/global normalization pass in that loader. Both feature roots have all526 UCF metadata files; metadata digests are recorded. Tensor payloads were not reread; existing cache/header receipts remain inherited.

The old frozen source and active immutable489dc77 snapshot have identical bytes for `data.py`, `models.py`, `metrics.py`, `eval_saved.py` and `launch.py`. `train.py` differs: old SHA `8454950b20ab7f34331b24f820f8d58126289c0f8bf7684016540903310ae4c7`, active SHA `33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4`. All existing top-level helper ASTs are identical, including loading rates, augmentation, batching, loss, EvalLoader, predict and score. The complete stored diff shows persistence/resume/overwrite guards and epoch logging changes in main; model/optimizer/selection/training computation remains unchanged. The already closed durability review is inherited from `docs/evidence/resume_20261112/checkpoint/REVIEW.md`; no fixture or review was repeated. Its CPU parity evidence does not establish CUDA bitwise equality.

Original-view validation selection remains the `(mae_p,-obo_p)` key, with saved full validation/test original+shift predictions and fixed arithmetic-mean inference. No new prediction, metric recomputation or checkpoint deserialization was performed here.

## Concrete qualifications and handoff

The examined historical run configs do not bind a contemporaneous run-start source commit/digest, complete manifest/cache/encoder payload digests, or full dependency/kernel environment. Nov11 source snapshots/provenance/preflight are later observations, and current matching hashes cannot manufacture absent run-start receipts. Checkpoint internals were deliberately not loaded, so internal cfg/epoch/state associations are not newly certified. A hypothetical fresh CUDA execution under the durability trainer is not claimed bitwise identical.

These are provenance limits, not an observed cfg/population/source-semantic mismatch. Preserve these three runs as potential reuse candidates. Do not launch replacement training solely to remove these qualifications, change the active group, or treat this report as release authorization. Group-end interpretation and any reuse decision remain with root under the frozen contract.

Machine-readable evidence: `evidence.json` contains raw server identities, complete saved configs/targets, source diff, role keys and all file hashes; `identity_assessment.json` separates reused/matched/potential/not-proven fields. `inventory_server.py` is the exact read-only stdlib-only server script. Original artifacts remain in place.
