# RepNet bounded closure review — 2026-11-11

Verdict: **CLEARED_FOR_SERVER_RUNTIME_GATE, same-family provisional**. The five original static pre-run blockers are closed in the exact candidate listed below. No remaining concrete launch or result-integrity blocker was found within this bounded recheck. This releases the review gate to the author's already-authorized server startup checks, restored-weight/kernel witness, then full357 evaluation under the existing GPU1 predecessor gate. It does not assert that the predecessor condition is already true, that the environment installation has succeeded, or that any model/evaluation has run.

Reviewer session `/root/repnet_official_review`; requested `gpt-6.1-sol`; actual backend/model/effort independently unattested; OpenAI same-family semantic assurance **provisional**. No local tests, installation, model import, server execution or source/run edits were performed by this reviewer. Only this review artifact was written.

| Original blocker | Closure | Direct final-source evidence |
|---|---|---|
| R1: legacy environment conflicts with literal Keras3 core | Closed, static | env-spec pins Python3.10, TF2.16.1, Keras3.3.3 and NumPy1.26.4. load_model155–174 verifies exact versions and preserves the literal core. Setup creates only the isolated prefix and does not import TensorFlow. Actual install/restore/kernel evidence remains the next runtime gate. |
| R2: GPU0 and missing predecessor condition | Closed, static | Config selects GPU1; gpu_check105–145 rejects other GPUs, verifies full_s1/complete_known_source.json run/GPU/50epochs/fixed152+2403+105+100 populations, matching queue_complete/worker387997/runs event, absent PID and matching full_s1 workers, then GPU1 free memory and empty compute-app ownership. Both witness326–327 and evaluation202–203 invoke it before TensorFlow/core import. finite worker waits for the identified predecessor process exit, launches one witness then one evaluation, and exits on failure. No training worker is interrupted. |
| R3: UCF whole-video mismatch | Closed, static | Runner229–239 uses the frozen zero-based inclusive range and existing end clamp, before get_counts. It verifies exact input length and saves full length/requested/effective range/clamp/metadata identity. No count or boundary tuning is introduced. |
| R4: incomplete decoding accepted | Closed, static | Preflight70–85 freezes qualified metadata identities and full/input lengths; runner225–239 requires OpenCV full decode length equal frozen full-video/len_reader metadata and cropped length equal frozen frames_decoded. Mismatch aborts before that row's inference. |
| R5: invalid documented CLI | Closed | SPEC/BUILD commands now supply --config, --output and --mode preflight/witness/evaluate. Witness and evaluation use different output directories; finite worker52–56 constructs that same interface. |

Package compatibility was checked against primary versioned source: [TF2.16.1 requirements](https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/tools/pip_package/setup.py) allow Keras>=3 and NumPy1.26.4 on Python3.10; [Keras3.3.3 setup](https://github.com/keras-team/keras/blob/v3.3.3/setup.py) supports Python>=3.9. Its [variable initializer API](https://github.com/keras-team/keras/blob/v3.3.3/keras/src/backend/common/variables.py) matches the literal author's constructors. These resolve the declared API mismatch; they do not substitute for runtime evidence.

## Bounded additional launch/integrity check

The new finite worker initially supplied KERAS_BACKEND both via **spec['env'] and as an explicit keyword, which would raise TypeError before child launch. The author removed only the redundant keyword. Final worker49–50 now has no duplicate; existing config source identity and FINAL_DRAFT_RECEIPT were refreshed consistently. The final six config source entries and all ten final-draft file receipt entries match the actual local files.

Atomic/resume paths were read for concrete result-integrity faults only. The runner writes a temporary trace then renames it before committing its per-row JSON. Explicit --resume requires unchanged config/input/source receipt, dataset/split/video/source identity, target count, finite prediction and trace checksum. A missing/uncommitted record is recomputed; a committed record with missing/corrupt trace aborts. No automatic retry, failed-row omission or prediction stub was added. I found no additional concrete blocker in those paths for this launch.

The official core and frozen metrics hashes are unchanged from the first review. The fixed inference settings, all152+105+100 records, zero count, disputed inputs, continuous/native-derived separation and frozen harmonized scores remain as reviewed. No matrix/design change was examined or requested.

## Exact final file identities
| File | SHA-256 |
|---|---|
| evaluation_config.json | `e5e25526d67a3dee4793695258d1c672413f54f48a852915447c13dc834b19ba` |
| evaluate_repnet.py | `71443ff160081b9d06c70cef483e5c55083815b2f959245f1bd4d8118343381c` |
| repnet_official_core.py | `af46180e5f7671efbd2e95b73630592802ec75752382800f4ba8ad903abc6475` |
| frozen_metrics.py | `fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0` |
| env-spec.json | `3a85c459413a5bd0f282a99fc3acd458096c8175c5198f334cd4404fd8746b39` |
| setup_repnet_env.py | `95f6be9a024e79b05294925b2c5a752bb5708eb11f2c2978a6e117e70e45b49b` |
| finite_repnet_worker.py | `46c473e8dc3b62bf23c62bd638588b024383871b57fa45f6649ffde8750c3362` |
| SPEC.md | `7834ce7ae5e046f6450208e2a619d5a47cea1beddc7fe8a7c402eb635b6b2c8c` |
| ASSUMPTIONS.md | `1a50f25b2508610ef590ee43afa5306c873b35aa7dc219aa6ada4092e783f8db` |
| RUN_CONTRACT.md | `7c9e9317865d8feffd4763fe3c7e981fab99f96878a31c2556c738cbce6774fd` |
The final config SHA is e5e25526d67a3dee4793695258d1c672413f54f48a852915447c13dc834b19ba. Earlier executor PREFLIGHT_OK357/py_compile evidence predates the one-line wrapper fix and source-hash refresh; the final receipt accurately leaves the startup recheck outstanding. Upload the exact corrected candidate and perform the already-authorized server startup recheck before dispatch. This is an execution-provenance step, not a new scientific stage or additional audit campaign.

Scope covered: original review, FINAL_DRAFT_RECEIPT, RUN_CONTRACT, updated runner/config/env/setup/finite worker and SPEC/BUILD/assumption commands; final local hashes. Runtime/package installation status, live server predecessor/ownership, checkpoint restoration, finite kernel outputs and full performance are not claimed by this static closure. No SOTA, project completion or accepted semantic verdict is given.
