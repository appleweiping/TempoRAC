# TempoRAC current progress — 2026-11-12

Latest actual continuation: [post-storage-maintenance live progress](LIVE_CONTINUATION_20261112.md).
That record and server logs supersede older runtime snapshots below. Completed
results and immutable source/data contracts below remain reusable.

The corrected full-model group is complete: three seeds each finished50epochs and all four frozen full-population evaluations. Last execution completed10:31:44UTC;13:43UTC is later collection time. Training source489dc77b91a21d21215784ff49961edb20dab4fb remains immutable. Aggregation reused saved predictions without retraining, inference or cache rebuilding.

| Acceptance requirement | Current outcome | Evidence / next work |
|---|---|---|
| Inherit research/assets | Pass for this continuation |15 framework/version records (14families ifcollapsed),29 original-protocol runs and frozen caches retained; full-read gaps labelled |
| Corrected full model, matched3seeds | Complete | [Full-group evidence](evidence/resume_20261112/full3-results/GROUP_REPORT.md),77 small raw source files,8280 seed-video records |
| Exact scoring/selection | Numerically verified; scientific qualifications retained |12 full evaluations recomputed from saved predictions;50 contiguous finite epochs/seed; validation checkpoint identity;31/31 cited-value existence precheck |
| Components contribute | Pending paired results | 3/21 new arms completed full training/evaluation; 3single-UCF reuse retained. GPU1 continues the frozen list; remaining GPU0 arms saved exactly during finite TransRAC handoff. |
| Official whole-method performance | Partial | ESCounts release inherited; RepNet updated2025 ckpt-70 full357complete. TransRAC official-code full200 GPU training started and first full epoch/checkpoint verified; final357 pending. RACnet official full-asset acquisition prepared. |
| Six datasets x seven convincing frameworks; SOTA | Unmet |4current evaluation datasets,2actually evaluated officialrelease families; stronger literature and population/resource differences remain visible |
| Observation, stability curves, efficiency | Partial / outstanding |3seed variability complete; mechanism evidence, final hyperparameter curves and feature-plus-inference efficiency remain |
| Final paper | Pending | Historical draft predates these results; claim/citation/evidence audits and all-page rendering required |
| Three-way correspondence | Verify latest receipt | Publication and immutable run sources intentionally differ; live source_sync.json binds exact published artifacts |

## Complete corrected3seed results

Mean +/- sample SD over seeds0/1/2, not best-seed or seed-ensemble. Primary NMAE is mean(abs(round(pred)-gt)/(gt+0.1)) on positive-count videos; NumPy ties-even; fixed original/shift continuous-count average before rounding. OBO is absolute rounded error<=1.

| Full population | All / positive N | NMAE down | OBO up |
|---|---:|---:|---:|
| RepCount-A |152 /151|0.141997 +/-0.016629|0.693157 +/-0.061176|
| Countix |2403 /2403|0.243386 +/-0.007086|0.695242 +/-0.004106|
| UCFRep |105 /105|0.194626 +/-0.012613|0.704762 +/-0.025198|
| QUVA transfer |100 /100|0.190627 +/-0.008051|0.530000 +/-0.060828|

[Exact group_summary](evidence/resume_20261112/full3-results/group_summary.json) retains OBZ/RMSE,/gt,continuous,all/zero-count metrics. RC zero-count video predicts5/4/4; all152 NMAE0.426150 is worse than evaluated ESCounts0.279472. UCF original-only NMAE0.190705 versus fixedtwo-view0.194626 does not justify changing inference on test. No universal view-benefit/all-population-superiority claim.

Training-only known-source correction retains every evaluation row. Historical test-informed design,2conflicting RC labels, unknown overlap/pretraining exposure remain unresolved. Old29 and corrected results stay separate. Per-target checkpoint uses original-view validation(mae_p,-obo_p); QUVAusesRC. This is one shared trajectory with target-selected checkpoints, not one common final weight for all domains.

## Official released weights actually evaluated

| Exact released weight | RCpositive NMAE / OBO | UCF NMAE / OBO | QUVA NMAE / OBO |
|---|---:|---:|---:|
| ESCounts acquiredRC checkpoint |0.215098 /0.556291|0.341636 /0.466667|0.299129 /0.300000|
| RepNet updated2025 ckpt-70 |0.287127 /0.410596|0.311727 /0.571429|0.240458 /0.520000|

These are author weights evaluated by us, not our retraining. ESCountsUCF/QUVA are RC-weight transfer, not strongestnative target-trained models. RepNet is updated2025, not original2020. Training budgets/populations differ; no significance/globalSOTA claim. Countix is outside RepNet357. [RepNet evidence](evidence/resume_20261112/repnet-full357/FULL357_RECEIPT.json) preserves196 adopted+161 new predictions. Stronger PAMS/CountLLM/Context/Sight&Sound literature rows and their protocol qualifications remain in the [Astra decision](evidence/resume_20261112/group-end-astra/DECISION.md).

## Actual next21 dispatch

At14:04:28UTC both finite queues started from launcher publication0d99761d2d315cd85e7d325b4a921007f060068a and immutable training489dc77. GPU0wo_pyramid_s0 worker435832/trainer435851; GPU1wo_shift_s0 worker435833/trainer435853. At14:08:25UTC each had a finite completed epoch and an actual persisted latest/best checkpoint; exactcfg, contract, source,cwd and GPUownership matched. [Runtime receipt](evidence/resume_20261112/ablation-runtime/FIRST_DURABLE_STATUS.json) contains observed progress, not a completed-run or scientific verdict. Full50epochs and all fixed evaluation continue before each successor. Reviewclosed afterone recovery-onlyfix; no trainingcore/cache/environment changes.
## Fixed next group and remaining deliverables

[RUN_CONTRACT](evidence/resume_20261112/group-end-astra/RUN_CONTRACT.json), SHAdc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167: remaining24 inheritedspecs become21newruns+3reusedsingle-UCF. Matchedmainreference is correctedfull3. No new matrix/method/metric duringexecution.

- GPU0: wo_pyramid s0-s2, wo_depth s0-s2, wo_balance s0-s2, wo_joint_rc s0-s2.
- GPU1: wo_shift s0-s2, wo_token s0-s2, wo_joint_cx s0-s2.
- Original50/80epochs,1320newepochs total. Joint arms fullRC152/CX2403/UCF105; singlearms theirfulltarget; no extraQUVAablation.
- Existing single-UCF80epoch checkpoints/predictions reused with historicalsource/environment/cachebinding gap disclosed. Singlearms omit token/repetition too: composite-regime contrasts. wo_balance changes epochschedule; wo_pyramid changesrepresentation/parameterbudget.
- Manualhold onlyaftercurrentarmfulltraining+allevaluation+receipt. Separatelyfrozenreadyofficialbaseline maytakeone lane, othercontinues. Explicitresumeusesexactremaininglist. Failurestops lane; noautoretry/scorepruning.

Astra requestedgpt-6-astra/xhigh; actualbackendunattested; samefamilyprovisional. This boundedgroupendrelease isnotultradeep/submissionaudit. HistoricalAugustintegrityFAILpreservedwitholdscope; currentfullscientificintegrityassuranceremainsoutstanding.

[Official baseline availability](evidence/resume_20261112/next-baselines/AVAILABILITY.md) establishes nextconcreteacquisition/trainingrequirements. Native TransRAC fulltrainingcontractpreparation proceeds independently; it doesnotalter21group. Needed afterward: complete officialperformance coverage, pairedcomponentjudgment, observation/stability/efficiency, qualifiedadditionaldatasets, finalwritingandrendering. Actualevidence determines whetherSOTA/componenttargetsaremet.

Latestuserexplicitlyrequestedtimedadvancement and reiteratedno rollback,wastefultoys orpartialexperiments; [verbatim record](USER_REQUIREMENTS_20261112.md). Currentapp prohibits completion-trackingautomations, so nonewascreated; finiteauthorizedserverworkers execute thefrozenlist. Allcompute staysunder /root/autodl-tmp/TempoRAC; E:forediting/control/evidence. No C:/D:projectwrites,otherprojectinterruption,submission/mainmerge orrawnegativeevidencedeletion.


## Separate official TransRAC full-training contract

The [Astra contract](evidence/resume_20261112/transrac-native-contract/RUN_CONTRACT.json), SHA1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a, freezes one seed1 x200epochs with native official TransRAC68bdd4d, correctedRC756/val131 and finalRC152+UCF105+QUVA100. It uses official pretrained Swin-T and fresh counting weights, preserving the source raw loader, model, density target/loss, scheduler and internal autocast. Single-GPU virtualfour replicas preserve the native shardBatchNorm semantics. Common validationselection and modernenvironment are labelled adaptations; this is official-code retraining, not authorreleasedtaskweights or paper-number exactreproduction.

Original complete cycleCSVs and all1244 requiredinput identities are located; official127414861-bytebackbone nowexactSHA9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc; isolateddependencyoverlay installed, full1244RGBcache andrequiredCPUchecks/review inprogress. No TransRACproductionrun is claimed. Onlyafter these actualpreconditions and one boundedcodecheck may root handoveratmostoneGPU at a completeablationarmboundary. The current21trainingmatrix andotherlane remainunchanged. All prelaunchprecisionclarifications, including the corrected mistakenFP32 inference, remain archived; no experimentused that mistakeninterpretation.

Runtime-only amendment before launch: native310 isolated overlay reads existing Python3.10.22/Torch2.5.1+cu124/CUDA/NumPy packages without modifying the active environment. This avoids duplicate multi-GB downloads; official scientific computation,200epochs and input/selection contract remain fixed. Historical311 and precision drafts are archived.


Additionaldatasetavailability: [boundedofficialsourceevidence](evidence/resume_20261112/additional-dataset-availability/AVAILABILITY.md) findsYTSegments100officialcloudunreachable andActCount172withoutverifiedpublicrelease. Neither countsasreadyorindependent; no inputsadded toongoinggroups. [Officialbackbone/inputreceipt](evidence/resume_20261112/transrac-native-assets/ASSET_READINESS.json) records exactreadiness, including the preservedtransferfault and9MiBtargetedrepair.

## Full inputs and first completed ablation — 2026-11-12 15:00 UTC

The first new arm, `wo_pyramid_s0`, completed all 50 epochs and full RC152,
Countix2403 and UCF105 evaluation at 14:58:15 UTC. Collection verified every
completion-bound artifact and both-view prediction membership, with zero transfer
SHA mismatches. This is 1/21 completed new arms, not a paired component verdict.
The same worker advanced to `wo_pyramid_s1` (PID482574); GPU1 continued
`wo_shift_s0` (PID435853). No run was stopped or moved to make room for TransRAC.

TransRAC's full 1244-video RGB input cache is complete. Input receipt SHA256 is
`9341a1b74120c5fd12172a1c8af1ecdba32403db32a6392c86a6343ba3e6882d`.
The normal final server CPU preflight passed: 756/131/152 RC inputs plus UCF105
and QUVA100, exact raw-loader sample/target comparisons, 183 exact official
backbone state entries, isolated environment and exact source binding. It did
not run a model forward or establish GPU runtime or scientific performance.
The one prelaunch worker-collation defect was repaired without changing target
values or cached RGB bytes; the actual epoch68 trigger and four other target
cases pass. The final bounded review is clean with zero engineering blockers and same-family
provisional assurance. Exact source publication precedes the root's manual handoff after an entire subsequent ablation arm is complete.

## Saved costs and next official assets — 2026-11-12

[Saved-log cost accounting](evidence/resume_20261112/completed-cost-accounting/COST_REPORT.md)
reuses all 150 completed epoch records and the actual train/evaluation process
bounds. Corrected full3 training processes sum to 18,484.737496 seconds
(5.134649 GPU-assigned wall-hours); the 12 full evaluation processes sum to
186.187768 seconds. These exclude upstream feature extraction. RepNet's saved
357-video timing is separated from the 161 newly evaluated videos in its last
process. End-to-end matched efficiency remains partial; no fair speedup follows
from these differently scoped timings. No experiment was repeated to obtain them.

The next official RACnet asset gate completed one bounded pass: 77,936,474
response bytes, eight distinct range requests, 53.72 seconds, no GPU or model.
[Public-safe provenance](evidence/resume_20261112/racnet-author-assets/PUBLIC_G1_PROVENANCE.json)
binds the preserved private raw receipts. Signed URL values and original arrays
are excluded from publication. The UCF archive contains all 105 NPZ members;
five explicit capitalization aliases account for the initial name mismatch.
Whole-input content/ROI parity and complete RC target arrays are not yet verified.

Astra's [single post-gate decision](evidence/resume_20261112/racnet-author-assets/G1_AMENDMENT.md)
releases a finite acquisition of the 61 fixed author RC/UCF objects, reusing
verified ranges, with a 48-hour limit, 384-GiB asset cap and 128-GiB free-space
reserve. Deployment is pending its concrete transport implementation; this is
not a training release. The future native labeled training population is 755:
the original missing-label row is retained in source evidence and explicitly
ineligible for optimization. No evaluation row is excluded, and the current
TempoRAC/TransRAC populations and ongoing experiments are unchanged.

## First launch fault and bounded recovery — 2026-11-12 16:10 UTC

GPU0 completed `wo_pyramid_s1` (50 epochs and all three frozen evaluations)
and the worker held only after that whole arm. GPU1 was left running.
TransRAC's first launch at 15:51 UTC exited before START/CUDA/any optimizer
step: JSON serialized package-identity tuples as lists, so Python's direct
identity comparison rejected unchanged environment facts. The initial log,
receipt and failure remain in the runtime evidence; no checkpoint was lost.

A one-line JSON-native list fix and targeted independent closure passed.
All 1244 cached inputs, official backbone, source, configuration and previously
completed CPU checks are reused. The rebound preflight records the original
receipt and exact code delta; no model or full input checks were repeated.
The new five-file map, preflight and root engineering gate are published
before the fresh first optimization attempt. Full 200-epoch training and final
357-video evaluation remain pending actual execution, not claimed complete.

## Formal TransRAC training now executing — 2026-11-12 16:16 UTC

The corrected driver publication is `39dcafb8d77de9e43f9b61ae0346d726eddd18b5`,
verified local/GitHub/server. The fresh first optimization attempt started on
GPU0 at 16:12:43 UTC (PID490003); the earlier failed attempt had zero steps.
[Durable runtime evidence](evidence/resume_20261112/transrac-first-durable/RUNTIME_RECEIPT.json)
verifies epoch0 covered all756 training rows exactly once,24 logical and actual
optimizer updates, finite gradients, zero overflow skips, and a persisted
checkpoint SHA `0467db6becceeee6a9fd6ce561d5c6b643ab42f0f58dd10d05221c0e2aebf05b`.
It is a full formal run, not a discarded pilot. The remaining199 training epochs
are approximately9.70 GPU-hours at the first actual epoch's175.41seconds pace;
validation cost remains unmeasured until its scheduled epoch51 start.
The full200 and final357 are still incomplete.

New ablation completion records now cover `wo_pyramid_s0`, `wo_pyramid_s1`
and `wo_shift_s0`. GPU1 continues `wo_shift_s1`; its worker and frozen protocol
were not changed. The exact ten residual GPU0 arms remain saved for explicit
continuation after the finite baseline. No component-effect conclusion is drawn
until its paired multiseed group is complete.
## Complete official RACnet asset acquisition deployment

The single transport repair batch closes rename/state recovery, terminal-buffer
accounting and nested extraction progress. Eight targeted server-CPU recovery
cases and the existing regression pass; original failed fixture and prior
negative review are retained. This is narrow operational closure with
same-family provisional assurance, not scientific or regular-tier acceptance.
The requested reviewer effort was high, so no xhigh-floor fulfillment is claimed.

The worker is bound to61 official immutable author objects totaling128,629,792,634
bytes, with77,936,474 already verified bytes reused. The fixed48-hour, one-CPU,
no-GPU/no-retry transfer follows the complete dataset manifest. A fresh resource
check and root gate precede launch; actual launch evidence is recorded separately.
Private raw HTTP receipts/signed URLs, arrays and downloads stay off GitHub.
Published CONTROL contains only122 static public canonical/mirror URLs, all
verified without credential/query values, and is byte-identical to the runtime
control. Current TransRAC and ablation model contracts are unchanged.

At16:25:24UTC the finite full RACnet author-feature acquisition actually started
(PID491938). At16:27:21UTC,369,098,752 new bytes were durably stored in addition
to77,936,474 reused bytes; no object is called complete until its full SHA matches.
The first complete native RC NPZ contains actual author `img_feature`, `gt_tsm`
and `length`; CRC/finite checks pass and original arrays remain unchanged.
The literal `frames` key is absent, so that naming adapter remains explicit.
[Runtime receipt](evidence/resume_20261112/racnet-acquisition-runtime/ACQUISITION_RUNTIME_RECEIPT.json)
verifies CPU26, numericthreads1, empty CUDA visibility and the fixed deadline.
This first member does not prove full population/input parity or release training;
the same worker continues the entire frozen61-object acquisition.

## Transport recovery and next official variant — 2026-11-12

The RACnet asset worker stopped once on a TLS handshake timeout after preserving
704,643,072 newly received bytes; this was not a model/schema failure. One
explicit continuation kept the exact code/control, original48-hour deadline,
verified G1 ranges and all earlier bytes. At16:41:06UTC PID493360 had completed
seven new32-MiB requests from the prior frontier:234,881,024 additional durable
bytes, cumulative939,524,096. No automatic retry was installed. The first failed
attempt and continuation evidence remain separate in the
[resume receipt](evidence/resume_20261112/racnet-acquisition-runtime/resume-attempt2/RESUME_OBSERVATION.json).

Astra closed the existing [DeTRC variant ambiguity](evidence/resume_20261112/detrc-official-route/ROUTE_DECISION.md):
the released implementation is TSN/ResNet50 with2048-D features and80epochs,
not the paper's strongest ViT variant. Source identity67/67 files is verified;
full weights/features/runtime and named mechanical interface fixes remain
prerequisites. No DeTRC run is released or counted as completed. This preparatory
work changes none of the current training/evaluation contracts.
