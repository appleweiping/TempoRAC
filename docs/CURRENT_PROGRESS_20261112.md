# TempoRAC current progress — 2026-11-12

The corrected full-model group is complete: three seeds each finished50epochs and all four frozen full-population evaluations. Last execution completed10:31:44UTC;13:43UTC is later collection time. Training source489dc77b91a21d21215784ff49961edb20dab4fb remains immutable. Aggregation reused saved predictions without retraining, inference or cache rebuilding.

| Acceptance requirement | Current outcome | Evidence / next work |
|---|---|---|
| Inherit research/assets | Pass for this continuation |15 framework/version records (14families ifcollapsed),29 original-protocol runs and frozen caches retained; full-read gaps labelled |
| Corrected full model, matched3seeds | Complete | [Full-group evidence](evidence/resume_20261112/full3-results/GROUP_REPORT.md),77 small raw source files,8280 seed-video records |
| Exact scoring/selection | Numerically verified; scientific qualifications retained |12 full evaluations recomputed from saved predictions;50 contiguous finite epochs/seed; validation checkpoint identity;31/31 cited-value existence precheck |
| Components contribute | Pending paired results | Astra released21new full runs+3single-UCF reuse; bounded launcher review then immutable dispatch |
| Official whole-method performance | Partial | ESCounts release inherited; RepNet updated2025 ckpt-70 full357complete; TransRAC/RACnet task weights unavailable; genuine native training route next |
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
