# Finite full RACnet runner

Source scripts/run_racnet_missing5_full.py isoutsideW andthefrozen6coreidentity. Itstarts onlybyexplicitrootinvocation:

`/root/autodl-tmp/TempoRAC/envs/racnet-native310/bin/python -B <immutable runner path> --release <actual root finite release JSON>`

Rootrelease fields: released_by=root; authorized_sequence=[extract-quva,refresh-inputs,train,evaluate]; start_before_epoch(onlyinitialentry); runner_sha256; candidate_identity(exact6core); contract_sha256(parent77),effective_contract_sha256(38),amendment_sha256(c783),correction_sha256(b414); authorized_gpu_count1,gpu_index,gpu_uuid; current_arm_complete_receipt,immutable_publication_receipt,fresh_review_receipt(eachpath/SHA); fresh_review_status; gates source_runtime/CPU_interfaces/lane/resources PASS; pending_author_bytes0; additional_bytes_reserved. Publication/reviewJSON also bindrunner_sha256 andcandidate_identity; reviewverdictPASS/PROVISIONAL_PASS. Noactualreleasefilecreated here.

Eachstage receipt inheritsactualrootfacts, rechecks exactsource andresources, andsets expires_epoch=now+600 solelyforexistingstartupgate. start_before_epoch isnotreapplied aftereachstage; no newscientificwallbudget. Existingchildgpu_gate checksfreeUUID atstageentry. Onefiniteforloop invokesup to4existingcommands; completedqualifiedstages skipped. Partialtrainusesexplicit--resume onlyonmatchingCOMMITidentity/reference. Full105required beforequalification;1243PASS beforetrain;TRAINING_COMPLETE beforeeval. Allfailuresexitwithoutretry anddurablecontext/exit/logs; noGPUwaiter,poller,scheduler,signalsornewexperimentparameters.

One metadata/staticfixture exercisesstagecompletion/skip/resume/conflict guards withlabelledsyntheticmetadata only. Noactualrootownership,childlaunch,GPUquery,model,array,trainingorfeaturework isdone.
