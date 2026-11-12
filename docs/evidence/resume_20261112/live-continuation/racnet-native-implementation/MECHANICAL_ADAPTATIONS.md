# Mechanical adaptation map

No original mathematical file was changed. Server PLAN.source_sha256 records original author files; isolated vendor/{model.py,dataset/gen_gt.py,dataset/cal_count.py,dataset/adaptive_loader.py} are copiedbyte-for-byte and verified. CANDIDATE.diff is the complete added publiccode diff.

| Original site | Added bridge site | Mechanical repair/boundary |
|---|---|---|
| train.py + phases.py | native_engine.train/batch_forward/validate | retains native model/AMP/two loss computations; explicit4used-tensor unpack and start branch; masks to sameGPU; state/log/selection plumbing |
| tools.set_seed missingimports | native_common.seed | identical fourseedcalls with explicitimports; no plotimports or unused dependency requirement |
| phases tools.misc/unusedobo imports | native_engine imports | direct ownedplumbing imports; no unavailabletools.misc/test imports |
| RACdata AdaptiveTData | SupervisedInputs | manifestnative755/131; originalimg_feature alias; originalTSM labelonly; targets from originalCSV/GenerateGT |
| originalcollate target-derivedmask | supervised_collate | mask derives only actualfeaturelength; CPU tensor-by-tensor exactmatch to original padding and loss tensors |
| final AdaptiveTData requiresunusedTSM | PredictionInputs + independent score | data-only inputmetadata has no count/bounds/targets/TSM; nativeRC GT in isolatedscorer |
| UCF TestData fullarray | PredictionInputs | exact105frozenROI slices88crop/17full, no feature modification/renormalization |
| test4video brokenmodelimport/config/featuredim | native_extract | originalSwin fullgraph with isolatedregistryimports, explicitbasecfg, strict183keys, singleframe5D→CHWflatten2D; originalnormalization/HWswap |
| upstream checkpoint storagesentinel1/bestnames | commitgenerations + minfinite nativeMAE | +inf-equivalent earliestmin selection; epoch11..199; OBOarchivalonly, noextra test |
