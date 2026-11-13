"""Server-only, standard-library aggregate integration. Never opens prediction/model/data payloads."""
import json, pathlib, hashlib, datetime, statistics
R = pathlib.Path('/root/autodl-tmp/TempoRAC')
O = R/'resume20261112/continuation-2244/completed-external-comparison'
assert not (O/'COMPARISON_SAFE.json').exists(), 'immutable packet already built'
bindings = {}
def read(rel):
    p=R/rel; b=p.read_bytes()
    bindings[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    return json.loads(b)
def source(rel):
    p=R/rel; b=p.read_bytes()
    bindings[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    return b.decode()
def write(name,obj):
    p=O/name; assert not p.exists(), name
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n' if not isinstance(obj,str) else obj,encoding='utf-8')
group=read('resume20261112/group-completion/group_summary.json')
group_receipt=read('resume20261112/group-completion/COMPLETION_RECEIPT.json')
audit=read('resume20261111analysis/baseline_source_audit.json')
provenance=read('resume20261111analysis/provenance.json')
rep=read('external-baselines-20261112/runs/repnet-full357-v4/metrics.json')
repconfig=read('external-baselines-20261112/runs/repnet-full357-v4/config.json')
repreceipt=read('external-baselines-20261112/FULL357_RECEIPT.json')
det=read('resume20261112/continuation-2244/detrc-full-completion/FINAL_METRICS_SAFE.json')
detreceipt=read('resume20261112/continuation-2244/detrc-full-completion/FULL_COMPLETION_RECEIPT.json')
detcontract=read('baselines/detrc_native_v1/RUN_CONTRACT.json')
scorer=source('source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py')
source('baselines/detrc_native_v1/driver/score.py')
source('baselines/detrc_native_v1/driver/inference.py')
source('baselines/detrc_native_v1/driver/prepare.py')
assert detreceipt['epochs_complete']==80 and detreceipt['phase_exit_codes']==dict(extract=0,train=0,evaluate=0)
assert det['complete357'] and detreceipt['final_coverage']==dict(rc_test=152,ucf_test=105,test=100)
assert detreceipt['selected_checkpoint']['sha256']==det['checkpoint_sha256']
assert bindings['resume20261112/continuation-2244/detrc-full-completion/FINAL_METRICS_SAFE.json']['sha256']=='043b4bf17969bdaaa30f12d8c89357c30682691a5c44694a22b4ac62fb3055d3'
assert bindings['resume20261112/continuation-2244/detrc-full-completion/FULL_COMPLETION_RECEIPT.json']['sha256']=='9c5bb62af2a0cab7a78205f1c7ef2a6101d95a8bb4994caa22125ecf393b0307'
assert bindings['source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py']['sha256']==det['metric_definitions_source']['sha256']
names={'rc_test':'RepCount-A','cx_test':'Countix','ucf_test':'UCFRep','quva':'QUVA'}
def fields(m):
    return {k:m.get(k) for k in ['n','n_pos','mae_p','mae','mae_pos','obo_p','obo','obz_p','obz','rmse_p','rmse','mae_raw_cont']}
def value(x):
    return x['mean'] if isinstance(x,dict) else x
def means(a):
    return {k:value(v) for k,v in a.items()}
rows=[]
for pop,b in group['benchmarks'].items():
    v=b['views']['two_view_mean'];agg=v['aggregate'];m=means(agg)
    rows.append(dict(method='TempoRAC corrected full3',dataset=names[pop],metrics=fields(m),
        sample_sd={k:v['sample_sd'] for k,v in agg.items() if isinstance(v,dict)},
        zero_aggregate=v['zero_aggregate'],n_seeds=3,metric_source='group_summary.json benchmarks/'+pop+'/views/two_view_mean/aggregate'))
es_map={'RepCount-A':'escounts_official','UCFRep':'escounts_official_ucfrep','QUVA':'escounts_official_quva'}
for dataset,run in es_map.items():
    result=read('runs/baselines/'+run+'/result.json')['test']
    a=audit['escounts_evaluations'][run]['splits']['test']
    assert result['mae']==a['mae_all']
    result.update(mae_p=a['mae_p'],obo_p=a['obo_p'])
    # Positive OBZ/RMSE are not present in the inherited RC aggregate audit.
    # No prediction or label file is opened to fill those fields.
    rows.append(dict(method='ESCounts released RC weights',dataset=dataset,metrics=fields(result),
        n_seeds=None,metric_source='existing result.json + baseline_source_audit.json primary fields',
        unavailable_reason='RC positive OBZ/RMSE absent from inherited aggregate; no rescoring authorized' if dataset=='RepCount-A' else None))
for pop,dataset in [('RepCountA','RepCount-A'),('UCFRep','UCFRep'),('QUVA','QUVA')]:
    rows.append(dict(method='RepNet released updated2025 ckpt70',dataset=dataset,metrics=fields(rep[pop]['harmonized']),
        derived_native_continuous=rep[pop]['derived_native_continuous'],n_seeds=None,metric_source='metrics.json '+pop+'/harmonized'))
for pop,dataset in [('rc_test','RepCount-A'),('ucf_test','UCFRep'),('test','QUVA')]:
    rows.append(dict(method='DeTRC TSN/R50 full80 seed0',dataset=dataset,metrics=fields(det['common'][pop]),
        n_seeds=1,metric_source='FINAL_METRICS_SAFE.json common/'+pop))
primary_deltas=[]
for dataset in ['RepCount-A','UCFRep','QUVA']:
    full=next(x for x in rows if x['dataset']==dataset and x['method'].startswith('TempoRAC'))['metrics']
    for baseline in [x for x in rows if x['dataset']==dataset and not x['method'].startswith('TempoRAC')]:
        primary_deltas.append(dict(dataset=dataset,baseline=baseline['method'],baseline_minus_full_primary_NMAE=baseline['metrics']['mae_p']-full['mae_p'],
            uncertainty='not evaluated for these external contrasts; descriptive arithmetic only'))
weights={
 'TempoRAC':{'training':'our corrected known-source-disjoint RC/Countix/UCF joint training;3seeds0/1/2,50epochs each',
   'training_commit':group['training_commit'],'data_manifest_sha256':group['manifest_sha256'],
   'checkpoint_selection':'per-target original-view validation (mae_p,-obo_p); QUVA uses RC checkpoint',
   'checkpoint_sha256_by_run':{k:v['completion_receipt']['checkpoints'] for k,v in group['run_closure'].items()},
   'inference':'fixed mean of original/shift continuous counts before rounding; mean of3 per-seed metrics, sample SD ddof1; no seed ensemble'},
 'ESCounts':{'weights':'author released RC task checkpoint; evaluated by us; no own retraining',
   'task_weight_recorded_sha256':provenance['pretrained/escounts_repcount.pyth'],
   'backbone_recorded_sha256':provenance['pretrained/VIT_B_16x4_MAE_PT.pyth'],
   'historical_binding_limitation':audit['historical_weight_binding'],
   'transfer':'UCF/QUVA are RC-weight transfer, not native target-trained strongest settings',
   'implementation':'verified ST-MAE encoder reimplementation + official decoder/two-offset exemplar-free decoding; retain adapter qualification'},
 'RepNet':{'weights':'author updated2025 ckpt70 evaluated by us, not original2020 ckpt88 and no own retraining',
   'checkpoint_sha256':repconfig['checkpoint_sha256'],'notebook_commit':repconfig['notebook_commit'],
   'source_sha256':repconfig['source_sha256'],'config':'full357, fixed notebook parameters,196 adopted+161 newly evaluated predictions'},
 'DeTRC':{'weights':'own full80 seed0 counting-model retraining from official TSN/R50 backbone; no acquired author counting task checkpoint',
   'original_commit':detcontract['identity']['original_commit'],'backbone_sha256':detcontract['identity']['backbone_sha256'],
   'selected_checkpoint_sha256':det['checkpoint_sha256'],'selected_epoch':53,
   'selected_criterion':'post-SoftNMS native130 validation OBO strict maximum; earliest tie; common131 diagnostic never selects',
   'training':'755 supervised RC videos after frozen2 known-source exclusions and native missing-count omission; batch64,80epochs,FP32,52native batches/epoch',
   'variant':'official released TSN/ResNet50 implementation; not paper strongest ViT or exact paper-number reproduction',
   'single_weight':'one RC-selected checkpoint for all357 final videos'},
 'pending':{'TransRAC':'not included as a completed/reproduced performance row','RACnet':'not included as a completed/reproduced performance row'}
}
definitions={
 'rounding':'p=np.round(pred), NumPy ties-to-even; err=abs(p-gt)',
 'primary_mae_p':'mean(err/(gt+0.1)) on gt>0; lower better',
 'all_mae':'mean(err/(gt+0.1)) on all valid scalar labels including zeros; lower better',
 'mae_pos':'mean(err/gt) on gt>0; distinct from primary and not raw count MAE',
 'obo_and_obo_p':'fraction err<=1 on all/positive population; higher better',
 'obz_and_obz_p':'fraction err==0 on all/positive population; higher better',
 'rmse_and_rmse_p':'sqrt(mean((p-gt)^2)) on all/positive population; lower better',
 'mae_raw_cont':'mean(abs(pred-gt)/(gt+0.1)) on all; normalized continuous diagnostic, not raw count MAE',
 'DeTRC_common':'count positive-class proposals with final score>0.2 after official SoftNMS sigma0.1/top_k30/threshold0.0001; frozen scalar labels',
 'DeTRC_common_ori':'same score>0.2 count before SoftNMS; retains pre-NMS diagnostics, not an alternative test-selected method',
 'DeTRC_native_RC':'author evaluator: valid native cycles only130val/151test; denominator gt+1e-6 and author cycle GT; native mAP percentage, not harmonized counting score',
 'native_label_disagreement':'one retained RC test scalar/cycle-count disagreement; safe packet records count only, private original evidence preserves identities and labels',
 'RepNet_derived_native':'continuous positive relative error /gt and continuous OBO, derived by prior collector; author notebook provides counts without a dataset metric'
}
qualifications=[
 'Frozen common UCF inclusive temporal ROIs; RC/QUVA full inputs as frozen. ROI availability is a disclosed input adaptation, not independent unknown-boundary evaluation.',
 'Shared full evaluation populations do not equal matched training, supervision, pretraining, budget, modality or checkpoint-selection protocol.',
 'Known-source correction is training-only. Historical test-informed design, inherited conflicting evaluation labels, unknown overlap/pretraining exposure remain unresolved.',
 'TempoRAC uses joint RC/Countix/UCF data and target-selected checkpoints; DeTRC uses native RC755/130 selection and a single RC-selected checkpoint; ESCounts transfers released RC weights; RepNet uses released synthetic/pretrained model.',
 'RC all-population TempoRAC NMAE0.426150442 exceeds ESCounts0.279472368; positive-only numerical gains do not establish universal superiority or zero-event accuracy.',
 'Original-only UCF TempoRAC NMAE0.190705453 is lower than fixed-two-view0.194626165; retain fixed inference and prior negative diagnostic.',
 'Three completed external families across three common domains, plus TempoRAC Countix-only row, do not satisfy6datasets x7convincing external frameworks. DeTRC journal/variant qualification remains; not every completed family meets requested top-conference qualification.',
 'Inherited stronger literature rows and protocol caveats remain in group-end-astra/DECISION.md: no new literature review or global ranking here. User preference not to place non-SOTA domains in the eventual main table is unresolved; this private inventory preserves all results.',
 'No external paired uncertainty analysis, equal-budget superiority, statistical significance, global SOTA, scientific acceptance or submission readiness claimed.',
 'Source small receipts/aggregate/source files hashed only. Model/video/feature/prediction/label payloads not read or rehashed. No rescoring, inference, training, GPU calls, signals, restarts, queue/code/config/shared-status or Git mutation.'
]
packet=dict(schema='temporac.completed_external_aggregate_comparison.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 requested_executor='gpt-6.1-sol',actual_backend_identity='unattested',assurance='deterministic aggregate integration; no new semantic review',
 metric_definitions=definitions,common_metrics=rows,descriptive_primary_deltas=primary_deltas,
 DeTRC_additional_aggregates={'common_ori':det['common_ori'],'native_RC':det['native_RC'],'common131_validation_diagnostic':det['common']['rc_valid'],
 'native_vs_scalar_counterexample_count':det['native_vs_scalar_counterexample_count'],'efficiency':det['efficiency']},
 weights_and_protocol=weights,qualifications=qualifications,
 acceptance={'DeTRC_full80_full357':'PASS','metric_definition_alignment':'PASS','existing_aggregate_metric_inventory':'PASS_WITH_EXPLICIT_ESCOUNTS_RC_POSITIVE_OBZ_RMSE_UNAVAILABLE',
 'negative_results_retained':'PASS','weight_provenance':'PARTIAL_ESCOUNTS_HISTORICAL_EVALUATION_HASH_BINDING_GAP','six_datasets_seven_external_frameworks':'UNMET','six_domain_SOTA':'UNMET','external_significance':'NOT_ESTABLISHED'})
write('COMPARISON_SAFE.json',packet)
write('SOURCE_BINDINGS.json',{'server_root':str(R),'files':bindings,'payload_hashing':False,'prediction_or_label_rows_read':0,'source_hash_mode':'small aggregate/receipt/source files only'})
def fmt(x):
    if x is None:return 'NA'
    if isinstance(x,dict):return f"{x['mean']:.6f} +/- {x['sample_sd']:.6f}"
    return f'{x:.6f}'
def cell(row,k):
    val=row['metrics'].get(k);sd=row.get('sample_sd',{}).get(k)
    return fmt({'mean':val,'sample_sd':sd}) if sd is not None else fmt(val)
text='# Completed external performance inventory (private, aggregate only)\n\n'
text+='Corrected full3 is inherited unchanged; DeTRC completed at '+detreceipt['completion_endtime_utc']+'. Raw numbers precede interpretation. All rates are fractions; six decimals. TempoRAC is mean +/- sample SD, not an ensemble. NA identifies fields absent from existing aggregate evidence.\n\n'
text+='| Method | Dataset | All/positive N | Primary NMAE positive | NMAE all | Relative error /gt positive |\n|---|---|---:|---:|---:|---:|\n'
for row in rows:
    m=row['metrics'];text+=f"| {row['method']} | {row['dataset']} | {m['n']}/{m['n_pos']} | {cell(row,'mae_p')} | {cell(row,'mae')} | {cell(row,'mae_pos')} |\n"
text+='\n| Method | Dataset | OBO positive/all | OBZ positive/all | RMSE positive/all | Continuous NMAE all |\n|---|---|---:|---:|---:|---:|\n'
for row in rows:
    text+=f"| {row['method']} | {row['dataset']} | {cell(row,'obo_p')} / {cell(row,'obo')} | {cell(row,'obz_p')} / {cell(row,'obz')} | {cell(row,'rmse_p')} / {cell(row,'rmse')} | {cell(row,'mae_raw_cont')} |\n"
text+='\nESCounts RC positive OBZ/RMSE are unavailable in the inherited aggregate files; all-population OBZ/RMSE remain listed. No predictions or labels were reopened to fill them.\n\n'
text+='| DeTRC pre-SoftNMS common_ori | All/positive N | Primary NMAE | All NMAE | OBO pos/all | OBZ pos/all | RMSE pos/all |\n|---|---:|---:|---:|---:|---:|---:|\n'
for pop,m in det['common_ori'].items():
    text+=f"| {pop} | {m['n']}/{m['n_pos']} | {fmt(m['mae_p'])} | {fmt(m['mae'])} | {fmt(m['obo_p'])}/{fmt(m['obo'])} | {fmt(m['obz_p'])}/{fmt(m['obz'])} | {fmt(m['rmse_p'])}/{fmt(m['rmse'])} |\n"
text+='\n| DeTRC native author cycles | N | MAE (/gt+1e-6) | OBO | ori_MAE | ori_OBO | mAP (%) |\n|---|---:|---:|---:|---:|---:|---:|\n'
for pop,m in det['native_RC'].items():
    n=130 if pop=='rc_valid' else 151
    text+=f"| {pop} | {n} | {fmt(m['MAE'])} | {fmt(m['OBO'])} | {fmt(m['ori_MAE'])} | {fmt(m['ori_OBO'])} | {fmt(m['mAP'])} |\n"
text+='\nNative short/medium/long MAE/OBO, IoU decay, common131 diagnostic and full pre-NMS fields are preserved in COMPARISON_SAFE.json. Native cycles differ from scalar labels on one retained test record; native/common numbers are not interchangeable.\n\n'
text+='The primary NMAE mean is numerically lower for TempoRAC than all three exact completed baseline settings on RC/UCF/QUVA. RC all-population NMAE remains worse than ESCounts. No external significance or SOTA conclusion follows. Countix has no completed external baseline in this scope. TransRAC/RACnet remain pending performance, not reproduced rows.\n\n'
text+='## Provenance and protocol\n\n'
for name,w in weights.items():
    text+='- '+name+': '+json.dumps(w,ensure_ascii=False)+'\n'
text+='\n## Metric definitions\n\n'
for k,v in definitions.items():text+='- '+k+': '+v+'\n'
text+='\n## Qualifications and retained negative evidence\n\n'
for q in qualifications:text+='- '+q+'\n'
text+='\n## Acceptance\n\n| Requirement | Outcome | Evidence |\n|---|---|---|\n'
for k,v in packet['acceptance'].items():text+=f'| {k} | {v} | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |\n'
text+='\nThis text/JSON inventory is not a Word/PDF deliverable; no paper or pagination was changed. Requested executor gpt-6.1-sol; actual backend identity unattested.\n'
write('PERFORMANCE_TABLES.md',text)
write('ACCEPTANCE.md','# Acceptance evidence\n\n'+text[text.index('## Acceptance'):])
manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in O.iterdir() if p.is_file() and p.name!='PACKET_SHA256.json'}
write('PACKET_SHA256.json',{'files':manifest,'scope':'immutable aggregate packet before transfer receipt; no self-reference'})
print(json.dumps({'packet':str(O),'rows':len(rows),'files':list(manifest),'completion':'aggregate inventory complete; benchmark breadth/SOTA unmet'}))
