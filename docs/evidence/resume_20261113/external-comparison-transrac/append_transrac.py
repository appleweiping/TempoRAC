"""Append collector-safe TransRAC aggregates once, on server, after root approval.
Never opens predictions, labels, model/feature/checkpoint payloads or reruns scoring.
"""
import argparse, copy, datetime, hashlib, json, math, pathlib

R = pathlib.Path('/root/autodl-tmp/TempoRAC')
BASE = R/'resume20261112/continuation-2244/completed-external-comparison'
COLLECT = R/'resume20261113/transrac-full-completion/evidence'
O = R/'resume20261113/external-comparison-transrac'
TC = '1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a'
SCORER = 'fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0'
PUBLICATION = '39dcafb8d77de9e43f9b61ae0346d726eddd18b5'
POPS = {'rc_test': ('RepCount-A', 152, 151), 'ucf_test': ('UCFRep', 105, 105), 'test': ('QUVA', 100, 100)}
FIELDS = ['n','n_pos','mae_p','mae','mae_pos','obo_p','obo','obz_p','obz','rmse_p','rmse','mae_raw_cont']

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--actualpacketready', action='store_true', required=True)
    a.add_argument('--approved-receipt-sha256', required=True)
    a.add_argument('--approved-metrics-sha256', required=True)
    args = a.parse_args()
    assert pathlib.Path('/proc').exists(), 'server-only adapter'
    assert not (O/'COMPARISON_SAFE.json').exists(), 'immutable output already exists'
    bindings = {}
    def read(p, expected=None):
        b = p.read_bytes(); digest = hashlib.sha256(b).hexdigest()
        assert expected is None or digest == expected, str(p)
        bindings[str(p.relative_to(R))] = {'bytes':len(b),'sha256':digest}
        return json.loads(b)
    old = read(BASE/'COMPARISON_SAFE.json', '2e6bf530d0eb7f9df78d89bdc29f5629bf2fd99031bc224c3ba2a247913dfa58')
    read(BASE/'SOURCE_BINDINGS.json', 'b6fdc32e9f07d443b37fd037f35e4a498ad77c02195ab1a2c6903221b736e6b7')
    receipt = read(COLLECT/'FULL_COMPLETION_RECEIPT.json', args.approved_receipt_sha256)
    safe = read(COLLECT/'FINAL_METRICS_SAFE.json', args.approved_metrics_sha256)
    zero = read(COLLECT/'ZERO_COUNT_PRESERVATION_SAFE.json')
    assert old['schema'] == 'temporac.completed_external_aggregate_comparison.v1'
    assert len(old['common_metrics']) == 13
    assert not any('TransRAC' in x['method'] for x in old['common_metrics'])
    assert receipt['status'] == 'COMPLETE' and receipt['completed_epochs'] == 200
    assert receipt['candidate_count'] == 50 and receipt['selection_tuple_minimum_verified']
    assert receipt['final_population_total'] == 357
    assert receipt['final_populations'] == {p:n for p,(_,n,_) in POPS.items()}
    assert receipt['exact_prediction_membership_verified'] and receipt['raw_individual_predictions_verified'] == 357
    assert receipt['actual_pipeline_exited'] and receipt['no_own_gpu_compute_remaining']
    assert receipt['source_publication'] == PUBLICATION
    assert receipt['contract_sha256'] == safe['contract_sha256'] == TC
    assert receipt['safe_metrics_sha256'] == args.approved_metrics_sha256
    assert receipt['completion_sha256'] == safe['native_completion_sha256']
    assert receipt['canonical_scorer_sha256'] == safe['canonical_scorer_sha256'] == SCORER
    assert receipt['original_final_metric_sha256'] == safe['original_final_metric_sha256']
    assert set(safe['original_final_metric_sha256']) == set(POPS)
    assert set(receipt['prediction_aggregate_sha256']) == set(POPS)
    for d in [receipt['completion_sha256'],receipt['best_checkpoint_sha256'],receipt['latest_sha256'],*safe['original_final_metric_sha256'].values(),*receipt['prediction_aggregate_sha256'].values()]:
        assert len(d)==64 and all(c in '0123456789abcdef' for c in d)
    assert safe['selected_epoch'] == receipt['best']['epoch']
    assert safe['selection'] == receipt['best']['selection_key']
    assert safe['selected_epoch'] in receipt['candidate_epoch_indices']
    assert 'zero-based' in safe['epoch_index_basis'] and 'zero-based' in receipt['epoch_index_basis']
    assert set(safe['metrics']) == set(POPS), 'validation must not appear as final performance'
    assert zero['n'] == 1 and zero['by_population'] == {'rc_test':1}
    assert zero['original_rows_retained'] and not zero['rescoring_repeated']
    rows = copy.deepcopy(old['common_metrics']); native = []; unavailable = []
    for pop,(dataset,n,npos) in POPS.items():
        original = safe['metrics'][pop]; m = original['harmonized']; nm = original['native']
        assert m['n'] == n and m['n_pos'] == npos and m['zero_count']['n'] == n-npos
        assert nm['n'] == n
        for block in [m,nm]:
            for key,value in block.items():
                if isinstance(value,(float,int)):
                    assert math.isfinite(value), (pop,key)
        assert m['mae_p'] is not None and m['mae'] is not None
        for key in FIELDS:
            if m.get(key) is None: unavailable.append({'population':pop,'field':key,'reason':'Absent from collector-safe aggregate; no rescoring or fill authorized'})
        rows.append({'method':'TransRAC official-code full200 seed1','dataset':dataset,
            'metrics':{k:m.get(k) for k in FIELDS},'n_seeds':1,
            'metric_source':'collector FINAL_METRICS_SAFE.json metrics/'+pop+'/harmonized',
            'zero_count':m['zero_count'],'selected_epoch_native_zero_based':safe['selected_epoch']})
        native.append({'method':'TransRAC official-code full200 seed1','dataset':dataset,
            'population':pop,'metrics':copy.deepcopy(nm),'metric_source':'collector FINAL_METRICS_SAFE.json metrics/'+pop+'/native',
            'qualification':'Official/native GT and rounding: '+nm['meaning']+'; distinct native evaluator retained without recomputation.'})
    assert rows[:13] == old['common_metrics']
    qualifications = [
        'Inherited aggregate and its 13 common rows are preserved exactly as historical evidence; historical pending-TransRAC and three-family statements inside it describe that earlier packet only.',
        'Appended full final populations are RC152 (151 positive), UCF105, QUVA100. RC131 validation is checkpoint selection only and never a final performance row.',
        'TransRAC is own official-code full200 seed1 counting-model retraining with official K400 backbone, corrected RC756 training, and a single common-validation-selected checkpoint; not author released counting-task weights or exact paper-number reproduction.',
        'Native selected epoch is zero-based. Native evaluator uses density-derived RC GT and scalar UCF/QUVA GT with native tensor rounding; harmonized scalar-label metrics remain separate.',
        'Existing zero-event, unfavorable, missing-metric and protocol qualifications remain in inherited_aggregate. No old result was recomputed, overwritten or omitted.',
        'Training, supervision, pretraining, input adaptation, checkpoint selection and budgets differ. No matched-budget, significance, universal superiority, global SOTA, scientific acceptance or submission-readiness claim.',
        'Four completed external families on three shared domains do not fulfill six datasets by seven convincing top-conference frameworks or six-domain SOTA. Countix still lacks a completed external comparator in this packet.',
        'Model/backend identity is unattested; same-family assurance is provisional. This is deterministic safe-metadata integration with no additional semantic review.',
        'No prediction/label/model/feature/checkpoint payloads opened or rehashed; no metric recomputation, experiment, inference, preprocessing, environment change, source/status/Git mutation or other process control.'
    ]
    packet = dict(schema='temporac.completed_external_aggregate_comparison.transrac_append.v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),requested_executor='gpt-6.1-sol',actual_backend_identity='unattested',same_family_assurance='provisional',
        inherited_aggregate=old,common_metrics=rows,TransRAC_native_rows=native,
        TransRAC_collector_safe_metadata=safe,TransRAC_completion_receipt=receipt,
        TransRAC_zero_preservation=zero,unavailable_metrics=unavailable,qualifications=qualifications,
        acceptance={'immutable_old_rows':'PASS_13_OF_13_UNCHANGED','full200_full357':'PASS_COLLECTOR_BOUND_METADATA',
          'currentfinal_metric_identity':'PASS_ORIGINAL_FINAL_SHA_MAP_AND_NATIVE_COMPLETION_SHA_MATCH',
          'common_scorer_identity':'PASS_EXACT_FROZEN_SHA','native_common_separation':'PASS','validation_selection_only':'PASS',
          'zero_and_negative_results':'PASS_INHERITED_AND_APPENDED_WITHOUT_FILTERING','missing_metrics':'EXPLICIT_NA_NO_FILL',
          'six_datasets_seven_frameworks':'UNMET','six_domain_SOTA':'UNMET','matched_efficiency':'PARTIAL_CORESIDENCY_AND_SCOPE_CAVEATS',
          'scientific_acceptance':'NOT_ESTABLISHED','public_publication':'DEFERRED_ROOT_EXPLICIT_ACTION'})
    def fmt(v):
        return 'NA' if v is None else str(v) if isinstance(v,str) else f'{v:.6f}'
    def cell(row,k):
        v=row['metrics'].get(k);sd=row.get('sample_sd',{}).get(k)
        return fmt(v)+(f' +/- {fmt(sd)}' if sd is not None else '')
    text = '# Completed external performance inventory with TransRAC (private)\n\n'
    text += 'All values are inherited aggregate scalars. TempoRAC is mean +/- sample SD over three seeds; external retraining rows have one seed. Rates are fractions. NA means unavailable; no new score calculation was performed.\n\n'
    text += '| Method | Dataset | All/positive N | NMAE positive | NMAE all | /gt positive | OBO positive/all | OBZ positive/all | RMSE positive/all | Continuous NMAE all |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        m=row['metrics']; text += f"| {row['method']} | {row['dataset']} | {m['n']}/{m['n_pos']} | {cell(row,'mae_p')} | {cell(row,'mae')} | {cell(row,'mae_pos')} | {cell(row,'obo_p')} / {cell(row,'obo')} | {cell(row,'obz_p')} / {cell(row,'obz')} | {cell(row,'rmse_p')} / {cell(row,'rmse')} | {cell(row,'mae_raw_cont')} |\n"
    text += '\n| TransRAC native evaluator | N | Native MAE | Native OBO | Native OBZ | Native RMSE | Meaning |\n|---|---:|---:|---:|---:|---:|---|\n'
    for row in native:
        m=row['metrics'];text += f"| {row['dataset']} | {m.get('n')} | {fmt(m.get('mae'))} | {fmt(m.get('obo'))} | {fmt(m.get('obz'))} | {fmt(m.get('rmse'))} | {str(m.get('meaning','See collector-safe metadata')).replace('|','/')} |\n"
    text += '\nTransRAC selected native epoch index: '+str(safe['selected_epoch'])+' (zero-based; never relabelled as completed epoch count). Training loop wall / training compute / validation seconds, as already recorded: '+str(safe.get('training_wall_seconds'))+' / '+str(safe.get('training_compute_seconds'))+' / '+str(safe.get('validation_seconds'))+'. '+safe['efficiency_qualification']+'\n\n'
    text += 'Official/native metrics are separated from common scalar scoring. The full original per-population safe metric objects, original final-metric SHA map, checkpoint SHA, zero-count receipt and inherited negative results remain in COMPARISON_SAFE.json. Missing ESCounts RC positive OBZ/RMSE remain NA.\n\n'
    text += '\n'.join('- '+q for q in qualifications)+'\n\n'
    text += 'Inherited protocol and unresolved limitations:\n\n'+'\n'.join('- '+q for q in old['qualifications'])+'\n'
    acceptance = '# Acceptance evidence\n\n| Requirement | Outcome | Evidence |\n|---|---|---|\n'
    for k,v in packet['acceptance'].items(): acceptance += f'| {k} | {v} | COMPARISON_SAFE.json and SOURCE_BINDINGS.json |\n'
    acceptance += '\nThis private JSON/Markdown packet changes no Word/PDF or paper. No page-rendering acceptance is implied. Publication is deferred to root.\n'
    outputs={'COMPARISON_SAFE.json':packet,'PERFORMANCE_TABLES.md':text,'ACCEPTANCE.md':acceptance,
       'SOURCE_BINDINGS.json':{'server_root':str(R),'input_files':bindings,'collector_validation_inherited':True,
        'adapter_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'rows_inherited_unchanged':13,'rows_appended_common':3,'rows_appended_native':3,
        'prediction_or_label_rows_read':0,'payload_hashing':False,'metric_recomputation':False}}
    for name in outputs: assert not (O/name).exists(), name
    O.mkdir(parents=True,exist_ok=True)
    for name,obj in outputs.items():
        (O/name).write_text(obj if isinstance(obj,str) else json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in O.iterdir() if p.is_file() and p.name!='PACKET_SHA256.json'}
    assert not (O/'PACKET_SHA256.json').exists()
    (O/'PACKET_SHA256.json').write_text(json.dumps({'files':manifest,'scope':'safe append packet; no payloads; no self-reference'},indent=2)+'\n')
    print(json.dumps({'packet':str(O),'old_common_rows_unchanged':13,'common_rows_total':len(rows),'native_rows_appended':len(native),'publication':'deferred root explicit action'}))

if __name__ == '__main__':
    main()