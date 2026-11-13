import pathlib,json,hashlib,math,datetime
O=pathlib.Path('/root/autodl-tmp/TempoRAC/resume20261112/continuation-2244/completed-external-comparison')
assert not (O/'ZERO_COUNTS_AGGREGATE.json').exists()
a=json.loads((O/'COMPARISON_SAFE.json').read_text());zeros=[]
for r in a['common_metrics']:
 if r['dataset']!='RepCount-A':continue
 m=r['metrics'];n=m['n'];p=m['n_pos'];z=n-p
 if r['method'].startswith('TempoRAC'):
  q=r['zero_aggregate'];mean=lambda k:q[k]['mean']
  d=dict(method=r['method'],n_zero=q['n'],zero_all_denominator_NMAE=mean('mae'),zero_mean_absolute_rounded_error=mean('rmse'),zero_OBO=mean('obo'),zero_OBZ=mean('obz'),zero_RMSE=mean('rmse'),aggregation='existing mean of per-seed zero aggregates;3seeds, no ensemble')
 else:
  v=(n*m['mae']-p*m['mae_p'])/z
  d=dict(method=r['method'],n_zero=z,zero_all_denominator_NMAE=v,zero_mean_absolute_rounded_error=v*0.1,zero_OBO=(n*m['obo']-p*m['obo_p'])/z,aggregation='aggregate identities only: (N*all-Npos*positive)/Nzero; no rows or labels read')
  for full,pos,key in [('obz','obz_p','zero_OBZ'),('rmse','rmse_p','zero_RMSE')]:
   d[key]=None if m[pos] is None else ((n*m[full]**2-p*m[pos]**2)/z)**0.5 if full=='rmse' else (n*m[full]-p*m[pos])/z
 zeros.append(d)
(O/'ZERO_COUNTS_AGGREGATE.json').write_text(json.dumps({'schema':'temporac.zero_count_aggregate_supplement.v1','source_comparison_sha256':hashlib.sha256((O/'COMPARISON_SAFE.json').read_bytes()).hexdigest(),'scope':'RC zero-count aggregate only; zero positive NMAE undefined','rows':zeros,'no_per_video_rows_or_labels_read':True},indent=2,allow_nan=False)+'\n')
t='# RC zero-count aggregate supplement\n\nDerived only from existing aggregate identities on the server, without rescoring. RC all152/positive151 leaves one zero-count evaluation case. TempoRAC values average the3 fitted models; sample dispersion remains in the original JSON zero_aggregate. No identities, predictions or labels are included.\n\n| Method | Zero N | Zero normalized error (denominator0.1) | Zero mean absolute rounded error | Zero OBO | Zero OBZ | Zero RMSE |\n|---|---:|---:|---:|---:|---:|---:|\n'
for r in zeros:
 fmt=lambda x:'NA' if x is None else f'{x:.6f}'
 t+='| '+r['method']+' | '+str(r['n_zero'])+' | '+' | '.join(fmt(r[k]) for k in ['zero_all_denominator_NMAE','zero_mean_absolute_rounded_error','zero_OBO','zero_OBZ','zero_RMSE'])+' |\n'
t+='\nTempoRAC zero mean absolute rounded error4.333333 is worse than ESCounts1.000000, RepNet3.000000 and DeTRC4.000000 in these exact evaluated settings. Positive-only superiority does not extend to zeros. ESCounts zero OBZ/RMSE are NA because its inherited RC aggregate lacks positive OBZ/RMSE; deriving missing fields from per-video evidence is outside this aggregate-only task.\n'
(O/'DIAGNOSTIC_SUPPLEMENT.md').write_text(t)
# Only schema/aggregate checks; no experiment imports or payload IO.
assert len(a['common_metrics'])==13
assert all('video_id' not in json.dumps(r) and 'zero_count_rows' not in json.dumps(r) for r in a['common_metrics'])
assert all(x['baseline_minus_full_primary_NMAE']>0 for x in a['descriptive_primary_deltas'])
qa={'status':'PASS_WITH_DISCLOSED_GAPS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'common_rows':13,'final_domains_common':3,'TempoRAC_total_domains':4,'completed_external_families':3,'pending_external':['TransRAC','RACnet'],'DeTRC_full80_complete':True,'external_primary_deltas_positive':True,'zero_negative_preserved':True,'six_datasets_seven_frameworks':'UNMET','six_domain_SOTA':'UNMET','missing_ESCounts_positive_OBZ_RMSE':'explicit NA; no rescore','semantic_review':'not performed; no new review of running experiments','per_video_rows_or_label_IO':0,'GPU_or_model_calls':0}
(O/'AGGREGATE_QA_RECEIPT.json').write_text(json.dumps(qa,indent=2)+'\n')
manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in O.iterdir() if p.is_file() and p.name not in ['FINAL_PACKET_SHA256.json','TRANSFER_VERIFICATION.json']}
(O/'FINAL_PACKET_SHA256.json').write_text(json.dumps({'files':manifest,'scope':'final immutable packet including supplements and context, before transfer receipt'},indent=2)+'\n')
print(json.dumps({'zero_aggregate':zeros,'QA':qa}))
