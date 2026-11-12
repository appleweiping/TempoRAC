"""Read-only server artifact inventory. stdlib only; no model/library import or checkpoint load."""
import ast, collections, difflib, hashlib, json, math, time
from pathlib import Path
R=Path('/root/autodl-tmp/TempoRAC')
A=R/'resume20261111analysis/known_source_v1'
S=R/'source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def ident(p):
    if not p.is_file(): return {'path':str(p),'exists':False}
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def can(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def key(x):return x['split']+'__'+x['video_id']
oldp=R/'data/UNION/manifest_dev.json';newp=R/'data/UNION_known_source_v1/manifest_dev.json'
old=read(oldp);new=read(newp)
report={'task':'bounded_single_UCF_saved_artifact_identity','observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'requested_model':'gpt-6.1-sol','actual_backend':None,'scientific_acceptance':False,'new_stage_invoked':False,'read_only':True,'model_imports':0,'checkpoint_loads':0,'new_predictions':0,'training':0,'gpu_operations':0,'files_written_server':0,'manifests':{'old':ident(oldp),'known_source':ident(newp)},'populations':{},'runs':{},'sources':{},'source_diffs':{},'inherited_receipts':{}}
for split in ['ucf_train','ucf_dev','ucf_test']:
    o=[x for x in old if x['split']==split];n=[x for x in new if x['split']==split]
    report['populations'][split]={'n_old':len(o),'n_corrected':len(n),'ordered_record_hash_old':can(o),'ordered_record_hash_corrected':can(n),'ordered_records_identical':o==n,'ordered_keys_sha256':can([key(x) for x in o]),'role_keys':[key(x) for x in o],'recorded_media_identities_sha256':can([{k:x.get(k) for k in ['split','video_id','sha256','bytes','frame_range','count','bounds','action','group','fsplit','num_frames_csv']} for x in o]),'feature_keys_sha256':can([x.get('fsplit',x['split'])+'__'+x['video_id'] for x in o])}
oldkeys={key(x) for x in old};newkeys={key(x) for x in new}
report['manifests']['removed_keys']=sorted(oldkeys-newkeys)
report['manifests']['added_keys']=sorted(newkeys-oldkeys)
specp=A/'frozen_27_spec.json';spec=read(specp)
report['frozen_spec']=ident(specp)
prior=read(R/'resume20261111analysis/provenance.json')
for seed in range(3):
    name=f'wo_joint_ucf_s{seed}';p=R/'runs/frozen_a'/name;c=read(p/'config.json');target=next(x for x in spec['runs'] if x['name']==name)
    logs=[json.loads(x) for x in (p/'log.jsonl').read_text().splitlines() if x.strip()];pred=read(p/'predictions.json');res=read(p/'result.json')
    h={f:ident(p/f) for f in ['config.json','log.jsonl','stdout.log','best.pt','predictions.json','result.json']}
    binding={}
    for f,v in h.items():
        rec=prior.get(str((p/f).relative_to(R)))
        binding[f]={'present_in_inherited_provenance':rec is not None,'matches_inherited_sha256':rec.get('sha256')==v.get('sha256') if rec else None,'inherited_record':rec}
    prediction_sets={k:{'n':len(v),'keys_sha256':can(list(v)),'key_set_matches_manifest':set(v)==set(report['populations']['ucf_dev' if k.startswith('val') else 'ucf_test']['role_keys'])} for k,v in pred.items() if isinstance(v,dict)}
    cfgdiff={k:{'old':c['cfg'].get(k),'frozen':target['cfg'].get(k)} for k in sorted(set(c['cfg'])|set(target['cfg'])) if c['cfg'].get(k)!=target['cfg'].get(k)}
    argdiff={k:{'old':c['args'].get(k),'frozen':target.get(k)} for k in ['seed','features','train_split','val_split','test_split'] if c['args'].get(k)!=target.get(k)}
    epochs=[x['epoch'] for x in logs]
    report['runs'][name]={'files':h,'config':c,'frozen_target':target,'cfg_canonical_sha256':can(c['cfg']),'effective_cfg_differences':cfgdiff,'seed_feature_split_differences':argdiff,'historical_provenance_binding':binding,'epochs':{'count':len(epochs),'first':epochs[0],'last':epochs[-1],'contiguous_0_to_79':epochs==list(range(80)),'nonfinite_steps_logged':sum(x.get('nonfinite_steps',0) for x in logs)},'prediction_populations':prediction_sets,'saved_result_best_epoch':res.get('best_epoch'),'saved_result_population_counts':{k:v.get('n') for k,v in res.items() if isinstance(v,dict)},'sampling':{'unique_train':331,'repeat_cfg':c['cfg'].get('repeat'),'sampled_entries_per_epoch':331,'batch':c['cfg']['batch'],'batches_per_epoch':math.ceil(331/c['cfg']['batch']),'epochs':80,'nominal_total_updates':80*math.ceil(331/c['cfg']['batch'])},'fresh_init_by_cfg':not any(c['cfg'].get(k) for k in ['init_from','llm_root','mllm_root','llm_desc_roots']),'reuse_status':'potential_equivalent_frozen_arm_not_released','remaining_historical_bindings':['No launch-time run-specific source/environment/cache digest is saved in this run config; inherited Nov11 source/cache receipts are later observations.','No checkpoint contents loaded: checkpoint internal cfg/epoch/state identity not independently inspected.']}
# Source-level comparison only: exact bytes plus AST fingerprints; do not import project modules.
for f in ['train.py','data.py','models.py','metrics.py','eval_saved.py','launch.py']:
    p=A/'source/temporac_cvpr'/f;q=S/f;c=R/'code/temporac_cvpr'/f
    report['sources'][f]={'old_frozen_copy':ident(p),'active_immutable':ident(q),'legacy_runtime_copy':ident(c),'old_equals_active':sha(p)==sha(q),'old_equals_legacy_runtime':sha(p)==sha(c)}
    ot=p.read_text();nt=q.read_text()
    if ot!=nt:
        report['source_diffs'][f]=''.join(difflib.unified_diff(ot.splitlines(True),nt.splitlines(True),fromfile=str(p),tofile=str(q)))
    def funcs(t):
        tree=ast.parse(t)
        return {x.name:can(ast.dump(x,include_attributes=False)) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    of=funcs(ot);nf=funcs(nt)
    report['sources'][f]['top_level_AST_equal']={k:of[k]==nf.get(k) for k in of}
for rel in ['resume20261111analysis/known_source_v1/source_manifest.json','resume20261111analysis/known_source_v1/preflight.json','resume20261111analysis/provenance.json','data/UNION_known_source_v1/derivation_receipt.json','source_sync.json']:
    report['inherited_receipts'][rel]=ident(R/rel)
# Current metadata identity only; avoid hashing feature tensor payloads or rechecking already-complete header pass.
ucf=[x for x in old if x['split'].startswith('ucf_')]
for root in ['UNION_stmae_b8_p4','UNION_stmae_p4']:
    metas=[];missing=[]
    for x in ucf:
        fk=x.get('fsplit',x['split'])+'__'+x['video_id'];p=R/'features'/root/'meta'/(fk+'.json')
        if p.is_file():metas.append({'key':fk,'sha256':sha(p),'bytes':p.stat().st_size})
        else:missing.append(str(p))
    report.setdefault('cache_metadata',{})[root]={'n':len(metas),'ordered_metadata_records_sha256':can(metas),'missing':missing,'tensor_payloads_read':0,'historical_runtime_digest_binding':'not_proven; inherited header/cache reuse receipts retained'}
report['inheritance_note']='Existing 15-framework literature and 29-run analysis inherited, not rerun. Snapshot/provenance hashes identify old evidence without scientific re-analysis.'
print(json.dumps(report,indent=2))
