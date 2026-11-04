"""Explicit CPU prepared-cache builder and consumers for frozen full NTR1 inputs."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import time
import os
import numpy as np
from pams.temporac.native_prepared_cache_v1 import (UnitStore, atomic_json, file_sha, json_sha,
    kernel_binding, natural_binding, read_derived, prepare_natural_arrays, cached_natural_unit)

MANIFEST_SCHEMA = 'temporac.prepared-cache.manifest.v1'
INVENTORY_SCHEMA = 'temporac.prepared-cache.inventory.v1'


def input_signature(cfg):
    return {k:cfg[k] for k in ('census','bridge_root','target_root','component_map','x0_root','artifacts')}


def population_signature(payload):
    return json_sha({'population':payload['population'],'targets':payload['targets']})


def paths_for(cfg,row):
    key,split=row['key'],row['split']
    base=Path(cfg['bridge_root'])/'identities'/split/key
    target=Path(cfg['target_root'])/'identities'/split/(key+'.npz')
    return {'feature_sha256':base.with_suffix('.npz'),'bridge_receipt_sha256':base.with_suffix('.json'),
        'derived_sha256':base.with_suffix('.derived.npz'),'target_sha256':target,
        'target_receipt_sha256':target.with_suffix('.json')}


def verify_row_files(cfg,row):
    paths=paths_for(cfg,row)
    for field,path in paths.items():
        if file_sha(path)!=row['bindings'][field]:
            raise ValueError('frozen input changed: '+str(path))
    bridge=json.loads(paths['bridge_receipt_sha256'].read_text(encoding='utf-8'))
    if bridge['feature_sha256']!=row['bindings']['feature_sha256'] or bridge['derived_sha256']!=row['bindings']['derived_sha256']:
        raise ValueError('bridge binding mismatch')
    return paths,bridge


def supervision_binding(row,kernel):
    return {'feature':natural_binding(row,kernel),'target_sha256':row['bindings']['target_sha256'],
        'target_receipt_sha256':row['bindings']['target_receipt_sha256']}


def row_supervision(identity,target,expected):
    from pams.temporac.native_training_runtime_v1 import legal_lf1_supervision
    s=legal_lf1_supervision(identity,target)
    actual={'eligible':bool(len(s['traversal_bounds'])),'traversals':len(target.traversal_bounds),
        'eligible_traversals':len(s['traversal_bounds']),'traversal_eligibility':s['eligibility_ledger'],
        'reason':None if len(s['traversal_bounds']) else 'no_complete_positive_and_negative_LF1_traversal'}
    if actual!=expected:
        raise ValueError('LF1 eligibility differs from frozen plan')
    return {'pulse':s['pulse'],'chi':s['chi'],'loss_mask':s['edge_mask'],
        'traversal_bounds':s['traversal_bounds']}, {'eligibility':actual}


class Progress:
    def __init__(self,root,phase,total):
        self.root=Path(root);self.start=time.monotonic();self.cpu=time.process_time();self.last=0.
        self.data={'phase':phase,'total':total,'committed':0,'hits':0,'failed':0,'current_key':None,
            'window_done':0,'window_total':0,'pid':os.getpid(),'verified':0,'bytes_committed':0}
        self.update(force=True)
    def update(self,force=False,**changes):
        self.data.update(changes)
        now=time.monotonic()
        if force or now-self.last>=30:
            atomic_json(self.root/'progress.json',dict(self.data,wall_seconds=now-self.start,
                cpu_seconds=time.process_time()-self.cpu))
            self.last=now
    def completed(self,row):
        with (self.root/'timings.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        if row['phase']=='inventory':
            self.update(force=True,verified=self.data['verified']+1,last_unit=row)
        else:
            self.update(force=True,committed=self.data['committed']+1,last_unit=row,
                bytes_committed=self.data['bytes_committed']+row.get('bytes',0))


def validate_full_inventory(cfg,payload,progress=None,supervision_root=None):
    """All input SHA/schema/eligibility checks, without preprocess or routing."""
    from pams.temporac import native_training_runtime_v1 as rt
    from pams.temporac.native_target_nt1 import load_native_target
    from pams.temporac.native_target_diagnostics_v1 import source_membership
    from pams.temporac.native_x0_adapter_v1 import validate_x0_bank
    rt._check_config(cfg)
    _,pending=rt._artifact_receipts(cfg)
    if pending: raise ValueError('unqualified artifacts: '+str(pending))
    census=rt._module_from_script('build_temporac_native_learner_bridge').read_census(cfg['census'])
    components,aliases=source_membership(census,json.loads(Path(cfg['component_map']).read_text(encoding='utf-8-sig')))
    if len(aliases)!=4 or any(any(x['split']!='train' for x in group) for group in aliases):
        raise ValueError('frozen alias groups differ')
    census_rows=[(f"{r['media_key']}_{p['slot']}",r['split'],r['length']) for r in census for p in r['persons']]
    pop=payload['population']
    if [(r['key'],r['split'],r['N']) for r in pop]!=census_rows:
        raise ValueError('frozen complete population order differs from census')
    if len(pop)!=2364 or Counter(r['split'] for r in pop)!={'train':1878,'val':486} or sum(r['N'] for r in pop)!=1758034:
        raise ValueError('full frozen population required')
    if {s:len({r['media_key'] for r in pop if r['split']==s}) for s in ('train','val')}!={'train':794,'val':183}:
        raise ValueError('full video population required')
    if set(payload['targets'])!={r['key'] for r in pop}:
        raise ValueError('full target eligibility index required')
    kernel=kernel_binding();store=UnitStore(supervision_root,'supervision') if supervision_root else None
    sup_records=[]
    for index,row in enumerate(pop):
        t=time.monotonic()
        if progress:progress.update(current_key=row['key'],window_done=0,window_total=0,unit_started_epoch=time.time())
        paths,bridge=verify_row_files(cfg,row)
        identity=read_derived(paths['feature_sha256'],paths['derived_sha256'],expected_N=row['N'],expected_key=row['key'],for_routing=False)
        target=load_native_target(paths['target_sha256'],expected_input_sha256=row['bindings']['feature_sha256'])
        if target.metadata['bridge_sha256']!=json_sha(bridge['binding']):
            raise ValueError('target/bridge binding mismatch')
        arrays,meta=row_supervision(identity,target,payload['targets'][row['key']])
        if row['status']!=identity.status or row['max_run']!=int(max((b-a+1 for a,b in identity.run_bounds),default=0)):
            raise ValueError('identity status/run census differs')
        if row['split']=='train' and row['component']!=components[row['media_key']]:
            raise ValueError('component differs from frozen plan')
        if store and row['split']=='train' and payload['targets'][row['key']]['eligible']:
            binding=supervision_binding(row,kernel)
            receipt=store.write(row['key'],binding,arrays,meta)
            sup_records.append(unit_record(store,row['key'],binding,receipt))
        if progress:progress.completed({'phase':'inventory','key':row['key'],'wall_seconds':time.monotonic()-t})
    rt.verify_evidence_bindings(cfg,pop)
    bank=validate_x0_bank(cfg['x0_root'])
    if bank.get('fresh_process_byte_check')!='complete':raise ValueError('X0 original parity incomplete')
    return {'schema':INVENTORY_SCHEMA,'status':'complete','population_signature':population_signature(payload),
        'input_signature':input_signature(cfg),'population':pop,'targets':payload['targets'],
        'source_binding':payload['population_binding']['sources'],'supervision':sup_records,
        'kernel':kernel,'x0_manifest_sha256':file_sha(Path(cfg['x0_root'])/'manifest.json')}


def unit_record(store,identity,binding,receipt):
    return {'identity':identity,'key':receipt['key'],'binding':binding,
        'receipt_sha256':file_sha(store.root/(receipt['key']+'.json')),'bytes':receipt['bytes']}


def load_unit(store,row):
    if store.key(row['identity'],row['binding'])!=row['key'] or file_sha(store.root/(row['key']+'.json'))!=row['receipt_sha256']:
        raise ValueError('manifest unit receipt mismatch')
    item=store.load(row['identity'],row['binding'])
    if item is None:raise ValueError('manifest points to incomplete unit')
    return item[0]


def load_and_verify_inventory(cfg,payload,ref,*,verify_inputs=True):
    if file_sha(ref['path'])!=ref['sha256']:raise ValueError('inventory manifest changed')
    inv=json.loads(Path(ref['path']).read_text(encoding='utf-8'))
    if inv.get('schema')!=INVENTORY_SCHEMA or inv.get('status')!='complete':raise ValueError('inventory incomplete')
    if inv['population_signature']!=population_signature(payload) or inv['input_signature']!=input_signature(cfg):
        raise ValueError('inventory belongs to different frozen inputs')
    if inv['kernel']!=kernel_binding():raise ValueError('cache kernel/runtime environment changed')
    if verify_inputs:
        # Recheck each full input checksum, schema and eligibility, never preprocessing/routing.
        validate_full_inventory(cfg,payload)
    return inv


def freeze_json(path,payload):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=payload:
            raise ValueError('immutable manifest differs: '+str(path))
    else:atomic_json(path,payload)
    return {'path':str(path.resolve()),'sha256':file_sha(path)}


def x0_binding(cfg,kernel,source):
    return {'kernel':kernel,'bank_manifest_sha256':file_sha(Path(cfg['x0_root'])/'manifest.json'),'source':source}


def x0_arrays(unit):
    return {'response_input':unit.response_input,'run_bounds':unit.run_bounds,
        'windows':np.asarray([(w.run_index,w.edge_start,w.edge_stop) if hasattr(w,'run_index') else (0,*w) for w in unit.windows],np.int64).reshape(-1,3),
        'gates':unit.gates,'denominator':unit.cache.denominator,'responsibility':unit.cache.responsibility,
        'support':unit.cache.support,'pulse':unit.pulse,'chi':unit.chi,'loss_mask':unit.loss_mask,
        'traversal_bounds':unit.traversal_bounds}


def prepare_natural_record(task):
    """One independent feature-only CPU job; no target access, no model execution."""
    cfg,row,root,kernel=task
    root=Path(root)
    progress=Progress(root/'workers'/str(os.getpid()),'natural_unit',1)
    begin=time.monotonic();progress.update(force=True,current_key=row['key'],unit_started_epoch=time.time())
    store=UnitStore(root/'natural','natural')
    binding=natural_binding(row,kernel);hit=store.load(row['key'],binding)
    timings={}
    if hit is None:
        paths=paths_for(cfg,row)
        for field in ('feature_sha256','bridge_receipt_sha256','derived_sha256'):
            if file_sha(paths[field])!=row['bindings'][field]:raise ValueError('NB1 input changed before preparation')
        t=time.monotonic()
        identity=read_derived(paths['feature_sha256'],paths['derived_sha256'],expected_N=row['N'],expected_key=row['key'])
        timings['load_reconstruct_seconds']=time.monotonic()-t
        arrays,timings2=prepare_natural_arrays(identity,progress=lambda done,total:progress.update(window_done=done,window_total=total))
        timings.update(timings2);t=time.monotonic()
        receipt=store.write(row['key'],binding,arrays)
        timings['write_seconds']=time.monotonic()-t
    else:
        receipt=hit[1]
    timing=dict(timings,phase='natural_unit',key=row['key'],worker_pid=os.getpid(),
        wall_seconds=time.monotonic()-begin,bytes=receipt['bytes'],cache_hit=hit is not None)
    progress.completed(timing)
    return unit_record(store,row['key'],binding,receipt),timing


def ordered_task_results(tasks,workers):
    """Yield completion receipts promptly; final manifest is sorted separately."""
    if type(workers) is not int or not 1<=workers<=3:
        raise ValueError('cache workers must be 1..3 (plus existing CPU worker <=4)')
    if workers==1:
        for task in tasks:yield prepare_natural_record(task)
        return
    if os.name=='nt':
        raise ValueError('parallel builder is Linux-only; Windows uses one non-spawning worker')
    from concurrent.futures import ProcessPoolExecutor,as_completed
    executor=ProcessPoolExecutor(max_workers=workers)
    futures=[executor.submit(prepare_natural_record,task) for task in tasks]
    try:
        for future in as_completed(futures):yield future.result()
    except BaseException:
        for future in futures:future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def build_caches(cfg,payload,root,phase='train',workers=1):
    if type(workers) is not int or not 1<=workers<=3:raise ValueError('cache workers must be 1..3')
    if phase not in ('inventory','train','dev'):raise ValueError('cache phase must be train or dev')
    from pams.temporac.native_training_runtime_v1 import prepare_x0_unit
    from pams.temporac.native_x0_adapter_v1 import X0AnalyticAdapterNTR1
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    kernel=kernel_binding();invpath=root/'inventory.json'
    if invpath.exists():
        invref={'path':str(invpath.resolve()),'sha256':file_sha(invpath)}
        inv=load_and_verify_inventory(cfg,payload,invref,verify_inputs=False)
    else:
        progress=Progress(root,'inventory',2364)
        inv=validate_full_inventory(cfg,payload,progress,root/'supervision')
        invref=freeze_json(invpath,inv)
    compatibility={'schema':'temporac.ntr1.cache-compatibility.v1','status':'complete',
        'plan_sha256':json_sha(payload),'inventory':invref,'kernel':kernel,
        'full_population':{'train_identities':1878,'val_identities':486,'train_videos':794,'val_videos':183},
        'eligible_train_identities':1084,'parent_plan_not_recomputed':True,'natural_GT_accessed':False}
    freeze_json(root/'compatibility.json',compatibility)
    if phase=='inventory':return invref
    rows=[r for r in payload['population'] if (r['split']=='val' if phase=='dev' else r['split']=='train' and payload['targets'][r['key']]['eligible'])]
    if len(rows)!=(486 if phase=='dev' else 1084):raise ValueError('required full cache population mismatch')
    store=UnitStore(root/'natural','natural');progress=Progress(root,phase,len(rows)+(224 if phase=='train' else 0));records=[]
    tasks=[(cfg,row,str(root),kernel) for row in rows]
    completed={}
    for record,timing in ordered_task_results(tasks,workers):
        completed[record['identity']]=record
        if timing['cache_hit']:progress.data['hits']+=1
        progress.completed(timing)
    records=[completed[row['key']] for row in rows]
    manifest={'schema':MANIFEST_SCHEMA,'status':'complete','phase':phase,'inventory':invref,
        'population_signature':population_signature(payload),'kernel':kernel,
        'natural':{'root':str(store.root.resolve()),'records':records}}
    if phase=='train':
        xstore=UnitStore(root/'x0','x0');adapter=X0AnalyticAdapterNTR1(cfg['x0_root']);xrecords=[]
        for source in range(32):
            binding=x0_binding(cfg,kernel,source)
            missing=[xstore.load(f'U{source:04d}_B{b}',binding) is None for b in range(7)]
            prepared=prepare_x0_unit(source,adapter) if any(missing) else None
            for block in range(7):
                key=f'U{source:04d}_B{block}';hit=xstore.load(key,binding)
                receipt=xstore.write(key,binding,x0_arrays(prepared[block])) if hit is None else hit[1]
                xrecords.append(unit_record(xstore,key,binding,receipt))
                progress.completed({'phase':'x0','key':key,'cache_hit':hit is not None,'bytes':receipt['bytes']})
        manifest['x0']={'root':str(xstore.root.resolve()),'records':xrecords}
        manifest['supervision']={'root':str((root/'supervision').resolve()),'records':inv['supervision']}
    ref=freeze_json(root/(phase+'_ready.json'),manifest)
    progress.update(force=True,phase=phase+'_complete',manifest=ref)
    return ref


def read_manifest(cfg,payload,path,phase,*,verify_inputs=True):
    manifest=json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema')!=MANIFEST_SCHEMA or manifest.get('status')!='complete' or manifest['phase']!=phase:
        raise ValueError('wrong/incomplete prepared manifest')
    if manifest['population_signature']!=population_signature(payload) or manifest['kernel']!=kernel_binding():
        raise ValueError('prepared manifest input/kernel mismatch')
    load_and_verify_inventory(cfg,payload,manifest['inventory'],verify_inputs=verify_inputs)
    expected=[r['key'] for r in payload['population'] if (r['split']=='val' if phase=='dev' else r['split']=='train' and payload['targets'][r['key']]['eligible'])]
    if [r['identity'] for r in manifest['natural']['records']]!=expected:
        raise ValueError('prepared cache does not cover exact ordered population')
    return manifest


def load_train_cache(cfg,payload,manifest_path):
    from pams.temporac.native_training_runtime_v1 import PreparedUnit,NOLACache
    from pams.temporac.cue import Window
    m=read_manifest(cfg,payload,manifest_path,'train')
    ns=UnitStore(m['natural']['root'],'natural');ss=UnitStore(m['supervision']['root'],'supervision')
    supervision={r['identity']:r for r in m['supervision']['records']}
    if set(supervision)!={r['identity'] for r in m['natural']['records']}:raise ValueError('supervision population mismatch')
    population={r['key']:r for r in payload['population']};natural={}
    for record in m['natural']['records']:
        key=record['identity'];row=population[key];paths=paths_for(cfg,row)
        if record['binding'] != natural_binding(row,m['kernel']): raise ValueError('natural binding differs')
        if supervision[key]['binding'] != supervision_binding(row,m['kernel']): raise ValueError('supervision binding differs')
        identity=read_derived(paths['feature_sha256'],paths['derived_sha256'],expected_N=row['N'],expected_key=key,for_routing=False)
        natural[key]=cached_natural_unit(key,identity,load_unit(ns,record),load_unit(ss,supervision[key]))
    xs=UnitStore(m['x0']['root'],'x0');xr=m['x0']['records']
    expected=[f'U{s:04d}_B{b}' for s in range(32) for b in range(7)]
    if [r['identity'] for r in xr]!=expected:raise ValueError('X0 cache must cover 224 original views')
    units={s:[] for s in range(32)}
    for record in xr:
        key=record['identity'];source=int(key[1:5])
        if record['binding'] != x0_binding(cfg,m['kernel'],source): raise ValueError('X0 binding differs')
        a=load_unit(xs,record)
        unit=PreparedUnit(key,a['response_input'],a['run_bounds'],tuple(Window(*map(int,w)) for w in a['windows']),a['gates'],
            NOLACache(a['denominator'],a['responsibility'],a['support']),a['pulse'],a['chi'],a['loss_mask'],a['traversal_bounds'],'X0')
        units[source].append(unit)
    return natural,{s:tuple(u) for s,u in units.items()}


def iter_dev_cache(cfg,payload,manifest_path):
    m=read_manifest(cfg,payload,manifest_path,'dev',verify_inputs=False);store=UnitStore(m['natural']['root'],'natural')
    population={r['key']:r for r in payload['population']}
    for record in m['natural']['records']:
        row=population[record['identity']];paths=paths_for(cfg,row)
        if record['binding'] != natural_binding(row,kernel_binding()): raise ValueError('natural binding differs')
        for field in ('feature_sha256','bridge_receipt_sha256','derived_sha256'):
            if file_sha(paths[field]) != row['bindings'][field]: raise ValueError('prediction NB1 input changed')
        identity=read_derived(paths['feature_sha256'],paths['derived_sha256'],expected_N=row['N'],expected_key=row['key'],for_routing=False)
        yield row['key'],identity,load_unit(store,record)
