"""Durable finite official training and full evaluation; root launches GPU phase."""
from __future__ import annotations
import copy
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from native_engine import (capture_rng, restore_rng, primitive_checks, optimizer_and_scheduler,
                           virtual_train_step, Primitive)
from native_inputs import (load_inputs, original_loader, sampler_checks, NativeDataset,
                           POPULATIONS, ASSETS, key, collate_native)

R=Path('/root/autodl-tmp/TempoRAC')
BACKBONE=ASSETS/'swin_tiny_patch244_window877_kinetics400_1k.pth'
CODE_FILES=['run.py','native_engine.py','native_inputs.py','native_run.py','env-spec.json']

def shell(*args):
    return subprocess.check_output(args,text=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip()

def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value

def source_identity(contract,digest):
    source=Path(contract['identity']['source_path'])
    assert shell('git','-C',str(source),'rev-parse','HEAD')==contract['identity']['source_commit']
    assert not shell('git','-C',str(source),'status','--porcelain','--untracked-files=no'), 'official tree modified'
    files=shell('git','-C',str(source),'ls-files').splitlines()
    return {'commit':contract['identity']['source_commit'],
            'files':{f:digest(source/f) for f in files}}

def environment_identity(digest,work):
    packages={name:importlib.metadata.version(name) for name in (
        'torch','torchvision','numpy','scipy','opencv-python-headless','mmcv','timm','einops','kornia')}
    expected={'torch':'2.5.1','torchvision':'0.20.1','numpy':'1.26.4','scipy':'1.14.1',
              'opencv-python-headless':'4.10.0.84','mmcv':'1.4.0','timm':'0.4.12','einops':'0.3.2','kornia':'0.5.11'}
    assert packages==expected,(packages,expected)
    import torchvision
    assert torch.__version__=='2.5.1+cu124' and torchvision.__version__=='0.20.1+cu124'
    assert sys.version_info[:2]==(3,10)
    assert Path(sys.prefix).resolve()==R/'envs/transrac-native310'
    assert Path(sys.base_prefix).resolve()==R/'env'
    distributions=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())
    return {'python':sys.version,'executable':sys.executable,'packages':packages,
            'distributions':distributions,'torch_cuda':torch.version.cuda,
            'module_versions':{'torch':torch.__version__,'torchvision':torchvision.__version__},
            'spec_sha256':digest(work/'env-spec.json'),
            'read_only_base_prefix':sys.base_prefix,
            'resolved_torch_file':torch.__file__,
            'torch_init_sha256':digest(torch.__file__),
            'torch_extension_sha256':digest(torch._C.__file__)}

def canonical_module(contract,digest):
    spec=contract['final_evaluation']['canonical_scorer']
    assert digest(spec['server_path'])==spec['sha256']
    module_spec=importlib.util.spec_from_file_location('transrac_frozen_common_metrics',spec['server_path'])
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module

def build_model(contract):
    import hashlib
    assert BACKBONE.stat().st_size==contract['model']['backbone_weights']['bytes']
    h=hashlib.sha256()
    with open(BACKBONE,'rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    assert h.hexdigest()==contract['model']['backbone_weights']['sha256'], 'never load unverified backbone'
    source=Path(contract['identity']['source_path'])
    sys.path.insert(0,str(source))
    from models.TransRAC import TransferModel
    assert Path(inspect.getfile(TransferModel)).resolve()==source/'models/TransRAC.py'
    torch.manual_seed(1)
    model=TransferModel(config=str(source/'configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'),
                        checkpoint=str(BACKBONE),num_frames=64,scales=[1,4,8],OPEN=False)
    assert model.OPEN is False and model.scales==[1,4,8]
    return model

def backbone_coverage(model,contract,digest):
    spec=contract['model']['backbone_weights']
    assert BACKBONE.stat().st_size==spec['bytes'] and digest(BACKBONE)==spec['sha256']
    loaded=torch.load(BACKBONE,map_location='cpu',weights_only=False)
    state=loaded.get('state_dict',loaded)
    # The author's load_checkpoint strips only a leading module. prefix. Check
    # every actual backbone parameter and persistent buffer after that mapping.
    state={k.removeprefix('module.'):v for k,v in state.items()}
    coverage=[]
    for name,value in model.backbone.state_dict().items():
        source_key='backbone.'+name
        assert source_key in state,source_key
        src=state[source_key]
        assert src.shape==value.shape and torch.equal(src,value),(source_key,tuple(src.shape),tuple(value.shape))
        coverage.append({'name':name,'shape':list(value.shape),'elements':value.numel(),'exact':True})
    unmatched=[k for k in state if k.startswith('backbone.') and k[len('backbone.'):] not in model.backbone.state_dict()]
    assert not unmatched,unmatched
    return {'status':'PASS','backbone_path':str(BACKBONE),'sha256':digest(BACKBONE),
            'entries':coverage,'loaded_elements':sum(r['elements'] for r in coverage),
            'counting_checkpoint_loaded':False,'graph_file':inspect.getfile(type(model.backbone)),
            'forward_executed':False}

def atomic_torch(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def epoch_resume_check(work):
    """Exercise production checkpoint serialization and complete-epoch RNG state."""
    torch.manual_seed(1);random.seed(1);np.random.seed(1)
    model=Primitive().double();opt,scheduler=optimizer_and_scheduler(model)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    train_gen=torch.Generator().manual_seed(1);valid_gen=torch.Generator().manual_seed(1)
    streams=[torch.Generator().manual_seed(1).get_state() for _ in range(4)]
    x=torch.randn(32,3,2,2,2,dtype=torch.float64);y=torch.randn(32,64,dtype=torch.float64)
    virtual_train_step(model,opt,scaler,x,y,streams);scheduler.step()
    payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
             'scaler':scaler.state_dict(),'rng':capture_rng(train_gen,valid_gen,streams),'completed_epoch':0,
             'logical_steps':1,'actual_optimizer_steps':1,'best':None,'log_offsets':{'fixture':0},'identity':{'fixture':True}}
    checkpoint=work/'checks/epoch_resume_fixture.pt';atomic_torch(checkpoint,payload)
    virtual_train_step(model,opt,scaler,x,y,streams);scheduler.step()
    final=copy.deepcopy(model.state_dict());final_opt=copy.deepcopy(opt.state_dict())
    after=(torch.rand(5),torch.randperm(29,generator=train_gen),torch.randperm(17,generator=valid_gen),random.random(),np.random.rand())
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
    restored=Primitive().double();rop,rs=optimizer_and_scheduler(restored)
    restored.load_state_dict(loaded['model']);rop.load_state_dict(loaded['optimizer']);rs.load_state_dict(loaded['scheduler'])
    sc=torch.amp.GradScaler('cuda',enabled=False);sc.load_state_dict(loaded['scaler'])
    streams=restore_rng(loaded['rng'],train_gen,valid_gen)
    virtual_train_step(restored,rop,sc,x,y,streams);rs.step()
    again=(torch.rand(5),torch.randperm(29,generator=train_gen),torch.randperm(17,generator=valid_gen),random.random(),np.random.rand())
    assert all(torch.equal(final[k],v) for k,v in restored.state_dict().items())
    for p,state in final_opt['state'].items():
        for k,value in state.items():
            assert torch.equal(value,rop.state_dict()['state'][p][k])
    assert all(torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b for a,b in zip(after,again))
    assert rs.state_dict()==scheduler.state_dict()
    return {'status':'PASS','epoch_checkpoint_restore':'bitwise CPU primitive parameters/Adam/scheduler/RNG/generators',
            'CUDA_and_AMP_parity':'not tested; no GPU used'}

def binding_identity(work,contract,digest,env=None):
    return {'contract_sha256':digest(work/'RUN_CONTRACT.json'),
            'code':{name:digest(work/name) for name in CODE_FILES},
            'official_source':source_identity(contract,digest),
            'backbone_sha256':digest(BACKBONE),'input_receipt_sha256':digest(work/'inputs/IDENTITIES.json'),
            'canonical_scorer_sha256':contract['final_evaluation']['canonical_scorer']['sha256'],
            'environment':env if env else environment_identity(digest,work)}

def cpu_preflight(work,contract,digest,atomic_json):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    env=environment_identity(digest,work)
    source=source_identity(contract,digest)
    rows=load_inputs(contract,digest)
    primitives=primitive_checks();atomic_json(work/'checks/primitives.json',primitives)
    resume=epoch_resume_check(work);atomic_json(work/'checks/epoch_resume.json',resume)
    checks=sampler_checks(work,contract,rows,digest);atomic_json(work/'checks/sampler.json',checks)
    assert len(json.load(open(work/'inputs/IDENTITIES.json'))['entries'])==1244
    model=build_model(contract)
    coverage=backbone_coverage(model,contract,digest);atomic_json(work/'checks/backbone.json',coverage)
    model.train()
    assert model.backbone.training and model.bn1.training and model.bn2.training
    assert not model.OPEN
    scorer=canonical_module(contract,digest)
    metric=scorer.count_metrics([2.5,3.5,-2.,1.],[2.,4.,1.,0.])
    assert metric['n']==4 and metric['n_pos']==3
    # Source provenance of all imported official modules, including native model
    # and loader graph. No synthetic mmaction package/approximate backbone.
    imported={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
              if name.split('.')[0] in ('models','mmaction','dataset') and getattr(module,'__file__',None)}
    assert all(Path(p).is_relative_to(Path(contract['identity']['source_path'])) for p in imported.values())
    receipt={'status':'PASS','scope':'server CPU only, no official network forward, no GPU kernel or training pilot',
             'environment':env,'source':source,'official_import_graph':imported,
             'primitives':primitives,'epoch_resume':resume,'sampler':checks,
             'backbone_entries':len(coverage['entries']),'input_rows':len(rows),
             'identity':binding_identity(work,contract,digest,env),'GPU_readiness':'pending formal launch',
             'fresh_review':'root-owned; pending'}
    assert not torch.cuda.is_initialized()
    atomic_json(work/'checks/CPU_PREFLIGHT.json',receipt)

def score_outputs(rows,scorer):
    predictions=[r['continuous_count'] for r in rows]
    truth=[np.nan if r['scalar_gt'] is None else r['scalar_gt'] for r in rows]
    common=scorer.count_metrics(predictions,truth)
    zero=[r for r in rows if r['scalar_gt']==0]
    common['zero_count']={'n':0} if not zero else scorer.count_metrics([r['continuous_count'] for r in zero],[0]*len(zero))
    # Native predictions were rounded in their native tensor dtype during
    # inference. Native GT remains boundary-derived for RC, scalar for transfers.
    g=np.asarray([r['native_gt'] for r in rows],dtype=np.float64)
    p=np.asarray([r['native_rounded_prediction'] for r in rows],dtype=np.float64)
    err=np.abs(p-g)
    native={'n':len(rows),'mae':float(np.mean([r['native_normalized_error'] for r in rows])),
            'obo':float(np.mean([r['native_within_one'] for r in rows])),
            'obz':float(np.mean(err==0)),'rmse':float(np.sqrt(np.mean((p-g)**2))),
            'meaning':'RC original density-derived GT; UCF/QUVA original scalar GT; native tensor rounded predictions'}
    return clean_json({'harmonized':common,'native':native})

def evaluate(model,dataset,scorer,device,validation=False,generator=None,existing=None,commit=None):
    model.eval()
    # Official final test defaults to batch1; validation retains logical32 with
    # torch.chunk into up to4 eval shards. No extra outer autocast here.
    existing={} if existing is None else existing
    remaining=[i for i,r in enumerate(dataset.rows) if key(r) not in existing]
    assert not validation or not existing
    loader=DataLoader(dataset,batch_size=32 if validation else 1,shuffle=validation,
                      sampler=None if validation else remaining,
                      generator=generator,num_workers=2,pin_memory=False,
                      prefetch_factor=1,persistent_workers=False,collate_fn=collate_native)
    records=list(existing.values());batches=[]
    with torch.no_grad():
        for inputs,targets,indices in loader:
            batch_records=[];batch_losses=[]
            for x,y,ids in zip(torch.chunk(inputs,4 if validation else 1),
                               torch.chunk(targets,4 if validation else 1),
                               torch.chunk(indices,4 if validation else 1)):
                output,_=model(x.float().to(device))
                assert output.shape==(len(x),64)
                if not bool(torch.isfinite(output).all()):raise FloatingPointError('nonfinite evaluation density')
                continuous=output.sum(dim=1)  # BEFORE any precision conversion
                pred_native=continuous.round()
                native_gt=y.sum(dim=1).float().round() if validation else y.sum(dim=1).round()
                if dataset.rows[int(ids[0])]['dataset']!='RepCountA':
                    native_gt=torch.tensor([dataset.rows[int(i)]['count'] for i in ids],dtype=torch.float32)
                native_gt_device=native_gt.to(device)
                native_gap=abs(pred_native-native_gt_device)
                native_relative=native_gap/(native_gt_device+.1)
                if dataset.rows[int(ids[0])]['dataset']=='RepCountA':
                    batch_losses.append((len(x),float(torch.nn.functional.mse_loss(output,y.float().to(device)))))
                for j,idx in enumerate(ids.tolist()):
                    row=dataset.rows[idx]
                    g=float(native_gt[j]) if row['dataset']=='RepCountA' else float(row['count'])
                    pred=float(continuous[j]);gt=row['count']
                    record={'key':key(row),'video_id':row['video_id'],'population':row['population'],
                            'source_video_sha256':row['source_video_sha256'],'frame_range':row['frame_range'],
                            'continuous_count':pred,'density':output[j].cpu().tolist(),
                            'density_dtype':str(output.dtype),'sum_dtype':str(continuous.dtype),
                            'native_rounded_prediction':float(pred_native[j]),'native_gt':g,
                            'native_gt_dtype':str(native_gt.dtype),
                            'native_normalized_error':float(native_relative[j]),
                            'native_within_one':bool(native_gap[j]<=1),
                            'scalar_gt':gt,'common_rounded_prediction':float(np.round(pred)),
                            'common_error':None if gt is None else abs(float(np.round(pred))-gt)}
                    records.append(record);batch_records.append(record)
                    if commit is not None:commit(record)
                del output
            # Original validation diagnostics: continuous prediction after native
            # sum then FloatTensor, density-GT rounded after FloatTensor. Source
            # averages batch means without weighting the final batch of3.
            p=torch.tensor([r['continuous_count'] for r in batch_records],dtype=torch.float32)
            g=torch.tensor([r['native_gt'] for r in batch_records],dtype=torch.float32)
            batches.append({'n':len(p),'native_continuous_mae':float((abs(p-g)/(g+.1)).mean()),
                            'native_continuous_obo':float((abs(p-g)<=1).float().mean()),
                            'density_mse':sum(n*loss for n,loss in batch_losses)/len(p) if batch_losses else None})
    assert len(records)==len(dataset) and {r['key'] for r in records}=={key(r) for r in dataset.rows}
    metrics=score_outputs(records,scorer)
    if validation:
        metrics['native_validation_batchmean_diagnostics']={
            'mae':float(np.mean([b['native_continuous_mae'] for b in batches])),
            'obo':float(np.mean([b['native_continuous_obo'] for b in batches])),
            'density_mse':float(np.mean([b['density_mse'] for b in batches])),
            'batches':batches,'selection_uses_this':False}
    return records,metrics

def append_jsonl(stream,value):
    stream.write(json.dumps(clean_json(value),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def resource_gate(gpu):
    # Runtime ownership gate, not an automatic queue hold or scheduler. Root must
    # finish current arm+eval before invoking this finite worker.
    listing=shell('nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits')
    selected=next(line for line in listing.splitlines() if int(line.split(',')[0])==gpu)
    fields=[s.strip() for s in selected.split(',')]
    if int(fields[2])>=500:raise RuntimeError('released lane not free: '+selected)
    return {'physical_gpu':gpu,'uuid':fields[1],'memory_used_mib_at_gate':int(fields[2]),
            'all_gpu_snapshot':listing,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']}

def train_evaluate(work,contract,digest,atomic_json,args):
    import fcntl
    lock=open(work/'run.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=work/'attempts'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    try:
        _train_evaluate(work,contract,digest,atomic_json,args,attempt)
    except BaseException as exc:
        atomic_json(attempt/'FAILURE.json',{'status':'FAILED','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'error':repr(exc),'traceback':traceback.format_exc(),'no_automatic_retry':True})
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def _train_evaluate(work,contract,digest,atomic_json,args,attempt):
    preflight=json.load(open(work/'checks/CPU_PREFLIGHT.json'))
    assert preflight['status']=='PASS'
    identity=binding_identity(work,contract,digest)
    assert identity==preflight['identity'],'CPU preflight stale'
    # Root installs this exact review receipt after its one bounded fresh Sol
    # review. Mere presence of old checks never authorizes a new code revision.
    review=json.load(open(work/'ENGINEERING_REVIEW.json'))
    assert review['status'] in ('clean','PASS') and review['code']==identity['code']
    assert review['contract_sha256']==identity['contract_sha256']
    assert review['complete'] is True
    resources=resource_gate(args.gpu)
    assert torch.cuda.device_count()==1
    device=torch.device('cuda:0')
    resources['device_name']=torch.cuda.get_device_name(0)
    assert '4090' in resources['device_name']
    atomic_json(attempt/'START.json',{'identity':identity,'resource_gate':resources,
                'resume':args.resume,'pid':os.getpid(),'label':contract['identity']['label']})
    random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    model=build_model(contract).to(device)
    optimizer,scheduler=optimizer_and_scheduler(model)
    scaler=torch.cuda.amp.GradScaler()
    train_gen=torch.Generator().manual_seed(1);valid_gen=torch.Generator().manual_seed(1)
    streams=[torch.Generator(device=device).manual_seed(1).get_state() for _ in range(4)]
    rows=load_inputs(contract,digest);scorer=canonical_module(contract,digest)
    train_dataset=NativeDataset(work,[r for r in rows if r['population']=='rc_train'])
    valid_dataset=NativeDataset(work,[r for r in rows if r['population']=='rc_valid'])
    latest=work/'checkpoints/latest.pt'
    completed=-1;logical=0;actual=0;skipped=0;best=None;history=[]
    if args.resume:
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['identity']==identity,'resume exact identity mismatch'
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        for optimizer_state in optimizer.state.values():
            for k,v in optimizer_state.items():
                if isinstance(v,torch.Tensor) and k!='step':optimizer_state[k]=v.to(device)
        streams=restore_rng(state['rng'],train_gen,valid_gen)
        completed=state['completed_epoch'];logical=state['logical_steps'];actual=state['actual_optimizer_steps']
        skipped=state['overflow_skips'];best=state['best'];history=state['epoch_history']
        assert logical==24*(completed+1) and actual+skipped==logical
        assert [r['epoch'] for r in history]==list(range(completed+1))
        # Recover a best written inside the atomic latest if a crash occurred
        # between latest commit and materializing independently durable best.pt.
        if state.get('best_model') is not None:
            best_path=work/'checkpoints/best.pt'
            materialized=torch.load(best_path,map_location='cpu',weights_only=False) if best_path.exists() else None
            matches=materialized is not None and materialized['best']==best and materialized['identity']==identity
            if matches:
                matches=all(torch.equal(materialized['model'][k],v) for k,v in state['best_model'].items())
            if not matches:
                if best_path.exists():
                    import shutil
                    shutil.copy2(best_path,attempt/'uncommitted_best_preserved.pt')
                atomic_torch(best_path,{'model':state['best_model'],'best':best,'identity':identity})
    elif latest.exists() or (work/'COMPLETION.json').exists():
        raise RuntimeError('Existing run requires root-explicit --resume; never silently overwrite')
    best_model=None
    if best is not None:
        best_state=torch.load(work/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        assert best_state['best']==best and best_state['identity']==identity
        best_model=best_state['model']
    steps=open(attempt/'steps.jsonl','a');epochs=open(attempt/'epochs.jsonl','a')
    candidates=set(range(51,199,3))
    for epoch in range(completed+1,200):
        started=time.monotonic();epoch_ids=[];losses=[];start_actual=actual;start_skipped=skipped
        lr=optimizer.param_groups[0]['lr']
        expected_lr=8e-6*.8**(1+epoch//40)
        assert abs(lr-expected_lr)<1e-19
        loader=DataLoader(train_dataset,batch_size=32,shuffle=True,generator=train_gen,
                          num_workers=2,pin_memory=False,drop_last=False,prefetch_factor=1,persistent_workers=False,
                          collate_fn=collate_native)
        for batch_idx,(inputs,targets,ids) in enumerate(loader):
            t=time.monotonic()
            epoch_ids.extend(ids.tolist())
            result=virtual_train_step(model,optimizer,scaler,inputs,targets.float(),streams)
            logical+=1;actual+=int(result['optimizer_updated']);skipped+=int(result['overflow_skip'])
            losses.append(result['loss']);result.pop('prediction')
            append_jsonl(steps,{'epoch':epoch,'batch':batch_idx,'logical_step':logical,'actual_optimizer_steps':actual,
                          'batch_size':len(ids),'row_indices':ids.tolist(),'lr':lr,'seconds':time.monotonic()-t,**result})
            print(f'EPOCH {epoch} BATCH {batch_idx+1}/24 logical={logical} mse={result["loss"]:.8g}',flush=True)
        assert sorted(epoch_ids)==list(range(756)) and len(losses)==24
        train_seconds=time.monotonic()-started
        val_seconds=0.;validation_metrics=None
        new_best=False
        if epoch>50:
            vt=time.monotonic()
            predictions,validation_metrics=evaluate(model,valid_dataset,scorer,device,True,valid_gen)
            val_seconds=time.monotonic()-vt
            atomic_json(attempt/f'validation/epoch{epoch:03d}.json',{'epoch':epoch,'candidate':epoch in candidates,
                        'predictions':predictions,'metrics':validation_metrics,'identity':identity})
            if epoch in candidates:
                common=validation_metrics['harmonized']
                assert common['n']==131 and common['n_pos']==130
                selection=[common['mae_p'],-common['obo_p'],epoch]
                # Hash model tensor bytes in deterministic order for candidate
                # identity without writing50 redundant full checkpoints.
                import hashlib
                h=hashlib.sha256()
                for name,tensor in model.state_dict().items():
                    h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
                    h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
                candidate={'epoch':epoch,'selection_key':selection,'model_tensor_sha256':h.hexdigest(),
                           'validation_file':str(attempt/f'validation/epoch{epoch:03d}.json'),
                           'validation_file_sha256':digest(attempt/f'validation/epoch{epoch:03d}.json')}
                atomic_json(attempt/f'candidates/epoch{epoch:03d}.json',candidate)
                atomic_json(work/f'candidates/epoch{epoch:03d}.json',candidate)
                if best is None or selection<best['selection_key']:
                    best=candidate;new_best=True
                    best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step()
        entry={'epoch':epoch,'training_rows':756,'logical_batches':24,'lr':lr,'next_lr':optimizer.param_groups[0]['lr'],
               'train_mse_batchmean':float(np.mean(losses)),'successful_updates':actual-start_actual,
               'overflow_skips':skipped-start_skipped,'train_seconds':train_seconds,'validation_seconds':val_seconds,
               'total_seconds':time.monotonic()-started,'validation_metrics':validation_metrics,'best':best}
        history.append(entry);append_jsonl(epochs,entry)
        state={'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
               'rng':capture_rng(train_gen,valid_gen,streams),'completed_epoch':epoch,
               'logical_steps':logical,'actual_optimizer_steps':actual,'overflow_skips':skipped,'best':best,
               'best_model':best_model,'epoch_history':history,'log_offsets':{
                   'steps':{'path':str(attempt/'steps.jsonl'),'bytes':steps.tell()},
                   'epochs':{'path':str(attempt/'epochs.jsonl'),'bytes':epochs.tell()}},
               'validation_completed':epoch>50,'attempt':str(attempt)}
        # latest is the single epoch transaction, including best weights. A best
        # from an uncommitted failed epoch never changes the committed selector.
        atomic_torch(latest,state)
        if new_best:atomic_torch(work/'checkpoints/best.pt',{'model':best_model,'best':best,'identity':identity})
        atomic_json(work/'PROGRESS.json',{'status':'training','completed_epoch':epoch,'logical_steps':logical,
                    'actual_optimizer_steps':actual,'overflow_skips':skipped,'best':best,
                    'latest_sha256':digest(latest),'attempt':str(attempt),'last_epoch':entry})
        if epoch==0:
            atomic_json(work/'FIRST_FORMAL_EPOCH_COST.json',{'based_on':'first complete formal epoch, no pilot',
                        'training_seconds':train_seconds,'estimated_remaining_training_GPU_hours':199*train_seconds/3600,
                        'validation_cost':'unmeasured until first scheduled full validation after epoch51',
                        'complete200_required':True,'reservation_GPU_hours':[24,48],
                        'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(0)})
        print(f'COMMITTED_EPOCH {epoch} logical={logical} updates={actual} skipped={skipped}',flush=True)
        del state
    steps.close();epochs.close()
    assert len(history)==200 and [r['epoch'] for r in history]==list(range(200))
    assert logical==4800 and actual+skipped==4800 and best is not None
    candidate_files=[work/f'candidates/epoch{e:03d}.json' for e in sorted(candidates)]
    assert len(candidate_files)==50 and all(p.exists() for p in candidate_files)
    model.load_state_dict(best_model,strict=True)
    checkpoint_sha=digest(work/'checkpoints/best.pt')
    final={};outputs=[]
    for population in ('rc_test','ucf_test','test'):
        dataset=NativeDataset(work,[r for r in rows if r['population']==population])
        existing={}
        for sample in dataset.rows:
            path=work/'final/records'/sample['population']/(sample['video_id']+'.json')
            if path.exists():
                saved=json.load(open(path))
                assert saved['identity']==identity and saved['best']==best and saved['checkpoint_sha256']==checkpoint_sha
                assert saved['row']['key']==key(sample) and saved['row']['source_video_sha256']==sample['source_video_sha256']
                assert saved['row']['scalar_gt']==sample['count'] and saved['row']['frame_range']==sample['frame_range']
                existing[key(sample)]=saved['row']
        def commit(row):
            row['checkpoint_sha256']=checkpoint_sha;row['selected_epoch']=best['epoch']
            row['input_receipt_sha256']=identity['input_receipt_sha256']
            path=work/'final/records'/row['population']/(row['video_id']+'.json')
            atomic_json(path,{'row':row,'identity':identity,'best':best,'checkpoint_sha256':checkpoint_sha})
        predictions,metrics=evaluate(model,dataset,scorer,device,existing=existing,commit=commit)
        # Recompute from saved raw rows after write, independently of inference.
        path=work/f'final/{population}_predictions.json'
        atomic_json(path,{'population':population,'rows':predictions,'identity':identity,
                         'checkpoint_sha256':checkpoint_sha,'best':best})
        reloaded=json.load(open(path))['rows']
        assert score_outputs(reloaded,scorer)=={k:v for k,v in metrics.items() if k!='native_validation_batchmean_diagnostics'}
        atomic_json(work/f'final/{population}_metrics.json',metrics)
        final[population]={'n':len(predictions),'metrics':metrics,'predictions_sha256':digest(path)}
        outputs.extend(predictions)
    assert len(outputs)==357 and len({r['key'] for r in outputs})==357
    receipt={'status':'COMPLETE','completed_epochs':200,'logical_optimizer_opportunities':logical,
             'successful_optimizer_updates':actual,'AMP_overflow_skips':skipped,'final_population_total':357,
             'best':best,'best_checkpoint_sha256':checkpoint_sha,'latest_sha256':digest(latest),
             'identity':identity,'final':final,'candidate_identities':{p.name:digest(p) for p in candidate_files},
             'label':contract['identity']['label'],'scientific_quality_or_SOTA_acceptance':False,
             'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'attempt':str(attempt)}
    atomic_json(work/'COMPLETION.json',receipt)
    print('FULL200_FULL357_COMPLETE',flush=True)
