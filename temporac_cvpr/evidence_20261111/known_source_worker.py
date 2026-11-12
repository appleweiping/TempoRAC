"""Finite authorized seed queue: fresh training, then one fixed full evaluation."""
import argparse, datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
R=Path('/root/autodl-tmp/TempoRAC'); E=R/'resume20261111analysis/known_source_v1'
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--runs',nargs='+',required=True);args=ap.parse_args()
spec=json.loads((E/'frozen_27_spec.json').read_text()); pf=json.loads((E/'preflight.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(pf['derived_manifest']))==pf['derived_sha256']
assert sha(E/'frozen_27_spec.json')==pf['spec_sha256']
allruns={r['name']:r for r in spec['runs']}
assert set(args.runs)<=set(spec['initial_release'])
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONPATH=str(E/'source'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(R/'tmp'))
py=str(R/'env/bin/python')
def event(status,**extra):
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_pid':os.getpid(),'gpu':args.gpu,'status':status,**extra}
    with (E/f'worker_gpu{args.gpu}.events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def execute(cmd,log,name,phase):
    with log.open('x') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=str(E/'source'),env=env)
        if phase=='train': log.with_name('pid').write_text(str(p.pid))
        event('started',run=name,phase=phase,pid=p.pid,command=cmd)
        code=p.wait()
    event('exited',run=name,phase=phase,pid=p.pid,returncode=code)
    if code:raise RuntimeError((name,phase,code))
for name in args.runs:
    r=allruns[name];d=R/'runs'/spec['group']/name
    # Never repeat partial or completed work. Operator diagnoses before resuming.
    assert not d.exists(),('output already exists; no automatic restart',str(d))
    gpu=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
    assert int(gpu)<500,('GPU acquired by another task',args.gpu,gpu)
    d.mkdir(parents=True)
    cmd=[py,'-u','-B','-m','temporac_cvpr.train','--manifest',*r['manifest'],'--features',r['features'],'--out',str(d),'--seed',str(r['seed']),'--cfg',json.dumps(r['cfg']),'--train-split',r['train_split'],'--val-split',r['val_split'],'--test-split',r['test_split']]
    execute(cmd,d/'stdout.log',name,'train')
    logs=[json.loads(s) for s in (d/'log.jsonl').read_text().splitlines() if s.strip()]
    assert [x['epoch'] for x in logs]==list(range(r['cfg']['epochs']))
    assert all(x.get('nonfinite_steps',0)==0 for x in logs)
    assert (d/'result.json').exists()
    got=json.loads((d/'config.json').read_text())
    assert got['cfg']==r['cfg'],('merged config drift',name)
    assert got['args']['seed']==r['seed'] and not got['args']['with_test']
    for val in ['rc_valid','cx_val','ucf_dev']:
        assert (d/f'best_{val}.pt').is_file()
        command=[py,'-u','-B','-m','temporac_cvpr.eval_saved','--run',str(d),'--manifest',*r['manifest'],'--features',r['features'],'--split',{'rc_valid':'rc_test','cx_val':'cx_test','ucf_dev':'ucf_test'}[val],'--tag',{'rc_valid':'rc_test','cx_val':'cx_test','ucf_dev':'ucf_test'}[val],'--ckpt',f'best_{val}.pt']
        execute(command,d/f'eval_{val}.log',name,'eval_'+val)
    command=[py,'-u','-B','-m','temporac_cvpr.eval_saved','--run',str(d),'--manifest',str(R/'data/QUVA/manifest.json'),'--features',str(R/'features/QUVA_stmae_b8_p4'),'--aux',str(R/'features/QUVA_stmae_p4'),'--split','test','--tag','quva','--ckpt','best_rc_valid.pt','--domain','RepCountA']
    execute(command,d/'eval_quva.log',name,'eval_quva')
    # Completion requires fixed original populations, not just process exit.
    expect={'rc_test':152,'cx_test':2403,'ucf_test':105,'quva':100}
    for tag,n in expect.items():
        ev=json.loads((d/f'eval_{tag}.json').read_text())
        assert ev['n']==n and len(ev['predictions'])==n and len(ev['predictions_shift'])==n
    receipt={'run':name,'gpu':args.gpu,'epochs':len(logs),'manifest_sha256':pf['derived_sha256'],'config_sha256':sha(d/'config.json'),'checkpoints':{p.name:sha(p) for p in d.glob('best*.pt')},'eval_populations':expect,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (d/'complete_known_source.json').write_text(json.dumps(receipt,indent=2))
    event('run_complete',run=name,receipt=receipt)
event('queue_complete',runs=args.runs)
