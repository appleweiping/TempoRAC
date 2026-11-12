"""Saved-log accounting. Server CPU/stdlib only; no model/checkpoint imports."""
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from collections import Counter
import hashlib,json,re,shutil,statistics,zipfile
R=Path('/root/autodl-tmp/TempoRAC').resolve()
O=R/'resume20261112/completed-cost-accounting'
assert R==Path('/root/autodl-tmp/TempoRAC') and O.resolve().is_relative_to(R)
assert not (O/'cost_table.json').exists(), 'Never overwrite prior completed accounting'
O.mkdir(exist_ok=True)
RAW=O/'raw'; RAW.mkdir(exist_ok=True)
manifest={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def record(p,copy=True):
 p=Path(p); assert p.resolve().is_relative_to(R) and p.is_file()
 rel=str(p.relative_to(R)); manifest[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
 if copy:
  assert p.suffix not in {'.pt','.pth','.pyth','.npz','.npy'}
  dest=RAW/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 return p
def load(p):return json.loads(record(p).read_text())
def rows(p):return [json.loads(x) for x in record(p).read_text().splitlines() if x.strip()]
def dt(x):return datetime.fromisoformat(x)
def duration(a,b):return (dt(b)-dt(a)).total_seconds()
def decimal_sum(xs):return float(sum((Decimal(str(x)) for x in xs),Decimal(0)))
def write(name,o):(O/name).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
S=R/'source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb'
for f in ['temporac_cvpr/train.py','temporac_cvpr/eval_saved.py','temporac_cvpr/extract_features.py','temporac_cvpr/escounts_official.py','temporac_cvpr/efficiency.py']:record(S/f)
control=R/'resume20261112/continuation'
record(control/'continuation_worker.py')
events={g:rows(control/f'worker_gpu{g}.events.jsonl') for g in [0,1]}
for g in [0,1]:load(control/f'launch_gpu{g}.json')
train=[]; epoch_detail=[]; bounds=[]; sizes=[]
for seed in [0,1,2]:
 name=f'full_s{seed}';d=R/'runs/frozen_known_source_v1_resume20261112'/name
 cfg=load(d/'config.json');done=load(d/'complete_known_source.json');logs=rows(d/'log.jsonl');record(d/'stdout.log')
 assert [x['epoch'] for x in logs]==list(range(50)) and cfg['cfg']['epochs']==done['epochs']==50
 assert done['source_commit']=='489dc77b91a21d21215784ff49961edb20dab4fb'
 gpu=done['gpu']; ev=events[gpu]; parts={}
 for phase in ['train','eval_rc_test','eval_cx_test','eval_ucf_test','eval_quva']:
  starts=[e for e in ev if e.get('run')==name and e.get('phase')==phase and e['status']=='started']
  exits=[e for e in ev if e.get('run')==name and e.get('phase')==phase and e['status']=='exited']
  assert len(starts)==len(exits)==1 and exits[0]['returncode']==0 and starts[0]['pid']==exits[0]['pid']
  a,b=starts[0],exits[0];parts[phase]={'start_utc':a['utc'],'exit_observed_utc':b['utc'],'seconds':duration(a['utc'],b['utc']),'pid':a['pid']}
  bounds.append(dict(run=name,gpu=gpu,phase=phase,**parts[phase]))
 for tag in ['rc_test','cx_test','ucf_test','quva']:
  record(d/f'eval_{tag}.log')
 for x in logs:epoch_detail.append({'run':name,'epoch':x['epoch'],'logged_train_loop_seconds':x['time'],'cumulative_peak_allocated_gib':x['gpu_mem_gb']})
 ck=[]
 for p in sorted(d.glob('*.pt')):
  entry={'run':name,'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'receipt_sha256':done['checkpoints'].get(p.name),'hash_source':'existing completion receipt; checkpoint bytes never opened'}
  sizes.append(entry);ck.append(entry)
 train.append({'run':name,'seed':seed,'gpu':gpu,'epochs':50,'params_decoder':cfg['params'],'epoch_loop_seconds_sum':decimal_sum(x['time'] for x in logs),'epoch_loop_seconds_mean':statistics.mean(x['time'] for x in logs),'peak_allocated_gib_logged':max(x['gpu_mem_gb'] for x in logs),'training_process_wall_seconds':parts['train']['seconds'],'full_eval_process_wall_seconds_sum':sum(p['seconds'] for k,p in parts.items() if k!='train'),'eval_populations':done['eval_populations'],'phase_bounds':parts,'checkpoint_sizes':ck,'feature_roots':[cfg['args']['features']]+cfg['cfg']['aux_features'],'inference':'fixed original+shift continuous-count average; existing cached block8+final-block features; target validation-selected checkpoints'})
# Official RepNet saved records retain timings of the process in which each was computed.
B=R/'external-baselines-20261112';V=B/'runs/repnet-full357-v4'; A=load(V/'adoption_receipt.json');P=rows(V/'predictions.jsonl');C=load(V/'completion.json');cfg=load(V/'config.json');restore=load(V/'restore.json')
assert len(P)==357 and A['adopted_n']==196 and A['remaining_n']==161
adopted={(x['dataset'],x['split'],x['video_id']) for x in A['records']}
assert len(adopted)==196
for f in ['evaluate_repnet.py','repnet_official_core.py','env-spec.json']:record(B/'repnet-runner'/f)
rep_groups={}
for label,selected in [('all357',P),('reused196',[x for x in P if (x['dataset'],x['split'],x['video_id']) in adopted]),('new161',[x for x in P if (x['dataset'],x['split'],x['video_id']) not in adopted])]:
 def summary(rr):return {'n':len(rr),'decode_seconds_sum':sum(x['decode_seconds'] for x in rr),'inference_seconds_sum':sum(x['inference_seconds'] for x in rr),'decode_plus_inference_seconds_sum':sum(x['decode_seconds']+x['inference_seconds'] for x in rr)}
 rep_groups[label]=dict(**summary(selected),by_population={pop:summary([x for x in selected if x['population']==pop]) for pop in sorted({x['population'] for x in selected})})
rep_events=[]
for v in [1,2,3,4]:
 es=rows(B/f'runs/official-worker-v{v}/events.jsonl')
 for a in [e for e in es if e['status']=='child_started']:
  bs=[e for e in es if e['status']=='child_exited' and e.get('mode')==a.get('mode')]
  assert len(bs)==1;b=bs[0]
  rep_events.append({'worker_version':v,'mode':a['mode'],'returncode':b['returncode'],'start_utc':a['utc'],'exit_observed_utc':b['utc'],'seconds':duration(a['utc'],b['utc'])})
rep_weights=[]
for name,digest in cfg['checkpoint_sha256'].items():
 p=Path(cfg['checkpoint_dir'])/name;rep_weights.append({'name':name,'bytes':p.stat().st_size,'receipt_sha256':digest,'hash_source':'inherited config; weight bytes never opened'})
rep={'official_version':'updated2025 ckpt-70','source_commit':'8facb655a897e01b421bb68b4214f65178f7b009','training_performed_here':False,'timing_groups':rep_groups,'process_history':rep_events,'v4_load_seconds':restore.get('load_seconds'),'v4_tf_peak_memory_bytes':C['peak_memory']['peak'],'released_checkpoint_sizes':rep_weights,'gpu_description_saved':C['gpu_before']['gpu_snapshot'],'inference_config':cfg['inference'],'cost_qualification':'all357 sums preserve timings from multiple processes; only new161 incurred model execution in v4. v4 wall includes restore, adoption/verification, decoding, official backbone+counting inference, trace writes and scoring. Failed v1-v3 and witness costs separately retained; none is a standalone full357 fresh-run timing.'}
# Legacy ESCounts and cache logs have rounded progress checkpoints, not final timings.
escounts=[]
for name in ['escounts_official','escounts_official_ucfrep','escounts_official_quva']:
 d=R/'runs/baselines'/name;rr=rows(d/'records.jsonl');load(d/'result.json');segments=[]
 for p in sorted(d.glob('stdout*.log')):
  lines=record(p).read_text(errors='replace').splitlines();progress=[x for x in lines if re.search(r'\d+/\d+ .*min$',x)]
  if progress:segments.append({'path':str(p.relative_to(R)),'first_progress_line':progress[0],'last_progress_line':progress[-1],'last_reported_minutes':float(re.search(r'([\d.]+)min$',progress[-1])[1]),'full_job_duration_available':False})
 escounts.append({'run':name,'records_n':len(rr),'split_counts':dict(Counter(x['split'] for x in rr)),'per_video_timing_available':False,'full_job_wall_seconds':None,'progress_segments':segments,'qualification':'rounded cumulative time since that invocation loop start; resumes skip existing records, RC includes validation+test. Last progress does not necessarily cover final videos. No segment sum used as full inference duration.'})
record(R/'logs/escounts_extra.sh')
preprocess=[]
for pat in ['extract_b8_*.log','extract_b8rest_*.log']:
 for p in sorted((R/'logs').glob(pat)):
  lines=record(p).read_text(errors='replace').splitlines();progress=[x for x in lines if re.match(r'\[\d+\] \d+/\d+ .*min$',x)]
  if not progress:continue
  last=progress[-1]; m=re.match(r'\[(\d+)\] (\d+)/(\d+) .* ([\d.]+)min$',last)
  preprocess.append({'path':str(p.relative_to(R)),'shard':int(m[1]),'last_progress_n':int(m[2]),'shard_n':int(m[3]),'last_reported_minutes':float(m[4]),'final_item_timed':m[2]==m[3],'reported_complete':any('EXTRACT_DONE' in x for x in lines),'last_progress_line':last})
for f in ['chain_blocks.sh','chain_b8_rest.sh','extract_b8rest_g0.done','extract_b8rest_g1.done']:record(R/'logs'/f)
pre={'block8_progress_records':preprocess,'exact_total_feature_build_seconds':None,'qualification':'historical block8 extraction covers dataset-wide train/val/test caches, shared across many methods/runs. Progress is rounded to 0.1min and excludes encoder setup; some shards do not time final items. RepCount/UCF b8rest adds r16, Countix/QUVA b8rest builds r2/r4/r8/r16. Auxiliary final-block base extraction and pool/cache transforms not fully timed; do not sum these into exact end-to-end cost. Original GPU/runtime identity for these historical builds not independently bound here.'}
result={'schema':'temporac.saved_completed_cost.v1','created_utc':datetime.now(timezone.utc).isoformat(),'scope':'completed correctedfull3, official RepNet updated2025 full357, inherited ESCounts; no active-ablation performance read','execution':'server CPU stdlib aggregation only; no model/checkpoint/feature arrays opened','prior_accounting':'bounded search found efficiency.py benchmark source, no saved cost/efficiency result table in docs, logs, runs/baselines, completed correctedfull3 or RepNet evidence; existing timing fields reused','training':train,'training_totals':{'epochs':150,'epoch_loop_seconds_sum':decimal_sum(t['epoch_loop_seconds_sum'] for t in train),'training_process_wall_seconds_sum':sum(t['training_process_wall_seconds'] for t in train),'full_eval_process_wall_seconds_sum':sum(t['full_eval_process_wall_seconds_sum'] for t in train),'train_plus_full_eval_process_wall_seconds_sum':sum(t['training_process_wall_seconds']+t['full_eval_process_wall_seconds_sum'] for t in train)},'repnet':rep,'escounts':escounts,'preprocessing':pre,'timing_semantics':{'epoch_loop':'train.py lines486-523: time.time duration of training minibatch loop, rounded0.1s; validation/checkpointing excluded, feature extraction excluded; not CUDA-event kernel time','process_bounds':'worker started/exited observer timestamps include Python/import, cached-feature loading, validation, checkpoint IO and final validation outputs. 3s poll means exit detection can lag actual exit, and the first checkpoint observation adds IO. Sum of one-GPU process walls is GPU-assigned wall time, not measured active-GPU utilization or billed time. Same GPU family saved as RTX4090D; resource/co-run conditions still differ.','repnet_inference':'monotonic interval begins before official get_counts and ends after trace npz write, so includes count/postprocess and trace persistence; not pure kernel latency. Decode interval includes RGB read, crop/validation/checks.','memory':'TempoRAC peak allocated is rounded PyTorch max_memory_allocated/2**30 cumulative; RepNet TF allocator peak bytes differs. Neither equals nvidia-smi process residency; no cross-framework memory superiority claim.'},'unavailable':{'end_to_end_temporac_feature_plus_decode_inference_seconds':None,'exact_shared_cache_preprocessing_cost':None,'escounts_full_input_inference_wall_seconds':None,'matched_kernel_latency_throughput_flops':None,'monetary_cost':'no verified billing rate/invoice in scope','total_project_compute':'excludes29 historical runs, paused lost-weight epochs, unrelated diagnostics/cache retries and active arms; not a project lifetime GPU total'},'claims':{'saved_cost_accounting':'complete for available logs','conference_scale_fair_efficiency_comparison':'partial; unfulfilled matched end-to-end efficiency matrix','fair_speedup_claim_supported':False,'semantic_review':'not invoked; deterministic evidence collection only; actual backend unattested; same-family provisional applies to later semantic judgment'}}
write('cost_table.json',result);write('epoch_times.json',epoch_detail);write('process_bounds.json',bounds);write('checkpoint_sizes.json',sizes);write('SOURCE_MANIFEST.json',manifest)
text=['# Completed-run saved-log cost accounting','', 'All numeric aggregation ran on server CPU using existing small logs/configs. No model/checkpoint loading, inference, training or cache construction occurred. No active-ablation performance was read.','', '| Completed run | Epochs | Logged train-loop sum (s) | Training process wall (s) | Four full-eval processes (s) | Logged PyTorch peak (GiB) |','|---|---:|---:|---:|---:|---:|']
for t in train:text.append(f"| {t['run']} |50|{t['epoch_loop_seconds_sum']:.1f}|{t['training_process_wall_seconds']:.6f}|{t['full_eval_process_wall_seconds_sum']:.6f}|{t['peak_allocated_gib_logged']:.2f}|")
t=result['training_totals'];text+=['',f"Total:150 epochs; logged training-loop {t['epoch_loop_seconds_sum']:.1f}s; three training processes {t['training_process_wall_seconds_sum']:.6f}s ({t['training_process_wall_seconds_sum']/3600:.6f} GPU-assigned wall-hours);12 full-eval processes {t['full_eval_process_wall_seconds_sum']:.6f}s. These are decoder/cache-based costs, excluding upstream RGB/feature extraction.", '', '| RepNet saved timing scope | N | Decode sum (s) | Official inference + trace sum (s) |','|---|---:|---:|---:|']
for k,x in rep_groups.items():text.append(f"| {k} |{x['n']}|{x['decode_seconds_sum']:.6f}|{x['inference_seconds_sum']:.6f}|")
v4=[x for x in rep_events if x['worker_version']==4][0];text+=['',f"RepNet v4 evaluation child wall {v4['seconds']:.6f}s covers only161 new inputs plus196-record adoption/setup/IO. Full357 sums combine retained source-process timings; no fresh357 benchmark was run. TF allocator peak {rep['v4_tf_peak_memory_bytes']}bytes. All failed attempts and witness intervals are retained in cost_table.json.",'','ESCounts has rounded progress markers only: RC validation+test resumed segments, UCF last101/105 at4.2min, QUVA last81/100 at7.3min. Full job and per-video times are unavailable; no exact full-population total is inferred. Historical block8 cache progress and source shell commands are retained; partial rounded progress is not a complete preprocessing bill.','','Timing, memory and storage qualifications are explicit in cost_table.json. Decoder parameter count is3,878,788 from saved config; checkpoint sizes are stat-only and their existing receipt hashes are reused. Full backbone parameters/FLOPs were not recomputed. No monetary cost is stated without a billing rate.','','The single remaining efficiency matrix need: on one documented GPU/software environment, all actually compared full methods on the same complete357 RC152/UCF105/QUVA100 inputs, retain each official frozen inference view/window setting; report RGB decode/preprocess, feature/backbone, decoder/postprocess, total cold-input end-to-end wall and steady-state wall separately, per-dataset N and frame workloads, identical I/O inclusion and warm-up rule, measured peak process/allocator memory, parameter/FLOP scope, and one-time cache-build/storage cost with an explicit reuse count. TempoRAC requires both final-block and block8 features plus fixed two-view decoder; RepNet requires all official stride searches; ESCounts requires all official windows. Existing logs cannot provide this matched comparison; no new benchmark is authorized by this accounting.','','Acceptance: saved timing/epoch/process/adoption evidence PASS; cache/end-to-end/fair conference efficiency PARTIAL; no speedup claim supported. This is evidence gathering, not a fresh semantic/integrity review.']
(O/'COST_REPORT.md').write_text('\n'.join(text)+'\n')
write('OUTPUT_SHA256.json',{str(p.relative_to(O)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in O.rglob('*') if p.is_file() and p.name!='OUTPUT_SHA256.json'})
with zipfile.ZipFile(O/'completed_cost_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in O.rglob('*'):
  if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(O)))
print(json.dumps({'training_totals':result['training_totals'],'repnet_groups':rep_groups,'report':str(O/'COST_REPORT.md'),'archive_sha256':sha(O/'completed_cost_evidence.zip')},indent=2))
