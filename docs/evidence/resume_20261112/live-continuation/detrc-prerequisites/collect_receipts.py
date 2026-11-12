import pathlib,hashlib,json,difflib,importlib.metadata as meta,sys,os,subprocess,datetime,tarfile,re
B=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites');C=B/'candidate';S=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261111/code/DeTRC')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
expected=json.loads((B/'EXPECTED_AUTHOR_SHA.json').read_text(encoding='utf-8-sig'))
sourcecheck=[dict(path=k,expected=v,actual=sha(S/k)) for k,v in expected.items()]
assert all(r['expected']==r['actual'] for r in sourcecheck)
(B/'IMMUTABLE_SOURCE_RECHECK.json').write_text(json.dumps(dict(files=len(sourcecheck),matched=len(sourcecheck),checks=sourcecheck),indent=2))
rows=[];patch=[];manifest=[]
for rel in sorted(set(expected)|{'DeTRC/durable_state.py','DeTRC/config/project_tsn_full80.py'}):
 p=C/rel;old=S/rel
 if not p.exists():raise FileNotFoundError(p)
 manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
 if not old.exists() or p.read_bytes()!=old.read_bytes():
  rows.append(dict(path=rel,before_sha256=sha(old) if old.exists() else None,after_sha256=sha(p)))
  a=old.read_text().splitlines(True) if old.exists() else []
  patch.extend(difflib.unified_diff(a,p.read_text().splitlines(True),fromfile='author/'+rel,tofile='candidate/'+rel))
(B/'CANDIDATE_DIFF.patch').write_text(''.join(patch));(B/'CHANGED_FILES.json').write_text(json.dumps(rows,indent=2));(B/'CANDIDATE_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
native=[]
for rel in ['DeTRC/model/grid_sample1d/grid_sample1d_cuda.cpp','DeTRC/model/grid_sample1d/grid_sample1d_cuda_kernel.cu','DeTRC/model/roi_align/src/roi_align_cuda.cpp','DeTRC/model/roi_align/src/roi_align_kernel.cu']:
 s=(S/rel).read_text();n=(C/rel).read_text()
 transformed=s.replace('.type().is_cuda()', '.is_cuda()').replace('AT_DISPATCH_FLOATING_TYPES(input.type(),','AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),').replace('AT_DISPATCH_FLOATING_TYPES(grad.type(),','AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),').replace('.data<scalar_t>()','.data_ptr<scalar_t>()')
 assert transformed==n,rel
 native.append(dict(path=rel,source_sha256=sha(S/rel),compiled_source_sha256=sha(C/rel),api_only_exact_equality=True))
for p in sorted(C.glob('DeTRC/model/*/*.so')):native.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
(B/'NATIVE_BUILD_RECEIPT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build_script_sha256=sha(B/'build_native.sh'),build_log_sha256=sha(B/'native-build.log'),argv=['bash',str(B/'build_native.sh')],exit_code=0,api_changes_only=native,cuda_dispatches=0,kernel_readiness='unverified until authorized full real run; no witness run'),indent=2))
packages=[]
for name in sorted({x.metadata['Name'] for x in meta.distributions()},key=str.lower):
 d=meta.distribution(name);record=pathlib.Path(d._path)/'RECORD'
 packages.append(dict(name=name,version=d.version,dist_info=str(d._path),record_sha256=sha(record) if record.exists() else None))
(B/'ENV_RESOLVED.json').write_text(json.dumps(packages,indent=2))
spec=json.loads((B/'ENV_SPEC.json').read_text(encoding='utf-8-sig'));canon=json.dumps(spec,sort_keys=True,separators=(',',':')).encode()
envrec=dict(spec_sha256=hashlib.sha256(canon).hexdigest(),python=sys.version,executable=sys.executable,base_prefix=sys.base_prefix,venv_config=(B/'env/pyvenv.cfg').read_text(),cuda_compiler=subprocess.check_output([str(B/'cuda-12.4/bin/nvcc'),'--version'],text=True),gcc=subprocess.check_output(['/usr/bin/g++','--version'],text=True).splitlines()[0],packages=packages,private_wheels=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((B/'wheelhouse').glob('*'))],author_environment_known=False,validation='CPU mapping/build/import/interface only; GPU witness prohibited and not run')
(B/'ENV_LOCK.json').write_text(json.dumps(envrec,indent=2))
# Full config schema and equality of scientifically frozen fields, no data open.
sys.path[:0]=[str(C),str(B/'vendor/mmaction2-0.24.1')]
from mmcv import Config
original=Config.fromfile(str(C/'DeTRC/config/repcount_tsn_feature_enc_contrastive.py'));project=Config.fromfile(str(C/'DeTRC/config/project_tsn_full80.py'))
for key in ['model','optimizer','lr_config','total_epochs','evaluation','clip_len','frame_interval_list','train_stride_rate','test_stride_rate']:
 assert project[key]==original[key],key
pretty=project.pretty_text
(B/'RESOLVED_FULL80_CONFIG.py').write_text(pretty)
cli={}
for name in ['train','test','extract']:
 log=(B/(name+'-entry-help.log')).read_text();assert 'Traceback' not in log and 'usage:' in log.lower(),name
 cli[name]=dict(log_sha256=sha(B/(name+'-entry-help.log')),exit_code=0)
(B/'ENTRY_AND_CONFIG_CHECKS.json').write_text(json.dumps(dict(entrypoints=cli,frozen_scientific_fields_equal=True,source_configs=['repcount_tsn_feature_enc_contrastive.py','project_tsn_full80.py'],gpu_runs=0),indent=2))
with tarfile.open(B/'candidate-source.tar.gz','w:gz') as t:
 for row in manifest:t.add(C/row['path'],arcname=row['path'])
print(json.dumps(dict(source_files_unchanged=len(sourcecheck),candidate_files=len(manifest),changed_files=len(rows),env_spec_sha256=envrec['spec_sha256'],cpu_checks_sha256=sha(B/'CPU_INTERFACE_CHECKS.json'),candidate_manifest_sha256=sha(B/'CANDIDATE_SOURCE_MANIFEST.json')),indent=2))
