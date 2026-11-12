import pathlib,re,difflib,json,hashlib
B=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites')
p=B/'vendor/mmaction2-0.24.1/mmaction/datasets/pipelines/loading.py'
s=p.read_text();n=re.sub(r'np\.int\b','int',s)
if n!=s:
 p.with_suffix('.py.author').write_text(s);p.write_text(n)
 (B/'MMACTION_COMPAT.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),n.splitlines(True),fromfile='mmaction2-0.24.1/mmaction/datasets/pipelines/loading.py',tofile='private-compatible/mmaction/datasets/pipelines/loading.py')))
for p in (B/'candidate').rglob('*.py'):
 s=p.read_text();n=re.sub(r'np\.int\b','int',s)
 if n!=s:p.write_text(n)
