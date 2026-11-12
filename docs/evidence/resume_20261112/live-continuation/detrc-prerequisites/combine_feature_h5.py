"""Parameterize the released pickle -> H5 assembly; preserve all feature values."""
import argparse,json,pathlib,pickle,hashlib,os
import h5py
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--plan',required=True);p.add_argument('--population',required=True)
p.add_argument('--pickle-dir',required=True);p.add_argument('--output',required=True)
a=p.parse_args();plan=json.loads(pathlib.Path(a.plan).read_text())
rows=[r for r in plan['entries'] if r['population']==a.population]
assert rows and len({r['video_id'] for r in rows})==len(rows)
dest=pathlib.Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
tmp=dest.with_suffix(dest.suffix+'.pending')
receipt=[]
with h5py.File(tmp,'w') as f:
 for row in rows:
  src=pathlib.Path(a.pickle_dir)/(row['video_id']+'.pkl')
  with src.open('rb') as inp:arr=pickle.load(inp)
  arr=np.asarray(arr)
  assert arr.shape==(row['native_frames'],2048),(row['key'],arr.shape)
  assert arr.dtype==np.float32,(row['key'],arr.dtype)
  assert np.isfinite(arr).all(),row['key']
  ds=f.create_dataset(row['video_id'],data=arr)
  ds.attrs['source_video_sha256']=row['source_video_sha256']
  ds.attrs['backbone_sha256']='cc665e2a86857b4ff4873747ae4e98f9cd6de3f72111cf3379954f551cc63c49'
  ds.attrs['sampling']='native-full-T interval1; JPEG sequence1..T'
  receipt.append(dict(key=row['key'],shape=list(arr.shape),dtype=str(arr.dtype),pickle_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
 f.flush()
with tmp.open('rb') as f:os.fsync(f.fileno())
os.replace(tmp,dest)
pathlib.Path(str(dest)+'.receipt.json').write_text(json.dumps(dict(entries=receipt,h5_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
