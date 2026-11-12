"""Atomic checkpoint hooks for the otherwise unchanged official TSN extractor."""
import os,pathlib,pickle
import numpy as np
from common import *
_CAPACITY=None

def context(output_file,length):
 work=pathlib.Path(os.environ['DETRC_WORK']);p=pathlib.Path(output_file)
 assert p.resolve().is_relative_to((work/'pickles').resolve()),p
 row=next(r for r in read(work/'inputs/production_plan.json')['entries'] if r['population']==p.parent.name and r['video_id']==p.stem)
 assert row['native_frames']==length
 raw=pathlib.Path(row['raw_dir'])/'IDENTITY.json'
 return work,p,row,dict(contract_sha256=binding(work),raw_identity_sha256=sha(raw),source_video_sha256=row['source_video_sha256'],backbone_sha256=BACKBONE_SHA,extractor_sha256=sha(work/'candidate/tools/data/activitynet/tsn_feature_extraction.py'))

def committed(output_file,length):
 work,p,row,identity=context(output_file,length);rec=p.with_suffix('.identity.json')
 if not rec.exists():
  # New numerical input: verify the actual complete JPEG byte identities once.
  rawdir=pathlib.Path(row['raw_dir']);j=read(rawdir/'IDENTITY.json')
  assert len(j['frames'])==length
  assert all(sha(rawdir/f['name'])==f['sha256'] for f in j['frames']),row['key']
  return False
 r=read(rec)
 assert r['identity']==identity,('feature identity differs',row['key'])
 assert p.exists() and sha(p)==r['pickle_sha256'],row['key']
 return True

def save_feature(output_file,feat,length):
 work,p,row,identity=context(output_file,length)
 arr=np.asarray(feat);assert arr.shape==(length,2048) and arr.dtype==np.float32 and np.isfinite(arr).all(),row['key']
 global _CAPACITY
 if _CAPACITY is None:_CAPACITY=Capacity(work)
 _CAPACITY.check(arr.nbytes+1048576)
 old=p.stat().st_size if p.exists() else 0
 temp=p.with_suffix('.pkl.pending')
 with temp.open('wb') as f:pickle.dump(feat,f);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 _CAPACITY.account(p.stat().st_size,old)
 atomic_json(p.with_suffix('.identity.json'),dict(key=row['key'],identity=identity,shape=list(arr.shape),dtype=str(arr.dtype),pickle_sha256=sha(p),bytes=p.stat().st_size))

 # Capacity includes each small durable feature identity receipt.
 _CAPACITY.account(p.with_suffix('.identity.json').stat().st_size)
