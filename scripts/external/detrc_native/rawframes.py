"""Exact released OpenCV -> short256 JPEG production, one atomic video at a time."""
import os,pathlib,time
from common import *

def resize_native(frame):
 import mmcv,numpy as np
 # Exact released build_rawframes.py new_short=256 branch, including int truncation.
 w,h,_=np.shape(frame)
 if min(h,w)==h:new_h=256;new_w=int((new_h/h)*w)
 else:new_w=256;new_h=int((new_w/w)*h)
 return mmcv.imresize(frame,(new_h,new_w))

def complete(row):
 path=pathlib.Path(row['raw_dir']);rec=path/'IDENTITY.json'
 if not rec.exists():return False
 r=read(rec)
 return r['contract_sha256']==binding(WORK) and r['producer_sha256']==sha(__file__) and r['source_video_sha256']==row['source_video_sha256'] and r['full_frames']==row['full_frames'] and r['native_frames']==row['native_frames'] and r['source_start']==row['source_start'] and r['source_end']==row['source_end'] and len(r['frames'])==row['native_frames'] and all((path/x['name']).is_file() and (path/x['name']).stat().st_size==x['bytes'] for x in r['frames'])

def build_video(row,capacity):
 import cv2,mmcv
 cv2.setNumThreads(1)
 if complete(row):return 'reused'
 path=pathlib.Path(row['raw_dir']);temp=path.with_name(path.name+'.pending');temp.mkdir(parents=True,exist_ok=True)
 assert not path.exists(),('incomplete final directory requires operator inspection',path)
 assert sha(row['file'])==row['source_video_sha256'],row['key']
 frames=[];n=0;start=time.monotonic();vr=mmcv.VideoReader(row['file'])
 for source_idx,frame in enumerate(vr):
  assert frame is not None,('official VideoReader decode failure',row['key'],source_idx)
  n=source_idx+1
  if not row['source_start']<=source_idx<=row['source_end']:continue
  local=source_idx-row['source_start']+1;name=f'img_{local:05d}.jpg';p=temp/name
  image=resize_native(frame);capacity.check(image.nbytes+65536)
  old=p.stat().st_size if p.exists() else 0
  assert mmcv.imwrite(image,str(p)),p
  size=p.stat().st_size;capacity.account(size,old)
  frames.append(dict(local_index=local,source_index=source_idx,name=name,bytes=size,sha256=sha(p)))
 assert n==row['full_frames'] and len(frames)==row['native_frames'],(row['key'],n,len(frames))
 # Reject stale frames from a failed incompatible prior attempt rather than mix them.
 assert {p.name for p in temp.glob('img_*.jpg')}=={x['name'] for x in frames}
 rec=dict(contract_sha256=binding(WORK),producer_sha256=sha(__file__),key=row['key'],source_video_sha256=row['source_video_sha256'],full_frames=n,native_frames=len(frames),source_start=row['source_start'],source_end=row['source_end'],preprocess='official mmcv.VideoReader BGR; exact author short256/int/imresize; default mmcv.imwrite JPEG; ROI0-based inclusive -> JPEG1..T',frames=frames,wall_seconds=time.monotonic()-start)
 old=(temp/'IDENTITY.json').stat().st_size if (temp/'IDENTITY.json').exists() else 0
 atomic_json(temp/'IDENTITY.json',rec);capacity.account((temp/'IDENTITY.json').stat().st_size,old)
 # Flush each JPEG before the directory completion marker becomes visible.
 for x in frames:
  with (temp/x['name']).open('rb') as f:os.fsync(f.fileno())
 os.replace(temp,path);fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return 'built'

def produce(work=WORK):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU JPEG stage must hide GPUs'
 assert os.getpriority(os.PRIO_PROCESS,0)>=10,'use the documented low-priority launcher'
 assert len(os.sched_getaffinity(0))<=2,'use two-CPU launcher affinity'
 plan=read(work/'inputs/production_plan.json');cap=Capacity(work);cap.check();stats={'built':0,'reused':0};start=time.monotonic()
 for row in plan['entries']:
  status=build_video(row,cap);stats[status]+=1
  atomic_json(work/'CPU_JPEG_PROGRESS.json',dict(contract_sha256=binding(work),completed=sum(stats.values()),total=len(plan['entries']),stats=stats,last_key=row['key'],derived_bytes=cap.used,wall_seconds=time.monotonic()-start,complete=sum(stats.values())==len(plan['entries'])))
 atomic_json(work/'RAWFRAMES_COMPLETE.json',dict(contract_sha256=binding(work),rows=1244,frames=sum(r['native_frames'] for r in plan['entries']),receipts={r['key']:sha(pathlib.Path(r['raw_dir'])/'IDENTITY.json') for r in plan['entries']}))
