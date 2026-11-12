"""Immutable RGB input binding; official raw sampler and target are preserved."""
from __future__ import annotations
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ASSETS=Path('/root/autodl-tmp/TempoRAC/baselines/transrac-native-assets20261112')
EXPECTED={'manifest.json':'61cbbd8c98c7a9764284dd2f032d85d2cfb9403fedabf695fe90271f31fa2333',
          'train.csv':'b4a7c0e360d87e47de21e0f52af5ba1250be5ccfa3a500f05dfa4dbd9ed88c83',
          'valid.csv':'498087551ad3464f4bbb293e4701d16136be3f11daa37e7fd80a43ec01ab43ae',
          'test.csv':'b89b7f3ba54bc29c72d460fed67d94313a4a7844011b56887da81b8774a290ec'}
POPULATIONS={'rc_train':756,'rc_valid':131,'rc_test':152,'ucf_test':105,'test':100}
SAMPLER='official-raw-rgb64-start-aligned-floor-opencv-linear-u8-v1'

def key(row):
    return row['population'] + '/' + row['video_id']

def load_inputs(contract,digest):
    for name,sha in EXPECTED.items():
        assert digest(ASSETS/'inputs'/name)==sha, f'input identity mismatch: {name}'
    for item in contract['inputs']['annotation_files']:
        assert digest(item['path'])==item['sha256']
    assert digest(contract['inputs']['union_manifest']['path'])==contract['inputs']['union_manifest']['sha256']
    rows=json.load(open(ASSETS/'inputs/manifest.json'))
    assert len(rows)==1244
    assert len({key(r) for r in rows})==len(rows)
    counts={p:sum(r['population']==p for r in rows) for p in POPULATIONS}
    assert counts==POPULATIONS, counts
    missing=[r for r in rows if r['count'] is None]
    assert len(missing)==1 and missing[0]['video_id']=='stu6_44' and missing[0]['population']=='rc_train' and missing[0]['bounds']==[]
    assert not ({'stu4_50','stu3_27'} & {r['video_id'] for r in rows if r['population']=='rc_train'})
    for row in rows:
        assert Path(row['file']).stat().st_size==row['bytes'], key(row)
        assert len(row['source_video_sha256'])==64
        assert len(row['bounds'])%2==0
    # Prepared CSV matches every complete ordered boundary and original scalar,
    # including the explicitly missing scalar. No inferred zero annotation.
    for role in ('train','valid','test'):
        table={Path(r['name']).stem:r for r in csv.DictReader(open(ASSETS/'inputs'/f'{role}.csv'))}
        selected=[r for r in rows if r['population']==f'rc_{role}']
        assert set(table)=={r['video_id'] for r in selected}
        for row in selected:
            item=table[row['video_id']]
            bounds=[int(v) for k,v in item.items() if 'L' in k and v!='']
            assert bounds==row['bounds'],key(row)
            value=item['count']
            assert (value=='' and row['count'] is None) or (value!='' and float(value)==row['count']),key(row)
    return rows

def original_loader(source):
    sys.path.insert(0,str(source))
    return importlib.import_module('dataset.RepCountA_raw_Loader')

def sample_indices(length):
    assert length>0
    return [i*length//64 for i in range(64)] if length>=64 else list(range(length))+[length-1]*(64-length)

def decode_rgb(row):
    """Two complete streaming decodes avoid retaining raw full-video RGB RAM."""
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(row['file'])
    if not cap.isOpened():
        raise RuntimeError('cannot open '+row['file'])
    total=0
    while True:
        ok,_=cap.read()
        if not ok: break
        total+=1
    cap.release()
    roi=row.get('frame_range')
    start,end=(0,total-1) if roi is None else tuple(map(int,roi))
    assert 0<=start<=end<total,(key(row),roi,total)
    length=end-start+1
    local=sample_indices(length)
    absolute=[start+i for i in local]
    required=set(absolute)
    sampled={}
    cap=cv2.VideoCapture(row['file'])
    assert cap.isOpened()
    n=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if n in required:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            sampled[n]=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
        n+=1
    cap.release()
    assert n==total and len(sampled)==len(required),(key(row),n,total,len(sampled),len(required))
    array=np.stack([sampled[i] for i in absolute])
    assert array.shape==(64,3,224,224) and array.dtype==np.uint8
    return array,{'full_decoded_frames':total,'roi_decoded_frames':length,
                  'local_indices':local,'source_indices':absolute,'frame_range':roi,'sampler':SAMPLER}

def cache_paths(work,row):
    stem=work/'rgb'/row['population']/row['video_id']
    return stem.with_suffix('.npy'),stem.with_suffix('.json')

def sampler_sha():
    return hashlib.sha256((inspect.getsource(sample_indices)+inspect.getsource(decode_rgb)).encode()).hexdigest()

def target_tensor(loader,row,length):
    # Exactly the official torch.tensor(list) constructor, preserving its native
    # inferred float32 or all-integer-zero dtype before DataLoader collation.
    return torch.tensor(loader.preprocess(length,row['bounds'],num_frames=64))

def prepare_inputs(work,contract,digest,atomic_json,build_cache=False):
    rows=load_inputs(contract,digest)
    loader=original_loader(contract['identity']['source_path'])
    metadata=[]
    def prepare_one(row):
        path,side=cache_paths(work,row)
        if side.exists():
            info=json.load(open(side))
            assert info['sample']==row and info['sampler']==SAMPLER and info['sampler_sha256']==sampler_sha(),key(row)
            assert path.exists() and digest(path)==info['rgb_sha256'],key(row)
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            assert array.shape==(64,3,224,224) and array.dtype==np.uint8
        elif build_cache:
            # Partial files from an interrupted entry are never trusted/reused.
            # A sidecar is committed only after bytes, source hash and target.
            assert digest(row['file'])==row['source_video_sha256'],key(row)
            array,info=decode_rgb(row)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+'.tmp')
            with open(temp,'wb') as stream:
                np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,path)
            info.update(sample=row,rgb_sha256=digest(path),source_sha256_verified=True,sampler_sha256=sampler_sha())
            label=target_tensor(loader,row,info['roi_decoded_frames']) if row['dataset']=='RepCountA' else None
            info['native_target']=None if label is None else label.tolist()
            info['native_target_dtype']=None if label is None else str(label.dtype)
            info['native_target_sum']=None if label is None else float(label.sum())
            info['native_validation_gt']=None if label is None else float(label.sum().float().round())
            info['native_test_gt']=None if label is None else float(label.sum().round())
            atomic_json(side,info)
        else:
            raise RuntimeError(f'RGB cache absent for {key(row)}; explicit --build-rgb-cache required; no unconditional cache construction')
        return {'key':key(row),'sidecar_sha256':digest(side),'rgb_sha256':info['rgb_sha256'],
                         'full_N':info['full_decoded_frames'],'roi_N':info['roi_decoded_frames'],
                         'native_target_sum':info['native_target_sum'],'scalar_count':row['count']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for idx,entry in enumerate(pool.map(prepare_one,rows)):
            metadata.append(entry)
            if (idx+1)%25==0: print(f'INPUT_PREPARED {idx+1}/{len(rows)}',flush=True)
    differences=[r for r in metadata if r['native_target_sum'] is not None and
                 (r['scalar_count'] is None or round(r['native_target_sum'])!=r['scalar_count'])]
    atomic_json(work/'inputs/target_count_differences.json',differences)
    atomic_json(work/'inputs/manifest.json',rows)
    identity={'status':'PASS','rows':len(rows),'populations':POPULATIONS,
              'prepared_manifest_sha256':EXPECTED['manifest.json'],'sampler':SAMPLER,
              'sampler_source_sha256':digest(__file__),'entries':metadata}
    atomic_json(work/'inputs/IDENTITIES.json',identity)
    return identity

class NativeDataset(Dataset):
    def __init__(self,work,rows):
        self.work=work
        self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        path,side=cache_paths(self.work,row)
        info=json.load(open(side))
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        inputs=torch.from_numpy(np.array(array,copy=True)).float().div_(255).transpose(0,1)
        values=info['native_target']
        target=torch.tensor(values) if values is not None else torch.zeros(64)
        return inputs,target,index

def collate_native(batch):
    from torch.utils.data._utils.collate import default_collate
    inputs,targets,indices=zip(*batch)
    # Worker default_collate allocates target out= from the first item's dtype.
    # Explicit stack retains ordinary PyTorch promotion, including int64 for
    # all-integer zero targets and float32 for any native mixed-density batch.
    return default_collate(inputs),torch.stack(targets),default_collate(indices)

def sampler_checks(work,contract,rows,digest):
    loader=original_loader(contract['identity']['source_path'])
    checked=[]
    # Long and zero examples compare directly to original VideoRead. Bounded
    # representative equivalence only; full population identities remain above.
    reps=[next(r for r in rows if r['population']=='rc_valid' and r['count']>0),
          next(r for r in rows if r['population']=='rc_valid' and r['count']==0)]
    for row in reps:
        original=loader.VideoRead(row['file'],64)
        expected=original.crop_frame()
        path,side=cache_paths(work,row)
        cached=torch.from_numpy(np.array(np.load(path),copy=True)).float().div_(255)
        assert torch.equal(expected,cached),key(row)
        info=json.load(open(side))
        assert original.frame_length==info['full_decoded_frames']
        label=target_tensor(loader,row,original.frame_length)
        assert label.tolist()==info['native_target']
        checked.append({'key':key(row),'exact_rgb_tensor':True,'exact_target':True,'N':original.frame_length})
    # Short-clip branch and inclusive ROI branch use actual existing video bytes
    # copied losslessly to a temporary FFV1 fixture; never a performance subset.
    # Official VideoRead performs the fixture's full decode independently.
    roi=next(r for r in rows if r['population']=='ucf_test')
    cap=cv2.VideoCapture(roi['file']);frames=[]
    for _ in range(11):
        ok,frame=cap.read();assert ok;frames.append(frame)
    cap.release()
    fixture=work/'checks/short_raw.avi'
    writer=cv2.VideoWriter(str(fixture),cv2.VideoWriter_fourcc(*'FFV1'),20,(frames[0].shape[1],frames[0].shape[0]))
    assert writer.isOpened()
    for frame in frames:writer.write(frame)
    writer.release()
    row={'population':'fixture','video_id':'short','file':str(fixture),'frame_range':None}
    a,info=decode_rgb(row)
    ref=loader.VideoRead(str(fixture),64).crop_frame()
    assert torch.equal(torch.from_numpy(a.copy()).float().div_(255),ref)
    row['frame_range']=[2,8]
    a,info=decode_rgb(row)
    selected=[frames[i] for i in range(2,9)]
    expected=[]
    for i in sample_indices(len(selected)):
        expected.append(cv2.resize(cv2.cvtColor(selected[i],cv2.COLOR_BGR2RGB),(224,224)).transpose(2,0,1))
    assert np.array_equal(a,np.stack(expected))
    checked.append({'fixture':'short11/inclusiveROI2..8','short_original_exact':True,'ROI_exact':True})
    return {'status':'PASS','cases':checked,'full_populations':POPULATIONS}
