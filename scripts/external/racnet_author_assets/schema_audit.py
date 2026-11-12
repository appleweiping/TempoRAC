"""Inspect preserved author NPZ bytes; never alias, synthesize, clamp or generate arrays."""
import hashlib,math,zipfile
from pathlib import PurePosixPath
import numpy as np

def inspect_npz(path,engine,kind,name):
    limit=engine.c['max_nested_npz_bytes'];arrays={};errors=[];total=0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            assert not info.is_dir() and info.filename.endswith('.npy') and '..' not in PurePosixPath(info.filename).parts and not PurePosixPath(info.filename).is_absolute() and '\\' not in info.filename,'unsafe NPZ member'
            key=info.filename[:-4];assert key not in arrays,'duplicate NPZ key';total+=info.file_size;assert total<=limit,'nested NPZ expansion cap'
            with z.open(info) as f:
                version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version,max_header_size=65536)
                header_end=f.tell();nbytes=math.prod(shape)*dtype.itemsize
                meta=dict(shape=list(shape),dtype=str(dtype),fortran_order=fortran,array_bytes=nbytes,header_bytes=header_end,crc32=f'{info.CRC:08x}')
                assert not dtype.hasobject and dtype.kind in 'biuf','nonobject real numeric arrays required'
                assert nbytes+header_end==info.file_size and nbytes<=limit,'NPY byte/shape mismatch'
                h=hashlib.sha256();payload_hash=hashlib.sha256();f.seek(0);head=f.read(header_end);h.update(head)
                while True:
                    engine.disk_write(0);block=f.read(1024*1024)
                    if not block:break
                    h.update(block);payload_hash.update(block)
                    assert len(block)%dtype.itemsize==0,'unaligned numeric array';assert np.isfinite(np.frombuffer(block,dtype=dtype)).all(),'nonfinite author array'
                meta.update(npy_sha256=h.hexdigest(),array_payload_sha256=payload_hash.hexdigest(),CRC_and_finite_verified=True);arrays[key]=meta
        scalar={}
        with np.load(path,allow_pickle=False) as data:
            for key in ['count','length']:
                if key in arrays:
                    assert math.prod(arrays[key]['shape'])==1,f'{key} not scalar'
                    value=data[key].item();assert isinstance(value,(int,float)) and math.isfinite(value)
                    scalar[key]=value
    stem=PurePosixPath(name).name;vid=stem[:-12] if stem.endswith('_feature.npz') else stem[:-4]
    if kind=='UCF':
        vid=engine.c['ucf_aliases'].get(vid,vid);key='ucf_test/'+vid
    else:
        parts=PurePosixPath(name).parts;roles=[p for p in parts if p in ['train','valid','test']]
        assert len(roles)==1,'Unknown RC member role';key='rc_'+roles[0]+'/'+vid
    ref=engine.references.get(key);applies=ref is not None and (not key.startswith('rc_train/') or key in engine.native_keys)
    candidates={k:v for k,v in arrays.items() if k in ['frames','img_feature'] and len(v['shape'])>=2 and v['shape'][0]>0 and math.prod(v['shape'][1:])==37632}
    Ts={v['shape'][0] for v in candidates.values()}
    if applies and not candidates:errors.append('Missing actual documented feature payload (frames OR img_feature with37632 values/frame)')
    if kind=='RC' and applies:
        if 'gt_tsm' not in arrays:errors.append('Missing author gt_tsm for applicable RC member')
        elif not any(arrays['gt_tsm']['shape']==[T,T] for T in Ts):errors.append('Author gt_tsm shape incompatible with featureT')
    if 'length' in scalar and Ts and scalar['length'] not in Ts:errors.append('Author length metadata conflicts with actual featureT')
    if kind=='UCF':
        if ref is None:errors.append('UCF member outside explicit105mapping')
        if 'count' not in scalar:errors.append('Missing actual UCF scalar count')
        elif ref and scalar['count']!=ref['count']:errors.append('UCF embedded count differs from frozen scalar')
        if ref and Ts and not any(T in [ref['full_N'],ref['ROI_N']] for T in Ts):errors.append('UCF featureT matches neither inherited full_N nor ROI_N')
    elif ref and Ts and ref['full_N'] not in Ts:errors.append('RC featureT differs from inherited decoded full_N')
    gaps=[]
    if 'frames' not in arrays and 'img_feature' in candidates:gaps.append('Literal native frames key absent; real img_feature preserved, future adapter needed')
    if 'length' not in arrays:gaps.append('Literal length key absent; actual featureT and frozen metadata recorded, no array length synthesized')
    return dict(qualified_key=key,applies_to_future_native_population=applies,excluded_from_optimization='missing_native_scalar_and_cycles' if ref and ref['count'] is None else 'outside_source_corrected_population' if key.startswith('rc_train/') and not applies else None,
        arrays=arrays,scalar_metadata=scalar,candidate_feature_keys=list(candidates),feature_Ts=sorted(Ts),nested_uncompressed_bytes=total,
        interface_gaps=gaps,asset_errors=errors,length_relations={} if not ref else {str(T):dict(matches_full_N=T==ref['full_N'],matches_ROI_N=T==ref['ROI_N']) for T in Ts},
        content_or_normalization_parity='UNRESOLVED',raw_arrays_unchanged=True)
