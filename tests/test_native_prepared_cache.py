"""Artificial CPU byte-equivalence and cache integrity; no training or natural data."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from pams.temporac import native_bridge_v1 as nb
from pams.temporac.cue import _coordinate_spectrum
from pams.temporac.native_time_cue_v1 import compute_native_routing
from pams.temporac.native_training_runtime_v1 import precompute_nola
from pams.temporac.native_fast_cue_v1 import coordinate_spectrum_ordered, compute_local_routing
from pams.temporac.native_prepared_cache_v1 import (read_derived, UnitStore, prepare_natural_arrays, FEATURE_ARRAYS)


def exact(a,b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def feature(n=140, mode='normal'):
    base=np.stack((np.linspace(-.8,.8,17),np.linspace(.2,1,17)),axis=1)
    base[11],base[12],base[5],base[6]=(-.2,0),(.2,0),(-.3,.8),(.3,.8)
    x=np.zeros((n,17,3),np.float32); x[:,:,:2]=base
    t=np.arange(n)
    x[:,7:11,0]+=(.15*np.sin(2*np.pi*t/19))[:,None].astype(np.float32)
    x[:,13:17,1]+=(.12*np.cos(2*np.pi*t/19))[:,None].astype(np.float32)
    x[:,:,2]=.9
    observed=np.ones((n,17),bool)
    source=np.ones(n,bool)
    if mode=='missing':
        observed[10:15,0]=False; x[10:15,0]=0; source[65:68]=False
    pts=np.cumsum(np.where(t%7==0,34,33)).astype(np.int64)
    present=np.ones(n,bool) if mode=='pts' else np.zeros(n,bool)
    return nb.NativeFeatureV1(np.arange(32,dtype=np.uint8),np.array([0],np.int64),
        np.array([n],np.int64),np.arange(n,dtype=np.int64),x,observed,source,
        pts,present,np.array([1,1000],np.int64))


def derived_arrays(identity):
    g=identity.geometry_features
    arrays={'response_input':identity.response_input,'joint_usable':identity.joint_mask,
      'frame_derived_valid':identity.frame_valid,'observed_edge_support':identity.observed_edge_support,
      'response_edge_support':identity.response_edge_support,'edge_lengths':g.edge_lengths,
      'direction':g.direction,'direction_mask':g.direction_element_mask,
      'run_bounds':identity.run_bounds,'short_runs':identity.short_runs}
    arrays.update({'map_'+k:v for k,v in identity.mapping.items()})
    return arrays


@pytest.mark.parametrize('n',[3,17,127,2603])
def test_frequency_bytes(n):
    x=np.arange(n,dtype=np.float64)+.5+np.sin(np.arange(n)*.3)*.05
    w=np.sin(np.pi*np.arange(n)/(n-1))**2; w[0]=w[-1]=0
    for kind in range(4):
        v=np.zeros(n) if kind==0 else np.sin(x/(5+kind))+.2*np.cos(x/(2.3+kind/20))
        keep=np.arange(n)%5!=2 if kind==3 else np.ones(n,bool)
        a=_coordinate_spectrum(v[keep],x[keep],w[keep]); b=coordinate_spectrum_ordered(v[keep],x[keep],w[keep])
        if a is None: assert b is None
        else: exact(a,b)


@pytest.mark.parametrize('mode',['normal','missing','pts'])
def test_derived_local_nola_bytes(tmp_path,mode):
    f=feature(mode=mode); identity=nb.preprocess_native(f)
    fp=tmp_path/'f.npz'; dp=tmp_path/'d.npz'
    np.savez_compressed(fp,**f.as_arrays()); np.savez_compressed(dp,**derived_arrays(identity))
    restored=read_derived(fp,dp,expected_N=140,expected_key=f.opaque_key.tobytes().hex()+'_0')
    exact(identity.geometry_features.geometry,restored.geometry_features.geometry)
    exact(identity.geometry_features.edge_coordinate_mask,restored.geometry_features.edge_coordinate_mask)
    exact(identity.clock.q,restored.clock.q)
    exact(identity.response_input,restored.response_input)
    routing=compute_native_routing(identity); fast=compute_local_routing(restored)
    assert routing.windows==fast.windows
    exact(routing.local_gates,fast.local_gates)
    original=precompute_nola(routing.windows,routing.local_gates,identity.response_edge_support)
    cached,_=prepare_natural_arrays(restored)
    exact(cached['denominator'],original.denominator)
    exact(cached['responsibility'],original.responsibility)
    exact(cached['support'],original.support)


def test_unit_store_roundtrip_resume_and_corruption(tmp_path):
    store=UnitStore(tmp_path,'natural')
    arrays={'windows':np.empty((0,3),np.int64),'local_gates':np.empty((0,3),np.float64),
            'denominator':np.zeros(4,np.float64),'responsibility':np.zeros((4,3),np.float64),'support':np.zeros(4,bool)}
    b={'input_sha':'a','kernel':'b'}
    receipt=store.write('id',b,arrays)
    loaded,again=store.load('id',b)
    for name in arrays: exact(arrays[name],loaded[name])
    assert store.write('id',b,arrays)==receipt
    assert store.load('id',dict(b,input_sha='changed')) is None
    path=tmp_path/(receipt['key']+'.npz')
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='payload changed'): store.load('id',b)


def test_uncommitted_payload_and_target_separation(tmp_path):
    store=UnitStore(tmp_path,'natural'); binding={'source':'sha'}; key=store.key('id',binding)
    (tmp_path/(key+'.npz')).write_bytes(b'interrupted')
    assert store.load('id',binding) is None
    with pytest.raises(ValueError,match='feature-only'):
        store.write('id',binding,{'pulse':np.zeros(2)})


def test_fast_kernel_is_actually_used(monkeypatch):
    import pams.temporac.native_fast_cue_v1 as fast
    original=fast.frequency_ordered_sum
    calls=[]
    def count(terms):
        calls.append(terms.shape)
        return original(terms)
    monkeypatch.setattr(fast,'frequency_ordered_sum',count)
    x=np.arange(127,dtype=np.float64)+.5
    assert fast.coordinate_spectrum_ordered(np.sin(x/7),x,np.ones(127)) is not None
    assert calls==[(48,127),(48,127)]


def test_target_never_changes_natural_key():
    from pams.temporac.native_prepared_cache_v1 import natural_binding
    row={'N':17,'bindings':{'feature_sha256':'f','bridge_receipt_sha256':'b','derived_sha256':'d','target_sha256':'t1'}}
    first=natural_binding(row,{'kernel':'fixed'})
    row['bindings']['target_sha256']='t2'
    assert natural_binding(row,{'kernel':'fixed'})==first


def test_prediction_loader_never_validates_or_opens_targets(tmp_path,monkeypatch):
    import json
    import pams.temporac.cache_build_v1 as build
    from pams.temporac.native_prepared_cache_v1 import kernel_binding,natural_binding,file_sha
    f=feature();identity=nb.preprocess_native(f);key=f.opaque_key.tobytes().hex()+'_0'
    bridge=tmp_path/'bridge'/'identities'/'val';bridge.mkdir(parents=True)
    fp=bridge/(key+'.npz');dp=bridge/(key+'.derived.npz');bp=bridge/(key+'.json')
    np.savez_compressed(fp,**f.as_arrays());np.savez_compressed(dp,**derived_arrays(identity));bp.write_text('{}')
    row={'key':key,'N':140,'split':'val','bindings':{'feature_sha256':file_sha(fp),
        'derived_sha256':file_sha(dp),'bridge_receipt_sha256':file_sha(bp)}}
    payload={'population':[row],'targets':{key:{'eligible':False}}}
    cfg={'census':'unused','bridge_root':str(tmp_path/'bridge'),'target_root':str(tmp_path/'MUST_NOT_OPEN'),
        'component_map':'unused','x0_root':'unused','artifacts':{}}
    kernel=kernel_binding();store=UnitStore(tmp_path/'natural','natural');binding=natural_binding(row,kernel)
    arrays,_=prepare_natural_arrays(identity);receipt=store.write(key,binding,arrays)
    inv={'schema':build.INVENTORY_SCHEMA,'status':'complete','population_signature':build.population_signature(payload),
      'input_signature':build.input_signature(cfg),'kernel':kernel}
    ip=tmp_path/'inventory.json';ip.write_text(json.dumps(inv));ref={'path':str(ip),'sha256':file_sha(ip)}
    manifest={'schema':build.MANIFEST_SCHEMA,'status':'complete','phase':'dev','inventory':ref,
      'population_signature':build.population_signature(payload),'kernel':kernel,
      'natural':{'root':str(store.root),'records':[build.unit_record(store,key,binding,receipt)]}}
    mp=tmp_path/'dev.json';mp.write_text(json.dumps(manifest))
    def forbidden(*a,**k):raise AssertionError('predict tried full target validation')
    monkeypatch.setattr(build,'validate_full_inventory',forbidden)
    result=list(build.iter_dev_cache(cfg,payload,mp))
    assert len(result)==1 and result[0][0]==key
    exact(result[0][2]['local_gates'],arrays['local_gates'])


def test_progress_counts_verified_and_committed(tmp_path):
    from pams.temporac.cache_build_v1 import Progress
    p=Progress(tmp_path,'inventory',2)
    p.completed({'phase':'inventory','key':'a'})
    assert p.data['verified']==1 and p.data['committed']==0
    p.completed({'phase':'natural_unit','key':'a','bytes':12})
    assert p.data['verified']==1 and p.data['committed']==1 and p.data['bytes_committed']==12


def test_worker_count_and_serial_dispatch(monkeypatch):
    import pams.temporac.cache_build_v1 as b
    monkeypatch.setattr(b,'prepare_natural_record',lambda task:('receipt',task))
    assert list(b.ordered_task_results([1,2],1))==[('receipt',1),('receipt',2)]
    for workers in (0,4,True):
        with pytest.raises(ValueError,match='1..3'):list(b.ordered_task_results([],workers))
    if b.os.name=='nt':
        with pytest.raises(ValueError,match='Linux-only'):list(b.ordered_task_results([],2))
