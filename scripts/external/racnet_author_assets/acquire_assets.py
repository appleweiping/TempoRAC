"""Finite single-writer acquisition. Original archives/arrays only; no model or generation."""
import argparse,datetime,fcntl,gzip,hashlib,io,json,math,os,shutil,tarfile,time,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from range_client import RangeResponse

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def union(ranges):
    result=[]
    for lo,hi in sorted(ranges):
        if result and lo<=result[-1][1]:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return result
def gaps(ranges,total):
    out=[];at=0
    for lo,hi in union(ranges):
        if at<lo:out.append([at,lo])
        at=max(at,hi)
    if at<total:out.append([at,total])
    return out
def safe(name):
    p=PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name
def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');data=json.dumps(value,indent=2,allow_nan=False).encode()
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

class Engine:
    def __init__(self,control,prepare=False):
        self.control_path=Path(control).resolve();self.c=json.loads(self.control_path.read_text());self.W=Path(self.c['output_root']).resolve();self.R=Path(self.c['server_root']).resolve()
        assert self.W.is_relative_to(self.R) and str(self.R)=='/root/autodl-tmp/TempoRAC'
        for name,expected in self.c['runtime_source_sha256'].items():assert sha(self.W/name)==expected,name
        assert sha(self.c['acquisition_manifest']['path'])==self.c['acquisition_manifest']['sha256']
        assert sha(self.c['amendment']['path'])==self.c['amendment']['sha256']
        assert sha(self.c['input_manifest']['path'])==self.c['input_manifest']['sha256']
        assert sha(self.c['decoded_identities']['path'])==self.c['decoded_identities']['sha256']
        self.objects=self.c['objects'];assert len(self.objects)==61 and sum(x['bytes'] for x in self.objects)==128629792634
        self.byid={x['id']:x for x in self.objects};self.state_path=self.W/'STATE.json';self.deadline=float('inf');self.attempt=None
        self.references={}
        ids=json.loads(Path(self.c['decoded_identities']['path']).read_text())['entries'];decoded={r['key']:r for r in ids}
        rows=json.loads(Path(self.c['input_manifest']['path']).read_text())
        for r in rows:
            key=r['split']+'/'+r['video_id'];d=decoded[key]
            self.references[key]=dict(count=r['count'],full_N=d['full_N'],ROI_N=d['roi_N'],source_video_sha256=r['source_video_sha256'])
        self.native_keys={k for k,r in self.references.items() if k.startswith('rc_train/') and r['count'] is not None}
        assert len(self.native_keys)==755
        self.member_index={}
        for index in sorted((self.W/'indices').glob('members_*.jsonl')):
            lines=index.read_bytes().splitlines(keepends=True)
            for i,line in enumerate(lines):
                if not line.endswith(b'\n'):
                    assert i==len(lines)-1;continue
                item=json.loads(line);self.member_index[item['asset_key']]=item
        if self.state_path.exists():
            self.s=json.loads(self.state_path.read_text());assert self.s['control_sha256']==sha(self.control_path),'control changed'
        else:
            assert prepare,'Run --prepare first'
            self.s=dict(status='prepared_not_launched',control_sha256=sha(self.control_path),started_utc=None,deadline_epoch=None,attempts=0,network_received_bytes=0,objects={x['id']:{'coverage':[],'verified':False} for x in self.objects})
        self.refresh()
    def refresh(self):
        self.logical=sum(p.stat().st_size for root in [self.W,Path(self.c['g1_root'])] for p in root.rglob('*') if p.is_file())
        self.disk=shutil.disk_usage(self.R)
    def resource(self,extra=0,refresh=True):
        if refresh:self.refresh()
        assert self.logical+extra<=self.c['asset_cap_bytes'],'combined asset cap'
        assert shutil.disk_usage(self.R).free-extra>=self.c['volume_free_reserve_bytes'],'volume free reserve'
        if time.time()>=self.deadline:raise TimeoutError('original48h deadline exhausted')
    def disk_write(self,n):
        assert shutil.disk_usage(self.R).free-n>=self.c['volume_free_reserve_bytes']+self.c['metadata_headroom_bytes'],'volume free reserve including failure/journal headroom'
        if time.time()>=self.deadline:raise TimeoutError('48h deadline before write')
    def path(self,obj):
        suffix='' if self.s['objects'][obj['id']]['verified'] else '.partial'
        return self.W/'objects'/(obj['path']+suffix)
    def make_object(self,obj):
        if not self.s['objects'][obj['id']]['verified']:self.reconcile_final(obj)
        p=self.path(obj)
        if not p.exists():
            self.resource(obj['bytes']+self.c['metadata_headroom_bytes']);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.truncate(obj['bytes']);f.flush();os.fsync(f.fileno())
        assert p.stat().st_size==obj['bytes'];return p
    def reconcile_final(self,obj):
        # rename may have completed before the verified state became durable.
        final=self.W/'objects'/obj['path'];st=self.s['objects'][obj['id']]
        if not final.exists():return
        assert final.stat().st_size==obj['bytes'] and sha(final)==obj['sha256'],'final object identity mismatch'
        if not st['verified']:
            st.update(verified=True,coverage=[[0,obj['bytes']]],sha256=obj['sha256']);self.save_state()
    def save_state(self):atomic(self.state_path,self.s)
    def prepare(self):
        if self.state_path.exists():raise RuntimeError('Existing state; prepare is not a reset/resume operation')
        self.refresh();assert self.disk.free>=self.c['asset_cap_bytes']+self.c['volume_free_reserve_bytes'],'fresh startup disk gate'
        g1=Path(self.c['g1_root']);assert sha(g1/'G1_RECEIPT.json')==self.c['g1_receipt_sha256']
        rows=json.loads((g1/'G1_RECEIPT.json').read_text())['requests'];records=[]
        for row in rows:
            assert row['status']=='complete';src=g1/row['body_file'];assert sha(src)==row['sha256'] and src.stat().st_size==row['actual_bytes']
            obj=next(x for x in self.objects if x['path']==row['object']);p=self.make_object(obj);lo=row['requested_range'][0];data=src.read_bytes();hi=lo+len(data)
            assert hi==row['requested_range'][1]+1
            previous=self.s['objects'][obj['id']]['coverage']
            with p.open('r+b') as f:
                for a,b in previous:
                    left,right=max(a,lo),min(b,hi)
                    if left<right:f.seek(left);assert f.read(right-left)==data[left-lo:right-lo],'conflicting reused overlap'
                for a,b in gaps(previous,obj['bytes']):
                    left,right=max(a,lo),min(b,hi)
                    if left<right:self.disk_write(right-left);f.seek(left);f.write(data[left-lo:right-lo])
                f.flush();os.fsync(f.fileno())
            with p.open('rb') as f:f.seek(lo);copied=hashlib.sha256(f.read(len(data))).hexdigest()
            assert copied==row['sha256'],'G1 destination range readback mismatch'
            self.s['objects'][obj['id']]['coverage']=union(previous+[[lo,hi]])
            records.append(dict(object_id=obj['id'],source=str(src),range=[lo,hi],bytes=len(data),sha256=row['sha256'],destination_readback_sha256=copied))
        unique=sum(b-a for item in self.s['objects'].values() for a,b in item['coverage'])
        self.s['reused_g1_unique_bytes']=unique;self.save_state();self.refresh()
        receipt=dict(status='prelaunch_prepared_no_network',g1_payload_bytes=sum(r['bytes'] for r in records),unique_reused_bytes=unique,overlap_bytes=sum(r['bytes'] for r in records)-unique,missing_network_bytes=128629792634-unique,records=records,
            object_coverage={k:v['coverage'] for k,v in self.s['objects'].items() if v['coverage']},disk=dict(total=self.disk.total,free=self.disk.free,logical_combined_assets=self.logical,asset_cap=self.c['asset_cap_bytes'],free_reserve=self.c['volume_free_reserve_bytes'],fresh_full_budget_gate_pass=self.disk.free>=self.c['asset_cap_bytes']+self.c['volume_free_reserve_bytes']),control_sha256=sha(self.control_path))
        atomic(self.W/'PRELAUNCH_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
    def replay_journals(self):
        for obj in self.objects:
            if not self.s['objects'][obj['id']]['verified']:self.reconcile_final(obj)
            st=self.s['objects'][obj['id']]
            if st['verified']:assert sha(self.path(obj))==obj['sha256'],'verified object modified'
        for journal in sorted((self.W/'journals').glob('segments_*.jsonl')):
            lines=journal.read_bytes().splitlines(keepends=True)
            for i,line in enumerate(lines):
                if not line.endswith(b'\n'):
                    assert i==len(lines)-1;continue # preserved incomplete last log line; new attempt gets a new journal
                r=json.loads(line);obj=self.byid[r['object_id']];st=self.s['objects'][obj['id']]
                if st['verified']:continue
                with self.path(obj).open('rb') as f:f.seek(r['start']);data=f.read(r['bytes'])
                assert hashlib.sha256(data).hexdigest()==r['sha256'],'durable segment mismatch'
                st['coverage']=union(st['coverage']+[[r['start'],r['start']+r['bytes']]])
        self.save_state()
        self.s['network_durable_bytes']=sum(b-a for item in self.s['objects'].values() for a,b in item['coverage'])-self.s['reused_g1_unique_bytes']
        self.s['network_received_bytes']=sum(json.loads(line)['received_bytes'] for journal in (self.W/'journals').glob('requests_*.jsonl') for line in journal.read_bytes().splitlines(keepends=True) if line.endswith(b'\n'))
        # Old receipts used received_bytes to mean journalled coverage.
        self.s['network_received_bytes']=max(self.s['network_received_bytes'],self.s['network_durable_bytes'])
        self.save_state()
    def commit(self,obj,at,data,reserved=False):
        if not reserved:self.disk_write(len(data))
        p=self.path(obj)
        with p.open('r+b') as f:f.seek(at);f.write(data);f.flush();os.fsync(f.fileno())
        append(self.W/'journals'/f'segments_{self.attempt:03d}.jsonl',dict(object_id=obj['id'],start=at,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        st=self.s['objects'][obj['id']];st['coverage']=union(st['coverage']+[[at,at+len(data)]])
    def request(self,obj,lo,hi):
        self.resource(self.c['metadata_headroom_bytes']);assert any(a<=lo and hi<b for a,b in gaps(self.s['objects'][obj['id']]['coverage'],obj['bytes'])),'request overlaps known bytes'
        row=dict(object_id=obj['id'],range=[lo,hi],attempted_bytes=hi-lo+1,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),received_bytes=0,durable_bytes=0);buffer=bytearray();h=hashlib.sha256();response=None;failure=None;checkpoint_failure=None;commit_failed=False;reserved_until=lo
        def header(record):append(self.W/'journals'/f'private_http_{self.attempt:03d}.jsonl',dict(object_id=obj['id'],range=[lo,hi],**record))
        def flush():
            nonlocal commit_failed,checkpoint_failure
            try:self.commit(obj,lo+row['received_bytes']-len(buffer),buffer,reserved=True)
            except BaseException as e:commit_failed=True;checkpoint_failure=e;raise
            row['durable_bytes']+=len(buffer);buffer.clear()
        try:
            response=RangeResponse(obj['transport_url'],lo,hi,obj['bytes'],self.deadline,self.c,header)
            while response.received<response.expected:
                at=lo+row['received_bytes']
                if at==reserved_until:
                    n=min(self.c['checkpoint_bytes'],response.expected-response.received);self.disk_write(n)
                    # Allocate physical capacity before receiving this checkpoint.
                    # A terminal flush never repeats reserve/deadline checks.
                    with self.path(obj).open('r+b') as f:os.posix_fallocate(f.fileno(),at,n)
                    reserved_until=at+n
                self.disk_write(0)
                block=response.read(min(65536,reserved_until-at));buffer.extend(block);h.update(block);row['received_bytes']+=len(block)
                if lo+row['received_bytes']==reserved_until:flush()
        except BaseException as e:failure=e
        finally:
            if response:
                try:response.close()
                except BaseException as e:
                    if failure is None:failure=e
            if buffer and not commit_failed:
                try:flush()
                except BaseException as e:
                    if failure is None:failure=e
            row.update(status='complete' if failure is None else 'failed',sha256=h.hexdigest(),error=None if failure is None else f'{type(failure).__name__}: {failure}',checkpoint_error=None if checkpoint_failure is None else f'{type(checkpoint_failure).__name__}: {checkpoint_failure}')
            self.s['network_received_bytes']=self.s.get('network_received_bytes',0)+row['received_bytes']
            self.s['network_durable_bytes']=sum(b-a for item in self.s['objects'].values() for a,b in item['coverage'])-self.s['reused_g1_unique_bytes']
            persistence_errors=[]
            # Receipt is independent of state persistence and checkpoint success.
            for persist in [lambda:append(self.W/'journals'/f'requests_{self.attempt:03d}.jsonl',row),self.save_state]:
                try:persist()
                except BaseException as e:
                    persistence_errors.append(f'{type(e).__name__}: {e}')
                    if failure is None:failure=e
            if persistence_errors:print(json.dumps(dict(event='request_receipt_persistence_failure',receipt=row,persistence_errors=persistence_errors)),flush=True)
        if failure:raise failure
        assert row['received_bytes']==hi-lo+1
    def verify_object(self,obj):
        st=self.s['objects'][obj['id']]
        if st['verified']:return
        assert not gaps(st['coverage'],obj['bytes']);p=self.path(obj);observed=sha(p)
        assert observed==obj['sha256'],f'LFS SHA mismatch for {obj["id"]}: {observed}'
        dest=self.W/'objects'/obj['path'];assert not dest.exists();p.rename(dest);st.update(verified=True,sha256=observed);self.save_state()
        self.refresh();elapsed=time.time()-self.started_epoch;done=sum(o['bytes'] for o in self.objects if self.s['objects'][o['id']]['verified'])
        durable=self.s.get('network_durable_bytes',self.s['network_received_bytes'])
        atomic(self.W/'PROGRESS.json',dict(status='acquiring',verified_objects=sum(x['verified'] for x in self.s['objects'].values()),verified_bytes=done,new_network_bytes=self.s['network_received_bytes'],network_durable_bytes=durable,elapsed_seconds=elapsed,ETA_seconds=None if not durable else (128629792634-self.s['reused_g1_unique_bytes']-durable)*elapsed/durable,disk_free=self.disk.free,logical_asset_bytes=self.logical))
        print(json.dumps({'event':'object_verified','object_id':obj['id'],'sha256':observed,'verified_bytes':done}),flush=True)
    def ensure(self,obj,at):
        self.make_object(obj);st=self.s['objects'][obj['id']]
        for lo,hi in st['coverage']:
            if lo<=at<hi:return hi
        gap=next((a,b) for a,b in gaps(st['coverage'],obj['bytes']) if a<=at<b)
        # Read-frontier calls only at a missing boundary; no unverified hole is consumed.
        assert at==gap[0];hi=min(gap[1],at+self.c['request_chunk_bytes'])
        self.request(obj,at,hi-1)
        if not gaps(st['coverage'],obj['bytes']):self.verify_object(obj)
        return hi
    def acquire_complete(self,obj):
        self.make_object(obj)
        while gaps(self.s['objects'][obj['id']]['coverage'],obj['bytes']):
            lo,hi=gaps(self.s['objects'][obj['id']]['coverage'],obj['bytes'])[0];self.ensure(obj,lo)
        self.verify_object(obj)
    def extract(self,asset_key,name,size,source,kind):
        assert safe(name) and size<=self.c['max_member_bytes'],'unsafe/oversize archive member'
        dest=self.W/'extracted'/kind/name;assert dest.resolve().is_relative_to(self.W/'extracted');dest.parent.mkdir(parents=True,exist_ok=True)
        existing=self.member_index.get(asset_key)
        orphan=not existing and dest.exists()
        if orphan:
            assert dest.stat().st_size==size;existing=dict(asset_key=asset_key,path=str(dest),archive_name=name,kind=kind,bytes=size,sha256=sha(dest))
        if existing:
            assert dest.exists() and sha(dest)==existing['sha256'],'indexed member changed'
            h=hashlib.sha256();n=0
            while n<size:
                self.disk_write(0);data=source.read(min(1024*1024,size-n));assert data,'member replay EOF';h.update(data);n+=len(data)
            assert h.hexdigest()==existing['sha256']
            if orphan:
                if name.endswith('.npz'):
                    from schema_audit import inspect_npz
                    existing.update(inspect_npz(dest,self,kind,name))
                self.index_member(existing)
            if existing.get('asset_errors'):raise RuntimeError('Previously indexed asset schema failure: '+str(existing['asset_errors']))
            return existing
        part=dest.with_name(dest.name+'.partial');progress_path=part.with_name(part.name+'.progress.json');received=0
        if part.exists():
            progress=json.loads(progress_path.read_text());assert progress['asset_key']==asset_key and part.stat().st_size==size;received=progress['received_bytes']
        else:
            self.resource(size+self.c['metadata_headroom_bytes'])
            atomic(progress_path,dict(asset_key=asset_key,received_bytes=0,declared_bytes=size))
            with part.open('xb') as f:f.truncate(size);f.flush();os.fsync(f.fileno())
        h=hashlib.sha256();at=0
        with part.open('r+b') as out:
            while at<size:
                self.disk_write(min(1024*1024,size-at));data=source.read(min(1024*1024,size-at));assert data,'member EOF';h.update(data)
                if at<received:
                    assert at+len(data)<=received or received==size,'unaligned replay checkpoint';out.seek(at);assert out.read(len(data))==data,'partial member replay mismatch'
                else:
                    out.seek(at);out.write(data);out.flush();os.fsync(out.fileno());atomic(progress_path,dict(asset_key=asset_key,received_bytes=at+len(data),declared_bytes=size))
                at+=len(data)
        record=dict(asset_key=asset_key,path=str(dest),archive_name=name,kind=kind,bytes=size,sha256=h.hexdigest())
        if name.endswith('.npz'):
            from schema_audit import inspect_npz
            try:record.update(inspect_npz(part,self,kind,name))
            except BaseException as error:record.update(asset_errors=[f'Safe NPZ parse failure: {type(error).__name__}: {error}'],schema_parse_complete=False)
        assert not dest.exists();part.rename(dest);self.index_member(record)
        if kind=='RC' and name.endswith('.npz') and not (self.W/'FIRST_COMPLETE_RC_SCHEMA.json').exists():atomic(self.W/'FIRST_COMPLETE_RC_SCHEMA.json',record)
        if record.get('asset_errors'):raise RuntimeError('Author asset schema incompatible: '+str(record['asset_errors']))
        return record
    def index_member(self,record):
        folder=self.W/'indices';folder.mkdir(exist_ok=True)
        append(folder/f'members_{self.attempt:03d}.jsonl',record);self.member_index[record['asset_key']]=record
        if record['kind']=='RC' and record['archive_name'].endswith('.npz') and not (self.W/'FIRST_COMPLETE_RC_SCHEMA.json').exists():atomic(self.W/'FIRST_COMPLETE_RC_SCHEMA.json',record)
    def ucf(self):
        if self.s.get('ucf_extracted'):return
        obj=next(o for o in self.objects if o['kind']=='UCF');self.acquire_complete(obj);seen=set();unpacked=0
        with zipfile.ZipFile(self.path(obj)) as z:
            for info in z.infolist():
                assert safe(info.filename),'unsafe ZIP path';mode=(info.external_attr>>16)&0o170000;assert mode in [0,0o100000,0o040000],'ZIP link/device'
                if info.is_dir():continue
                unpacked+=info.file_size;assert unpacked<=self.c['max_ucf_uncompressed_bytes'],'UCF extraction cap'
                with z.open(info) as f:r=self.extract('UCF:'+info.filename,info.filename,info.file_size,f,'UCF')
                if info.filename.endswith('.npz'):
                    key=r['qualified_key'];assert key not in seen,'duplicate UCF mapping';seen.add(key)
        assert seen=={k for k in self.references if k.startswith('ucf_test/')} and len(seen)==105,'UCF105 membership'
        self.s['ucf_extracted']=True;self.save_state()
    def run(self,resume,review,review_sha):
        assert Path(review).is_file() and sha(review)==review_sha,'root review receipt mismatch'
        if self.s['status']=='complete_assets_not_training_ready':raise RuntimeError('Asset task already complete; no repeat acquisition')
        if self.s['started_utc'] is None:
            assert not resume;self.refresh();assert self.disk.free>=self.c['asset_cap_bytes']+self.c['volume_free_reserve_bytes'],'fresh launch disk gate'
            self.started_epoch=time.time();self.s.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),started_epoch=self.started_epoch,deadline_epoch=self.started_epoch+self.c['wall_seconds'])
        else:
            assert resume,'Existing attempt requires explicit --resume';self.started_epoch=self.s['started_epoch']
        self.deadline=self.s['deadline_epoch'];self.resource();self.s['attempts']+=1;self.attempt=self.s['attempts'];(self.W/'journals').mkdir(exist_ok=True)
        self.s.update(status='running',review_sha256=review_sha,pid=os.getpid());self.save_state()
        try:
            self.replay_journals()
            raw=RCReader(self)
            with gzip.GzipFile(fileobj=raw,mode='rb') as gz:
                counted=Counted(gz,self)
                with tarfile.open(fileobj=counted,mode='r|') as archive:
                    for member in archive:
                        assert safe(member.name),'unsafe TAR path'
                        if member.isdir():continue
                        assert member.isreg(),'TAR link/device rejected'
                        source=archive.extractfile(member);assert source is not None
                        with source:self.extract('RC:'+str(member.offset)+':'+member.name,member.name,member.size,source,'RC')
                while counted.read(1024*1024):pass # validate gzip footer and remaining compressed bytes without a joined archive
            self.ucf()
            for obj in self.objects:self.acquire_complete(obj)
            assert all(x['verified'] for x in self.s['objects'].values())
            required={k for k in self.references if k.startswith(('rc_valid/','rc_test/'))}|self.native_keys
            found={r.get('qualified_key') for r in self.member_index.values() if r.get('kind')=='RC'}
            assert required<=found,'Missing native755/validation/test assets'
            self.s['status']='complete_assets_not_training_ready';self.save_state();self.resource();atomic(self.W/'COMPLETION.json',dict(status=self.s['status'],objects_verified=61,archive_bytes=128629792634,network_bytes=self.s['network_received_bytes'],reused_unique_bytes=self.s['reused_g1_unique_bytes'],members=len(self.member_index),future_native_labeled_train=755,evaluation_unchanged=357,training_released=False,control_sha256=self.s['control_sha256']))
        except BaseException as e:
            self.s['status']='stopped_on_failure';self.save_state();atomic(self.W/f'FAILURE_{self.attempt:03d}.json',dict(status='failed_no_retry',error_type=type(e).__name__,error=str(e),control_sha256=self.s['control_sha256'],deadline_epoch=self.deadline,network_bytes=self.s['network_received_bytes'],objects=self.s['objects']));raise

class RCReader(io.RawIOBase):
    def __init__(self,engine):self.e=engine;self.parts=[o for o in engine.objects if o['kind']=='RC'];self.index=0;self.at=0
    def readable(self):return True
    def read(self,n=-1):
        if n==0:return b''
        if self.index>=len(self.parts):return b''
        obj=self.parts[self.index]
        if self.at==obj['bytes']:
            self.e.verify_object(obj);self.index+=1;self.at=0
            if self.index==1:self.e.ucf()
            return self.read(n)
        end=self.e.ensure(obj,self.at);take=min(1024*1024 if n<0 else n,end-self.at,obj['bytes']-self.at)
        with self.e.path(obj).open('rb') as f:f.seek(self.at);data=f.read(take)
        assert len(data)==take;self.at+=len(data);return data
class Counted:
    def __init__(self,source,engine):self.source=source;self.e=engine;self.n=0
    def read(self,n=-1):
        self.e.disk_write(0);data=self.source.read(n);self.n+=len(data);assert self.n<=self.e.c['max_rc_uncompressed_bytes'],'RC decompression cap';return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--review-file');p.add_argument('--review-sha256');a=p.parse_args();assert a.prepare!=a.run
    root=Path(a.control).resolve().parent;root.mkdir(parents=True,exist_ok=True)
    with (root/'WRITER.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            e=Engine(a.control,prepare=a.prepare)
            if a.prepare:e.prepare()
            else:e.run(a.resume,a.review_file,a.review_sha256)
        except BaseException as error:
            atomic(root/f'PROCESS_FAILURE_{time.time_ns()}.json',dict(error_type=type(error).__name__,error=str(error),no_retry=True));raise
if __name__=='__main__':main()
