"""Narrow R1/R2/R3 server CPU fault injections. No external socket/model/data."""
import copy,hashlib,io,json,os,tempfile,time
from pathlib import Path
from unittest.mock import patch
from acquire_assets import Engine,RCReader,atomic,append,sha,gaps
from range_client import RangeResponse
ROOT=Path('/root/autodl-tmp/TempoRAC/baselines/racnet-author-assets20261112')
RESULTS=[]
def obj(oid,path,data,kind='RC'):
    return dict(id=oid,path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),transport_url='https://fixture.invalid/fixed',kind=kind)
def engine(p,objects):
    (p/'objects').mkdir();(p/'journals').mkdir()
    e=object.__new__(Engine);e.W=p;e.R=p;e.attempt=1;e.deadline=float('inf');e.started_epoch=time.time();e.state_path=p/'STATE.json'
    e.objects=objects;e.byid={o['id']:o for o in objects};e.member_index={}
    e.c=dict(asset_cap_bytes=1024**3,volume_free_reserve_bytes=1024,metadata_headroom_bytes=1024,g1_root=str(p/'empty'),checkpoint_bytes=4,max_member_bytes=1024**2,connect_timeout_seconds=1,idle_timeout_seconds=1,request_timeout_seconds=5,allowed_https_hosts=['fixture.invalid'])
    e.s=dict(objects={o['id']:dict(coverage=[],verified=False) for o in objects},reused_g1_unique_bytes=0,network_received_bytes=0,deadline_epoch=123456,attempts=1)
    for o in objects:
        target=e.path(o);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bytes(o['bytes']))
    e.save_state();return e

def r1(p,journal):
    data=b'final-complete';o=obj('one','one.bin',data);e=engine(p,[o]);e.path(o).write_bytes(data);e.s['objects']['one']['coverage']=[[0,len(data)]];e.save_state()
    if journal:append(p/'journals/segments_001.jsonl',dict(object_id='one',start=0,bytes=len(data),sha256=o['sha256']))
    def stop():raise OSError('injected state failure after rename')
    with patch.object(e,'save_state',side_effect=stop):
        try:e.verify_object(o)
        except OSError:pass
        else:raise AssertionError('missing injected rename/state failure')
    assert (p/'objects/one.bin').exists() and not (p/'objects/one.bin.partial').exists()
    e.s=json.loads(e.state_path.read_text());e.replay_journals();e.make_object(o)
    assert e.s['objects']['one']['verified'] and e.s['objects']['one']['coverage']==[[0,len(data)]]
    assert e.s['deadline_epoch']==123456 and not (p/'objects/one.bin.partial').exists()
    RESULTS.append(dict(case='R1_final_adoption_'+('with_journal' if journal else 'without_journal'),status='PASS',durable_bytes=len(data),network_requests=0,deadline_epoch=e.s['deadline_epoch']))

def wrong_final(p):
    o=obj('one','one.bin',b'correct');e=engine(p,[o]);(p/'objects/one.bin').write_bytes(b'incorrect')
    try:e.replay_journals()
    except AssertionError as error:assert 'identity' in str(error)
    else:raise AssertionError('wrong final accepted')
    assert not e.s['objects']['one']['verified']
    RESULTS.append(dict(case='R1_wrong_final_identity',status='PASS'))

class Sock:
    def __init__(self,after):
        self.header=b'HTTP/1.1 206 Partial\r\nContent-Range: bytes 0-7/8\r\nContent-Length: 8\r\n\r\n';self.body=b'ABCDEFGH';self.h=0;self.b=0;self.after=after;self.closed=False;self.body_calls=0
    def settimeout(self,n):pass
    def connect(self,a):pass
    def sendall(self,b):pass
    def recv(self,n):
        if self.h<len(self.header):r=self.header[self.h:self.h+n];self.h+=len(r);return r
        self.body_calls+=1;r=self.body[self.b:self.b+min(n,2)];self.b+=len(r);self.after(self);return r
    def close(self):self.closed=True
class TLS:
    def wrap_socket(self,s,server_hostname):return s

def r2(p,mode):
    data=b'ABCDEFGH';o=obj('one','one.bin',data);e=engine(p,[o]);clock=[0];free=[10**9]
    def after(s):
        if mode in ['deadline','deadline_checkpoint_failure']:clock[0]=10
        if mode=='reserve':free[0]=2047
    sock=Sock(after);allocation=[];real_allocate=os.posix_fallocate;real_append=append;commits=[]
    def allocate(fd,at,n):
        assert sock.b==at;real_allocate(fd,at,n);allocation.append([at,n])
    def journal(path,row):
        if Path(path).name.startswith('segments_'):
            commits.append(dict(row))
            if mode in ['checkpoint_failure','deadline_checkpoint_failure']:raise OSError('injected segment journal failure')
        return real_append(path,row)
    class Disk:
        @property
        def free(self):return free[0]
    with patch('range_client.socket.getaddrinfo',return_value=[(None,None,None,None,('127.0.0.1',443))]),patch('range_client.socket.socket',return_value=sock),patch('range_client.ssl.create_default_context',return_value=TLS()),patch('range_client.time.monotonic',side_effect=lambda:clock[0]),patch('acquire_assets.shutil.disk_usage',return_value=Disk()),patch('acquire_assets.os.posix_fallocate',side_effect=allocate),patch('acquire_assets.append',side_effect=journal):
        try:e.request(o,0,7)
        except (TimeoutError,AssertionError,OSError) as error:failure=str(error);failure_type=type(error).__name__
        else:raise AssertionError('expected injected terminal stop')
    rows=[json.loads(x) for x in (p/'journals/requests_001.jsonl').read_text().splitlines()];assert len(rows)==1;row=rows[0]
    expected_received=4 if mode=='checkpoint_failure' else 2
    expected_durable=0 if 'checkpoint_failure' in mode else 2
    assert row['attempted_bytes']==8 and row['received_bytes']==expected_received and row['durable_bytes']==expected_durable and row['status']=='failed'
    assert e.s['network_received_bytes']==expected_received and e.s['network_durable_bytes']==expected_durable
    assert sock.b==expected_received and sock.closed and len(commits)==1 and allocation==[[0,4]]
    assert e.s['deadline_epoch']==123456
    if mode.startswith('deadline'):assert failure_type=='TimeoutError' and 'finite request' in row['error']
    if mode=='reserve':assert 'reserve' in row['error']
    if expected_durable:assert e.path(o).read_bytes()[:2]==b'AB' and gaps(e.s['objects']['one']['coverage'],8)==[[2,8]]
    if 'checkpoint_failure' in mode:assert 'segment journal failure' in row['checkpoint_error']
    # Actual restart replay recovers exactly the journalled bytes and keeps accounting.
    e.s=json.loads(e.state_path.read_text());e.replay_journals();assert e.s['network_received_bytes']==expected_received and e.s['network_durable_bytes']==expected_durable
    RESULTS.append(dict(case='R2_'+mode,status='PASS',receipt=row,body_read_calls=sock.body_calls,physical_reservations=allocation,commit_attempts=len(commits),raised=failure))

def r3(p):
    a=b'ABC';b=b'DEF';o0=obj('rc00','rc00',a);o1=obj('rc01','rc01',b);e=engine(p,[o0,o1])
    for o,data in [(o0,a),(o1,b)]:e.path(o).write_bytes(data);e.s['objects'][o['id']]['coverage']=[[0,len(data)]]
    # Native RCReader invokes UCF exactly between rc00 and rc01 while RC extract is active.
    order=[]
    def ucf():
        order.append('UCF')
        class Failing:
            n=0
            def read(self,n):
                self.n+=1
                if self.n==1:return b'UV'
                raise OSError('injected nested UCF transfer stop')
        e.extract('UCF:active','active.txt',4,Failing(),'UCF')
    e.ucf=ucf
    try:e.extract('RC:active','active.txt',6,RCReader(e),'RC')
    except OSError as error:assert 'nested UCF' in str(error)
    else:raise AssertionError('expected nested failure')
    rc_progress=p/'extracted/RC/active.txt.partial.progress.json';ucf_progress=p/'extracted/UCF/active.txt.partial.progress.json'
    assert json.loads(rc_progress.read_text())['received_bytes']==3 and json.loads(ucf_progress.read_text())['received_bytes']==2
    class SameChunkSource(io.BytesIO):
        def read(self,n=-1):return super().read(min(2,n))
    e.extract('UCF:active','active.txt',4,SameChunkSource(b'UVWX'),'UCF')
    e.s['ucf_extracted']=True;e.ucf=lambda:order.append('UCF_replay_already_extracted')
    result=e.extract('RC:active','active.txt',6,RCReader(e),'RC')
    assert Path(result['path']).read_bytes()==b'ABCDEF' and (p/'extracted/UCF/active.txt').read_bytes()==b'UVWX'
    assert not list((p/'journals').glob('requests_*'))
    RESULTS.append(dict(case='R3_active_RC_nested_UCF_failure_and_replay',status='PASS',member_progress_files_distinct=True,RC_durable_prefix=3,UCF_durable_prefix=2,order=order,network_requests=0))

def main():
    # Block accidental external network in any fixture not deliberately mocked.
    with tempfile.TemporaryDirectory(prefix='recovery-r1-r3-',dir=ROOT) as tmp:
        p=Path(tmp)
        cases=[('r1_journal',lambda p:r1(p,True)),('r1_no_journal',lambda p:r1(p,False)),('wrong_final',wrong_final)]+[(mode,lambda p,mode=mode:r2(p,mode)) for mode in ['deadline','reserve','checkpoint_failure','deadline_checkpoint_failure']]+[('nested',r3)]
        for name,fn in cases:
            d=p/name;d.mkdir();fn(d)
    receipt=dict(status='PASS',scope='Targeted actual Engine/RCReader/RangeResponse CPU fault injections with mocked socket; no network downloads/model/arrays/training',cases=RESULTS)
    atomic(ROOT/'CPU_RECOVERY_R1_R3_FIXTURE.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
