"""Server-only transport/resource bookkeeping fixture; no new arrays, model or network."""
import hashlib,io,json,tempfile,time
from pathlib import Path
from unittest.mock import patch
from acquire_assets import Engine,union,gaps,atomic,append
from range_client import RangeResponse

class SocketFixture:
    def __init__(self,data):self.data=data;self.at=0;self.closed=False
    def settimeout(self,value):pass
    def connect(self,address):pass
    def sendall(self,data):pass
    def recv(self,n):data=self.data[self.at:self.at+n];self.at+=len(data);return data
    def close(self):self.closed=True

def transport_case(status,range_value,length,expect_ok):
    header=f'HTTP/1.1 {status} Test\r\nContent-Range: {range_value}\r\nContent-Length: {length}\r\n\r\n'.encode()
    sock=SocketFixture(header+b'DATA');limits=dict(connect_timeout_seconds=1,idle_timeout_seconds=1,request_timeout_seconds=5,allowed_https_hosts=['fixture.invalid'])
    class TLS:
        def wrap_socket(self,plain,server_hostname):return plain
    with patch('range_client.socket.getaddrinfo',return_value=[(None,None,None,None,('127.0.0.1',443))]),patch('range_client.socket.socket',return_value=sock),patch('range_client.ssl.create_default_context',return_value=TLS()):
        if expect_ok:
            response=RangeResponse('https://fixture.invalid/fixed-revision',0,3,4,time.time()+10,limits,lambda value:None)
            assert response.read(4)==b'DATA';response.close()
        else:
            try:RangeResponse('https://fixture.invalid/fixed-revision',0,3,4,time.time()+10,limits,lambda value:None)
            except AssertionError:pass
            else:raise AssertionError('Expected header rejection')
            assert sock.at==len(header),'Rejected response body was consumed'
        assert sock.closed

def main():
    root=Path('/root/autodl-tmp/TempoRAC/baselines/racnet-author-assets20261112')
    with tempfile.TemporaryDirectory(prefix='transport-fixture-',dir=root) as tmp:
        p=Path(tmp);original=b'0123456789abcdefghijklmnop'
        # Overlapping seed intervals must be unioned; only two actual holes remain.
        coverage=union([[0,5],[3,8],[12,len(original)]])
        assert coverage==[[0,8],[12,26]] and gaps(coverage,26)==[[8,12]]
        (p/'objects').mkdir();(p/'journals').mkdir();target=p/'objects/fixture.bin.partial';target.write_bytes(bytes(26))
        with target.open('r+b') as f:
            for lo,hi in coverage:f.seek(lo);f.write(original[lo:hi])
        # Simulated network fault after2received bytes: durable segment journal precedes stale state.
        segment=original[8:10]
        obj=dict(id='fixture',path='fixture.bin',bytes=26,sha256=hashlib.sha256(original).hexdigest())
        writer=object.__new__(Engine);writer.W=p;writer.R=root;writer.state_path=p/'STATE.json';writer.attempt=1;writer.deadline=float('inf')
        writer.objects=[obj];writer.byid={'fixture':obj};writer.c=dict(volume_free_reserve_bytes=1,asset_cap_bytes=16*1024**2,g1_root=str(p/'empty'),metadata_headroom_bytes=1024,max_member_bytes=1024**2)
        writer.s=dict(objects={'fixture':dict(coverage=coverage,verified=False)},reused_g1_unique_bytes=22,deadline_epoch=123456,network_received_bytes=0)
        writer.save_state();writer.commit(obj,8,segment)
        writer.s=json.loads(writer.state_path.read_text()) # stale state, durable segment must be replayed by actual implementation
        writer.replay_journals();restored=writer.s['objects']['fixture']['coverage']
        assert gaps(restored,26)==[[10,12]] and writer.s['deadline_epoch']==123456 and writer.s['network_received_bytes']==2
        with target.open('r+b') as f:f.seek(10);f.write(original[10:12])
        assert target.read_bytes()==original
        writer.member_index={};record=writer.extract('fixture-text','fixture.txt',26,io.BytesIO(original),'RC')
        assert Path(record['path']).read_bytes()==original
        assert writer.extract('fixture-text','fixture.txt',26,io.BytesIO(original),'RC')['sha256']==record['sha256']
        try:writer.extract('bad','../escape.txt',1,io.BytesIO(b'x'),'RC')
        except AssertionError:pass
        else:raise AssertionError('Traversal accepted')
    transport_case(200,'bytes 0-3/4',4,False);transport_case(206,'bytes 1-4/5',4,False);transport_case(206,'bytes 0-3/4',5,False);transport_case(206,'bytes 0-3/4',4,True)
    dummy=object.__new__(Engine);dummy.R=root;dummy.logical=20;dummy.deadline=float('inf');dummy.c={'asset_cap_bytes':100,'volume_free_reserve_bytes':50};dummy.refresh=lambda:None
    class Disk:free=70
    with patch('acquire_assets.shutil.disk_usage',return_value=Disk()):
        dummy.resource(20,refresh=False)
        for extra in [21,81]:
            try:dummy.resource(extra,refresh=False)
            except AssertionError:pass
            else:raise AssertionError('Expected reserve/cap rejection')
    receipt=dict(status='PASS',scope='Actual segment replay/extraction and mocked HTTP/resource fixture; no real network/features/model',overlap_union_exact=True,partial_failure_resume_only_missing=True,deadline_not_reset=True,safe_member_replay_no_duplicate=True,traversal_rejected=True,HTTP200_closed_unread=True,wrong_range_length_closed_unread=True,exact206_stream_accepted=True,free_reserve_and_asset_cap_enforced=True)
    atomic(root/'CPU_TRANSPORT_FIXTURE.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
