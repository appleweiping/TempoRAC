"""Factored from probe_g1.py: same IPv4/raw-header/exact206 protocol, streaming sink."""
import email.parser,socket,ssl,time,urllib.parse

class RangeResponse:
    def __init__(self,url,lo,hi,total,deadline,limits,on_header):
        self.sock=None;self.expected=hi-lo+1;self.received=0;self.started=time.monotonic();self.deadline=deadline;self.limits=limits
        try:
            for hop in range(6):
                self.guard();u=urllib.parse.urlsplit(url)
                assert u.scheme=='https' and u.hostname and not u.username and u.port in [None,443]
                assert u.hostname in limits['allowed_https_hosts'],'Unapproved redirect/transport host'
                address=socket.getaddrinfo(u.hostname,443,socket.AF_INET,socket.SOCK_STREAM)[0][4]
                raw=socket.socket(socket.AF_INET,socket.SOCK_STREAM);raw.settimeout(min(self.remaining(),limits['connect_timeout_seconds']));raw.connect(address)
                self.sock=ssl.create_default_context().wrap_socket(raw,server_hostname=u.hostname)
                self.sock.settimeout(min(self.remaining(),limits['idle_timeout_seconds']))
                target=urllib.parse.urlunsplit(('', '',u.path or '/',u.query,''))
                request=(f'GET {target} HTTP/1.1\r\nHost: {u.hostname}\r\nRange: bytes={lo}-{hi}\r\nAccept-Encoding: identity\r\nUser-Agent: TempoRAC-fixed-author-assets/1.0\r\nConnection: close\r\n\r\n').encode('ascii')
                self.sock.sendall(request);header=bytearray()
                while not header.endswith(b'\r\n\r\n'):
                    self.guard();self.sock.settimeout(min(self.remaining(),limits['idle_timeout_seconds']))
                    byte=self.sock.recv(1);assert byte,'EOF before HTTP headers';header+=byte;assert len(header)<=131072,'header cap'
                status=int(header.split(b' ',2)[1]);headers=email.parser.BytesParser().parsebytes(bytes(header).split(b'\r\n',1)[1])
                on_header(dict(hop=hop,url=url,status=status,raw_headers=header.decode('iso8859-1'),body_consumed=0))
                if status in [301,302,303,307,308]:
                    location=headers.get('Location');assert location,'redirect without Location';self.close();url=urllib.parse.urljoin(url,location);continue
                assert status==206,f'HTTP{status}; body closed unread'
                assert headers.get('Content-Range')==f'bytes {lo}-{hi}/{total}','Content-Range mismatch'
                assert int(headers.get('Content-Length','-1'))==self.expected,'Content-Length mismatch'
                assert not headers.get('Transfer-Encoding') and headers.get('Content-Encoding','identity')=='identity','unexpected encoding'
                self.final_url=url;return
            raise RuntimeError('redirect limit')
        except BaseException:
            self.close();raise
    def remaining(self):
        return min(self.limits['request_timeout_seconds']-(time.monotonic()-self.started),self.deadline-time.time())
    def guard(self):
        if self.remaining()<=0:raise TimeoutError('finite request/48h deadline reached')
    def read(self,n):
        self.guard();self.sock.settimeout(min(self.remaining(),self.limits['idle_timeout_seconds']))
        data=self.sock.recv(min(n,self.expected-self.received));assert data,'premature response EOF'
        self.received+=len(data);return data
    def close(self):
        if self.sock:self.sock.close();self.sock=None
