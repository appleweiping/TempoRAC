import hashlib,json,pathlib
p=pathlib.Path(__file__).resolve().parent
value=json.loads((p/'PACKET_BINDINGS.json').read_text(encoding='utf-8-sig'))
rows=[]
for r in value['files']:
    q=p/r['name']; b=q.read_bytes()
    assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],r['name']
    rows.append(r)
(p/'TRANSFER_VERIFIED.json').write_text(json.dumps(dict(status='PASS',files=len(rows),scope='E-origin packet source and metadata bytes/SHA; no experimental payload'),indent=2)+'\n')
print('TRANSFER_PASS',len(rows))
