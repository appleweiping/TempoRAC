import os,json,hashlib,urllib.request,tarfile,pathlib,datetime
R=pathlib.Path('/root/autodl-tmp/TempoRAC')
B=R/'external-baselines-20261112/detrc-prerequisites'
(B/'assets').mkdir(parents=True,exist_ok=True)
def fetch(url,name):
 p=B/'assets'/name
 if not p.exists():
  with urllib.request.urlopen(url,timeout=90) as f,open(str(p)+'.part','wb') as o:
   import shutil
   shutil.copyfileobj(f,o)
  os.replace(str(p)+'.part',p)
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return {'url':url,'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
url='https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth'
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'backbone':fetch(url,url.rsplit('/',1)[1])}
(B/'ASSET_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
receipt['mmaction']=fetch('https://codeload.github.com/open-mmlab/mmaction2/tar.gz/refs/tags/v0.24.1','mmaction2-v0.24.1.tar.gz')
with tarfile.open(receipt['mmaction']['path']) as t:t.extractall(B/'vendor',filter='data')
for name,version in [('cuda-nvcc','12.4.131'),('cuda-cudart-dev','12.4.127'),('cuda-cccl','12.4.127')]:
 meta=json.load(urllib.request.urlopen('https://api.anaconda.org/package/nvidia/'+name,timeout=90))
 found=[x for x in meta['files'] if x['version']==version and x['basename'].startswith('linux-64/')]
 if not found:raise RuntimeError((name,version))
 x=found[-1]
 rec=fetch('https:'+x['download_url'] if x['download_url'].startswith('//') else x['download_url'],x['basename'].split('/')[-1])
 receipt[name]=rec
 if x.get('sha256'):assert x['sha256']==rec['sha256'],(name,x['sha256'])
 with tarfile.open(rec['path']) as t:t.extractall(B/'cuda-12.4',filter='data')
 (B/'ASSET_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
 print(name,rec,flush=True)
(B/'ASSET_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
