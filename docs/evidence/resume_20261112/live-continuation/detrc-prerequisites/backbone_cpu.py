import pathlib,sys,os,json,hashlib,datetime
B=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites')
sys.path.insert(0,str(B/'vendor/mmaction2-0.24.1'))
import torch
from mmaction.models import build_model
assert not torch.cuda.is_initialized()
m=build_model(dict(type='Recognizer2D',backbone=dict(type='ResNet',pretrained=None,depth=50,norm_eval=False),train_cfg=None,test_cfg=dict(feature_extraction=True)))
p=next((B/'assets').glob('tsn*.pth'))
c=torch.load(p,map_location='cpu',weights_only=False)
s=c['state_dict']; expected=m.state_dict();rows=[]
missing=[];wrong=[]
for k,v in expected.items():
 source=k if k in s else None
 if source is None:missing.append(k)
 elif list(s[source].shape)!=list(v.shape):wrong.append(k)
 rows.append(dict(expected=k,checkpoint_key=source,shape=list(v.shape),checkpoint_shape=list(s[source].shape) if source else None))
extra=sorted(set(s)-set(expected))
assert not missing and not wrong,(missing,wrong)
result=m.load_state_dict({k:s[k] for k in expected},strict=True)
assert not result.missing_keys and not result.unexpected_keys
assert all(k.startswith('cls_head.') for k in extra),extra
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_cfg=dict(type='Recognizer2D',backbone=dict(type='ResNet',pretrained=None,depth=50,norm_eval=False),train_cfg=None,test_cfg=dict(feature_extraction=True)),mapped_count=len(rows),missing=missing,wrong_shape=wrong,extra_classifier_keys=extra,mapping=rows,strict_encoder_load=True,forward_calls=0,gpu_initialized=torch.cuda.is_initialized(),torch=torch.__version__)
(B/'BACKBONE_STATE_MAPPING.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='mapping'}))
