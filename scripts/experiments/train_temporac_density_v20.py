"""v20 supervised temporal density counter; all v18/v19 sources remain unchanged."""
from pathlib import Path
import argparse,json,random
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from pams.warp_phase.packing import trusted_load_source_pickle

class Block(nn.Module):
 def __init__(self,d,dil):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(d,d,5,padding=2*dil,dilation=dil,groups=d),nn.Conv1d(d,d,1),nn.GroupNorm(8,d),nn.GELU(),nn.Conv1d(d,d,1),nn.Dropout(.05))
 def forward(self,x):return x+self.net(x)
class Counter(nn.Module):
 def __init__(self):
  super().__init__();self.stem=nn.Conv1d(120,128,5,padding=2);self.blocks=nn.Sequential(*[Block(128,d) for d in (1,2,4,8,16,1,2,4)]);self.head=nn.Sequential(nn.Conv1d(128,64,1),nn.GELU(),nn.Conv1d(64,1,1),nn.Softplus())
 def forward(self,x,mask):
  den=self.head(self.blocks(self.stem(x))).squeeze(1)*mask
  return den,den.sum(1)
def load(root,split):
 o,_=trusted_load_source_pickle(root/'data/counting_multirep_skeleton_pose_expanded_v44_len320'/f'{split}.pkl',repository_root=root,split=split);xs=[];ds=[];ys=[];vs=[]
 for s in o[split]:
  for p,on in enumerate(s['person_mask']):
   if not on:continue
   x=np.asarray(s['motion'][p],np.float32);m=np.asarray(s['frame_mask'][p],np.float32);center=x.mean(1,keepdims=True);z=x-center;scale=np.median(np.linalg.norm(z,axis=-1)[m>0]);z/=max(scale,1e-4);vel=np.diff(z,axis=0,prepend=z[:1]);acc=np.diff(vel,axis=0,prepend=vel[:1]);t=np.linspace(-1,1,len(x),dtype=np.float32)[:,None];feat=np.concatenate([z.reshape(320,-1),vel.reshape(320,-1),np.linalg.norm(vel,axis=-1),m[:,None]],1);xs.append(feat.T);ds.append(np.asarray(s['density_gt'][p],np.float32));ys.append(float(s['count_gt'][p]));vs.append(str(s['canonical_video_id']))
 return torch.tensor(np.stack(xs)),torch.tensor(np.stack(ds)),torch.tensor(ys),np.array(vs)
def metric(y,p,v):
 p=np.maximum(0,np.rint(p));A=[];O=[]
 for k in sorted(set(v)):
  q=v==k;A.append(np.mean(abs(p[q]-y[q])/y[q]));O.append(np.mean(abs(p[q]-y[q])<=1))
 return float(np.mean(A)),float(np.mean(O))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repository',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=600);ap.add_argument('--seed',type=int,default=20260925);a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);root=a.repository.resolve();X,D,Y,V=load(root,'train');XV,DV,YV,VV=load(root,'val');dev=torch.device('cuda:0');model=Counter().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);loader=DataLoader(TensorDataset(X,D,Y),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(a.seed));best=(9.,0.);a.output.mkdir(parents=True,exist_ok=False);log=open(a.output/'metrics.jsonl','w')
 for ep in range(1,a.epochs+1):
  model.train();tot=0
  for x,d,y in loader:
   x,d,y=x.to(dev),d.to(dev),y.to(dev);mask=x[:,-1];opt.zero_grad();pd,pc=model(x,mask);count=torch.nn.functional.smooth_l1_loss(pc,y,beta=1.);density=torch.nn.functional.smooth_l1_loss(pd,d,beta=.02);loss=count+.25*density;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();tot+=float(loss)
  if ep==1 or ep%10==0:
   model.eval();pp=[]
   with torch.no_grad():
    for i in range(0,len(XV),64):
     x=XV[i:i+64].to(dev);pp.extend(model(x,x[:,-1])[1].cpu().tolist())
   mae,obo=metric(YV.numpy(),np.array(pp),VV);row={'epoch':ep,'train_loss':tot/len(loader),'avg_mae':mae,'avg_obo':obo};print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
   if (mae,-obo)<(best[0],-best[1]):best=(mae,obo);torch.save({'format':'temporac-density-v20','epoch':ep,'model':model.state_dict(),'metrics':row,'seed':a.seed,'test_used':False},a.output/'best.pt')
   if mae<=.168 and obo>=.725:break
 log.close();(a.output/'completion.json').write_text(json.dumps({'completed_epoch':ep,'best_avg_mae':best[0],'best_avg_obo':best[1],'target_reached':best[0]<=.168 and best[1]>=.725,'test_used':False},indent=2))
if __name__=='__main__':main()
