"""Autoresearch v19 round 2: channel-resolved spectral counting."""
from pathlib import Path
import json, numpy as np
from pams.warp_phase.packing import trusted_load_source_pickle
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor,RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def records(root,split):
 o,_=trusted_load_source_pickle(root/'data/counting_multirep_skeleton_pose_expanded_v44_len320'/f'{split}.pkl',repository_root=root,split=split); out=[]
 for s in o[split]:
  for pi,m in enumerate(s['person_mask']):
   if m: out.append((s['motion'][pi],s['frame_mask'][pi],float(s['count_gt'][pi]),str(s['canonical_video_id']),float(s['source_length'])))
 return out

def feature(row):
 x,mask,y,v,L=row; x=np.asarray(x,float)[np.asarray(mask,bool)]; n=len(x); c=x.mean(1,keepdims=True); z=x-c; sc=np.median(np.linalg.norm(z,axis=-1)); z/=max(sc,1e-6)
 vel=np.diff(z,axis=0,prepend=z[:1]); signals=np.concatenate([z.reshape(n,-1),np.linalg.norm(vel,axis=-1)],axis=1)
 signals=signals-signals.mean(0); p=np.abs(np.fft.rfft(signals*np.hanning(n)[:,None],axis=0))**2; p=p/(p.sum(0,keepdims=True)+1e-9)
 B=np.zeros((80,p.shape[1])); q=min(80,len(p)-1); B[:q]=p[1:q+1]
 dom=np.argmax(B,axis=0)+1
 # channel-resolved spectrum + ranked frequencies and confidence
 top=np.argsort(B,axis=0)[-5:][::-1]+1; conf=np.sort(B,axis=0)[-5:][::-1]
 stats=np.array([n,L,L/max(n,1),sc,np.mean(np.abs(vel)),np.std(vel)])
 return np.r_[stats,B.T.ravel(),dom,top.T.ravel(),conf.T.ravel()]

def score(y,p,v):
 p=np.maximum(0,np.rint(p)); A=[];O=[]
 for k in sorted(set(v)):
  q=v==k; A.append(np.mean(abs(p[q]-y[q])/y[q]));O.append(np.mean(abs(p[q]-y[q])<=1))
 return float(np.mean(A)),float(np.mean(O)),float(np.mean(p))

def main():
 root=Path('.').resolve(); tr=records(root,'train'); va=records(root,'val'); X=np.stack([feature(r) for r in tr]); y=np.array([r[2] for r in tr]);V=np.stack([feature(r) for r in va]);Y=np.array([r[2] for r in va]);vid=np.array([r[3] for r in va])
 models={}
 for a in (10,30,100,300,1000): models[f'ridge{a}']=make_pipeline(StandardScaler(),Ridge(alpha=a))
 for leaf in (1,2,3,4,6): models[f'extra{leaf}']=ExtraTreesRegressor(n_estimators=1000,min_samples_leaf=leaf,max_features=.5,n_jobs=-1,random_state=29)
 pred={};res={}
 for k,m in models.items(): m.fit(X,y);pred[k]=m.predict(V);res[k]=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,pred[k],vid)))
 # affine calibrations and blends selected on development
 base=list(pred)
 for k in base:
  for scale in np.arange(.7,1.31,.05):
   for bias in np.arange(-3,3.1,.5):
    name=f'cal:{k}:{scale:.2f}:{bias:.1f}';res[name]=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,pred[k]*scale+bias,vid)))
 for i,a in enumerate(base):
  for b in base[i+1:]:
   for w in np.arange(.1,1.,.1):
    name=f'blend:{a}:{b}:{w:.1f}';res[name]=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,w*pred[a]+(1-w)*pred[b],vid)))
 best_mae=min(res,key=lambda k:(res[k]['avg_mae'],-res[k]['avg_obo']));best_obo=max(res,key=lambda k:(res[k]['avg_obo'],-res[k]['avg_mae']))
 out={'schema':'temporac.autoresearch-v19.round2','feature_dim':X.shape[1],'best_mae':best_mae,'best_mae_metrics':res[best_mae],'best_obo':best_obo,'best_obo_metrics':res[best_obo],'base_results':{k:res[k] for k in base},'target':{'avg_mae':.168,'avg_obo':.725},'test_used':False}
 Path('results/autoresearch-v19/round2.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
