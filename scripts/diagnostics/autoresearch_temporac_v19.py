"""Train-supervised autoresearch probe for MultiRep v44; v18 files stay unchanged."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pams.warp_phase.packing import trusted_load_source_pickle
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor,RandomForestRegressor,GradientBoostingRegressor,RandomForestRegressor,StackingRegressor,RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,ElasticNet


def one_feature(motion,mask,source_length):
    x=np.asarray(motion,np.float64); m=np.asarray(mask,bool)
    x=x[m]; n=len(x)
    if n<4: return np.zeros(620,np.float64)
    # translation/scale normalized pose plus raw center motion
    center=np.nanmean(x,axis=1,keepdims=True)
    centered=x-center
    scale=np.nanmedian(np.linalg.norm(centered,axis=-1),axis=1)
    scale=np.nanmedian(scale[scale>1e-6]) if np.any(scale>1e-6) else 1.
    z=centered/scale
    # signals: joint xyz, velocity magnitudes, selected limb distances
    sig=z.reshape(n,-1)
    vel=np.diff(z,axis=0,prepend=z[:1])
    acc=np.diff(vel,axis=0,prepend=vel[:1])
    vm=np.linalg.norm(vel,axis=-1)
    am=np.linalg.norm(acc,axis=-1)
    # robust stats per coordinate/joint
    feats=[np.array([n,source_length,source_length/max(n,1),scale]),
      np.nanmean(np.abs(vel),axis=0).ravel(),np.nanstd(vel,axis=0).ravel(),
      np.nanmean(vm,axis=0),np.nanstd(vm,axis=0),np.nanquantile(vm,[.25,.5,.75,.9],axis=0).ravel(),
      np.nanmean(am,axis=0),np.nanstd(am,axis=0)]
    # power by temporal cycle index; normalization removes amplitude
    pool=np.concatenate([sig,vm],axis=1)
    pool=pool-np.mean(pool,axis=0,keepdims=True)
    win=np.hanning(n)[:,None]
    power=np.abs(np.fft.rfft(pool*win,axis=0))**2
    power=power/(power.sum(0,keepdims=True)+1e-12)
    k=min(96,power.shape[0]-1)
    band=power[1:k+1]
    feats += [np.mean(band,axis=1),np.median(band,axis=1),np.max(band,axis=1),np.quantile(band,.9,axis=1)]
    # dominant frequency summaries across channels
    dom=1+np.argmax(band,axis=0)
    feats += [np.quantile(dom,[0,.1,.25,.5,.75,.9,1]),np.bincount(dom,minlength=97)[1:97]/len(dom)]
    out=np.concatenate([np.atleast_1d(v).ravel() for v in feats])
    out=np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.)
    # stable fixed width
    if len(out)<620: out=np.pad(out,(0,620-len(out)))
    return out[:620]

def load(root,split):
    obj,_=trusted_load_source_pickle(root/'data/counting_multirep_skeleton_pose_expanded_v44_len320'/f'{split}.pkl',repository_root=root,split=split)
    X=[]; y=[]; vids=[]; persons=[]
    for vi,s in enumerate(obj[split]):
      for pi,valid in enumerate(s['person_mask']):
       if not valid: continue
       X.append(one_feature(s['motion'][pi],s['frame_mask'][pi],s['source_length']))
       y.append(float(s['count_gt'][pi])); vids.append(str(s['canonical_video_id'])); persons.append(pi)
    return np.stack(X),np.array(y),np.array(vids),np.array(persons)

def score(y,p,vids):
    p=np.maximum(0,np.rint(p))
    vals=[]; obs=[]
    for v in sorted(set(vids)):
      q=vids==v; vals.append(np.mean(np.abs(p[q]-y[q])/y[q])); obs.append(np.mean(np.abs(p[q]-y[q])<=1))
    return {'avg_mae':float(np.mean(vals)),'avg_obo':float(np.mean(obs)),'mean_pred':float(np.mean(p))}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repository',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); root=a.repository.resolve()
 X,y,v,_=load(root,'train'); XV,yV,vV,pV=load(root,'val')
 models={
  'ridge10':make_pipeline(StandardScaler(),Ridge(alpha=10)),
  'ridge100':make_pipeline(StandardScaler(),Ridge(alpha=100)),
  'extra_leaf1':ExtraTreesRegressor(n_estimators=700,min_samples_leaf=1,max_features=.8,n_jobs=-1,random_state=19),
  'extra_leaf2':ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=19),
  'extra_leaf4':ExtraTreesRegressor(n_estimators=700,min_samples_leaf=4,max_features=1.,n_jobs=-1,random_state=19),
  'rf_leaf2':RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=19),
  'gbr':GradientBoostingRegressor(n_estimators=500,max_depth=2,loss='huber',learning_rate=.03,random_state=19),
 }
 preds={}; results={}
 for name,m in models.items():
  m.fit(X,y); preds[name]=m.predict(XV); results[name]=score(yV,preds[name],vV)
 # exhaustive two-model convex blends on dev, retained as autoresearch model selection
 names=list(preds)
 for i,l in enumerate(names):
  for r in names[i+1:]:
   for w in np.linspace(.1,.9,9):
    name=f'blend:{l}:{r}:{w:.1f}'; results[name]=score(yV,w*preds[l]+(1-w)*preds[r],vV)
 best=min(results,key=lambda k:(results[k]['avg_mae'],-results[k]['avg_obo']))
 out={'schema':'temporac.autoresearch-v19.v1','train_identities':len(y),'val_identities':len(yV),'feature_dim':X.shape[1],'results':results,'best':best,'best_metrics':results[best],'target':{'avg_mae_max':.168,'avg_obo_min':.725},'test_used':False}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
