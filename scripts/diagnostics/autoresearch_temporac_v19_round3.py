"""Autoresearch v19 round 3: integer/ordinal heads for AvgOBO."""
from pathlib import Path
import json,numpy as np
from autoresearch_temporac_v19_round2 import records,feature,score
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def main():
 root=Path(__file__).resolve().parents[2];tr=records(root,'train');va=records(root,'val');X=np.stack([feature(r) for r in tr]);y=np.array([int(r[2]) for r in tr]);V=np.stack([feature(r) for r in va]);Y=np.array([r[2] for r in va]);vid=np.array([r[3] for r in va])
 models={}
 for leaf in (1,2,3,4): models[f'etc{leaf}']=ExtraTreesClassifier(n_estimators=1600,min_samples_leaf=leaf,max_features=.6,class_weight='balanced',n_jobs=-1,random_state=39)
 for leaf in (1,2,3): models[f'rfc{leaf}']=RandomForestClassifier(n_estimators=1200,min_samples_leaf=leaf,max_features=.6,class_weight='balanced_subsample',n_jobs=-1,random_state=39)
 for C in (.03,.1,.3,1,3,10): models[f'svc{C}']=make_pipeline(StandardScaler(),PCA(n_components=.98,whiten=True,random_state=39),SVC(C=C,kernel='rbf',class_weight='balanced'))
 pred={};res={}
 for k,m in models.items():
  m.fit(X,y);pred[k]=m.predict(V).astype(float);res[k]=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,pred[k],vid)))
 # combine classifier decisions with strong round2 regression
 reg=ExtraTreesRegressor(n_estimators=1600,min_samples_leaf=4,max_features=.5,n_jobs=-1,random_state=29).fit(X,y); rp=reg.predict(V)
 res['reg']=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,1.05*rp-.5,vid)))
 for k in list(pred):
  for w in np.arange(.1,1.,.1):
   name=f'blend:{k}:reg:{w:.1f}';res[name]=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,w*pred[k]+(1-w)*(1.05*rp-.5),vid)))
 best_mae=min(res,key=lambda k:(res[k]['avg_mae'],-res[k]['avg_obo']));best_obo=max(res,key=lambda k:(res[k]['avg_obo'],-res[k]['avg_mae']))
 passing=[k for k,z in res.items() if z['avg_mae']<=.168 and z['avg_obo']>=.725]
 out={'schema':'temporac.autoresearch-v19.round3','best_mae':best_mae,'best_mae_metrics':res[best_mae],'best_obo':best_obo,'best_obo_metrics':res[best_obo],'passing':passing,'passing_metrics':{k:res[k] for k in passing},'base_results':{k:res[k] for k in models},'test_used':False}
 (root/'results/autoresearch-v19/round3.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
