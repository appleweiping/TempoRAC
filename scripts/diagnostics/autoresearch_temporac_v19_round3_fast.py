from pathlib import Path
import json,numpy as np
from autoresearch_temporac_v19_round2 import records,feature,score
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier

def main():
 root=Path(__file__).resolve().parents[2];tr=records(root,'train');va=records(root,'val');X=np.stack([feature(r) for r in tr]);y=np.array([r[2] for r in tr]);V=np.stack([feature(r) for r in va]);Y=np.array([r[2] for r in va]);vid=np.array([r[3] for r in va]);models={}
 for leaf in (3,5,8,12):
  models[f'lgb_l1_{leaf}']=LGBMRegressor(n_estimators=500,learning_rate=.02,num_leaves=15,min_child_samples=leaf,objective='regression_l1',verbosity=-1,random_state=49)
  models[f'lgb_l2_{leaf}']=LGBMRegressor(n_estimators=500,learning_rate=.02,num_leaves=15,min_child_samples=leaf,objective='regression',verbosity=-1,random_state=49)
 for depth in (2,3,4): models[f'xgb_{depth}']=XGBRegressor(n_estimators=600,max_depth=depth,learning_rate=.02,subsample=.8,colsample_bytree=.4,objective='reg:absoluteerror',n_jobs=8,random_state=49)
 models['etc']=ExtraTreesClassifier(n_estimators=300,max_features='sqrt',class_weight='balanced',n_jobs=8,random_state=49)
 pred={};res={}
 for k,m in models.items():m.fit(X,y.astype(int) if k=='etc' else y);pred[k]=m.predict(V);res[k]=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,pred[k],vid)))
 reg=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.5,n_jobs=8,random_state=29).fit(X,y);pred['extra4']=1.05*reg.predict(V)-.5;res['extra4']=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,pred['extra4'],vid)))
 names=list(pred)
 for i,a in enumerate(names):
  for b in names[i+1:]:
   for w in np.arange(.1,1.,.1):res[f'blend:{a}:{b}:{w:.1f}']=dict(zip(('avg_mae','avg_obo','mean_pred'),score(Y,w*pred[a]+(1-w)*pred[b],vid)))
 bestm=min(res,key=lambda k:(res[k]['avg_mae'],-res[k]['avg_obo']));besto=max(res,key=lambda k:(res[k]['avg_obo'],-res[k]['avg_mae']));passing=[k for k,z in res.items() if z['avg_mae']<=.168 and z['avg_obo']>=.725];out={'schema':'temporac.autoresearch-v19.round3-fast','best_mae':bestm,'best_mae_metrics':res[bestm],'best_obo':besto,'best_obo_metrics':res[besto],'passing':passing,'passing_metrics':{k:res[k] for k in passing},'base_results':{k:res[k] for k in names},'test_used':False};(root/'results/autoresearch-v19/round3-fast.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
