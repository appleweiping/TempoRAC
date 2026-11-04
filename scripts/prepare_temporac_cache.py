"""CPU-only explicit preparation of immutable NTR1 caches."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pams.temporac.native_cached_training_runtime_v1 import load_verified_frozen_plan
from pams.temporac.cache_build_v1 import build_caches
from pams.temporac.native_training_runtime_v1 import _checked_path
from pams.temporac.native_prepared_cache_v1 import file_sha,atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','config-sha','plan','plan-sha','validate','validate-sha','cache-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=('inventory','train','dev'),default='train')
    p.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    a=p.parse_args()
    if os.name=='nt' and a.workers>1:raise ValueError('parallel cache preparation is Linux-only')
    if file_sha(a.config)!=a.config_sha:raise ValueError('original config changed')
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    cache_root=_checked_path(a.cache_root)
    for name in ('bridge_root','target_root','x0_root'):
        source=Path(cfg[name]).resolve()
        if cache_root==source or cache_root in source.parents or source in cache_root.parents:
            raise ValueError('cache root must be separate from immutable inputs')
    plan=load_verified_frozen_plan({'path':a.plan,'sha256':a.plan_sha},{'path':a.validate,'sha256':a.validate_sha})
    try:
        result=build_caches(cfg,plan.payload,a.cache_root,a.phase,workers=a.workers)
    except BaseException as exc:
        root=Path(a.cache_root);previous={}
        if (root/'progress.json').exists():previous=json.loads((root/'progress.json').read_text())
        atomic_json(root/'progress.json',dict(previous,phase=a.phase+'_failed',error=str(exc),failed=previous.get('failed',0)+1))
        raise
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
