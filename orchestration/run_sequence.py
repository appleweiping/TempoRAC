"""One ordered full-data NTR1 campaign; no retries, cap overrides or new models."""
import hashlib
import argparse
import os
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

R = Path('/mnt/4tssd/lilei/TempoRAC')
W = R / 'work/ntr1-full-execution-20261104'
CODE = R / 'repo'
CLI = CODE / 'scripts/train_temporac_native_response.py'


def source_head():
    return json.loads((W/'execution_freeze.json').read_text())['source_commit']


def ref(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def write_state(data):
    temp = W / 'campaign_state.json.tmp'
    temp.write_text(json.dumps(data, indent=2, sort_keys=True))
    temp.replace(W / 'campaign_state.json')


def run(stage, seed, variant):
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip() != source_head():
        raise ValueError('production source changed')
    cfg = W / 'config' / f'{seed}-{variant}.json'
    log = W / 'logs' / f'{seed}-{variant}-{stage}.log'
    if log.exists():
        raise ValueError('existing stage log requires explicit continuation; refusing an automatic repeat')
    argv = [sys.executable, '-B', str(CLI), stage, '--config', str(cfg)]
    start = time.time()
    with log.open('x') as stream:
        child = subprocess.Popen(argv, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        state = dict(schema='temporac.ntr1.campaign-state.v1', status='running', stage=stage,
            seed=seed, variant=variant, pid=child.pid, source_commit=source_head(), command=argv,
            started_utc=datetime.now(timezone.utc).isoformat(), log=str(log),
            config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest())
        write_state(state)
        rc = child.wait()
    state.update(status='stage_complete' if rc == 0 else 'stage_failed', exit_code=rc,
                 elapsed_seconds=time.time()-start, ended_utc=datetime.now(timezone.utc).isoformat())
    (W / 'logs' / f'{seed}-{variant}-{stage}.receipt.json').write_text(json.dumps(state,indent=2,sort_keys=True))
    write_state(state)
    if rc:
        raise RuntimeError(f'{stage}/{seed}/{variant} exit {rc}; preserve budget/recovery and stop campaign')


def score_arm(seed, variant):
    freeze_path=W/'execution_freeze.json'
    freeze=json.loads(freeze_path.read_text())
    arm=W/'runs'/str(seed)/variant
    artifacts=freeze['gate']['artifacts']
    # Parent supplies the exact diagnostic row reference, never a GT path to runtime.
    rows=freeze['gate']['diagnostic_identity_rows']
    selection=json.loads((arm/'selection.json').read_text())
    config={'protocol':'temporac.native-prediction-score.v1','source_commit':source_head(),
        'seed':seed,'variant':variant,'prediction_root':str(arm),
        'output':str(W/'scores'/str(seed)/variant),'inputs':{
            'diagnostics':artifacts['diagnostics'],'identity_rows':rows,
            'count_policy':artifacts['count_policy'],'execution_freeze':ref(freeze_path),
            'plan':ref(arm.parent/'frozen_plan.json'),'selection':ref(arm/'selection.json'),
            'predictions':ref(arm/'predictions.json'),'checkpoint':ref(arm/selection['selected']['file'])}}
    path=W/'config'/f'{seed}-{variant}-score.json'
    with path.open('x') as f:json.dump(config,f,sort_keys=True,indent=2)
    log=W/'logs'/f'{seed}-{variant}-score.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    argv=[sys.executable,'-B',str(CODE/'scripts/score_temporac_native_predictions.py'),'--config',str(path)]
    with log.open('x') as stream:
        child=subprocess.Popen(argv,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_state({'status':'isolated_scoring','seed':seed,'variant':variant,'pid':child.pid,'command':argv})
        rc=child.wait()
    receipt={'status':'score_complete' if rc==0 else 'score_failed','exit_code':rc,
             'seed':seed,'variant':variant,'command':argv,'log':str(log)}
    (W/'logs'/f'{seed}-{variant}-score.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True))
    write_state(receipt)
    if rc:raise RuntimeError('isolated scoring failed; remaining arms blocked')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--remaining-release', type=Path)
    args = parser.parse_args()
    if not (W / 'execution_freeze.json').exists():
        raise ValueError('final parent gates and resource freeze are missing')
    (W / 'logs').mkdir(parents=True, exist_ok=True)
    seeds = (20260815,)
    if args.remaining_release:
        release = json.loads(args.remaining_release.read_text())
        if release.get('decision') != 'continue_remaining_four' or release.get('source_commit') != source_head():
            raise ValueError('remaining arms need parent release after first-pair scoring/diagnosis/resource arrangement')
        arrangement=release['resource_arrangement']
        if ref(Path(arrangement['path'])) != arrangement:
            raise ValueError('parent resource arrangement missing/changed')
        for variant in ('canonical', 'capacity-control'):
            receipts = release['first_pair'][variant]
            for kind in ('selection', 'predictions', 'score'):
                item = receipts[kind]
                if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
                    raise ValueError('first-pair release artifact changed: '+kind)
            expected=W/'scores'/'20260815'/variant/'score.json'
            report=json.loads(expected.read_text())
            if (ref(expected)!=receipts['score'] or report['status']!='complete_population'
                    or report['seed']!=20260815 or report['variant']!=variant
                    or report['source_commit']!=source_head()
                    or report['full_population']['videos']!=183 or report['full_population']['identities']!=486):
                raise ValueError('first-pair complete scores absent from release')
            arm=W/'runs'/'20260815'/variant
            for kind in ('selection','predictions'):
                expected_input=arm/(kind+'.json')
                if (ref(expected_input)!=receipts[kind]
                        or report['receipt']['inputs_sha256'].get(str(expected_input.resolve()))!=receipts[kind]['sha256']):
                    raise ValueError('first-pair release differs from scored '+kind)
        seeds = (20260816, 20260817)
    for seed in seeds:
        # One full plan/validate pair per seed. Arm is excluded from the frozen paired-plan identity.
        run('plan', seed, 'canonical')
        run('validate', seed, 'canonical')
        p = json.loads((W/'runs'/str(seed)/'plan.json').read_text())
        v = json.loads((W/'runs'/str(seed)/'validate.json').read_text())
        if p['plan_sha256'] != v['plan_sha256'] or p['pending'] or v['pending']:
            raise ValueError('full plan/validate mismatch or pending input')
        for variant in ('canonical', 'capacity-control'):
            run('train', seed, variant)
            run('predict', seed, variant)
            out = W/'runs'/str(seed)/variant
            prediction = json.loads((out/'predictions.json').read_text())
            if prediction['status'] != 'complete_population' or len(prediction['records']) != 486:
                raise ValueError('full prediction population incomplete')
            score_arm(seed,variant)
    write_state(dict(schema='temporac.ntr1.campaign-state.v1',
        status='remaining_four_scored' if args.remaining_release else 'first_pair_scored_awaiting_parent_diagnosis_and_resource_release',
        seeds=list(seeds), variants=['canonical','capacity-control'],
        source_commit=source_head(), isolated_scoring='complete', scientific_acceptance='not_assigned',
        ended_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
