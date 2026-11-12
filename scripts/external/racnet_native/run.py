"""Fail-closed native RACnet entry point. CPU preparation never dispatches CUDA."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from native_common import ROOT, PACKET, contract, digest, identity, load, savejson, sha


def plan(work):
    c = contract()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    source = Path(c['paths']['official_source'])
    evidence = load(PACKET / 'SOURCE_SCHEMA_EVIDENCE.json')
    checks = {}
    for name, expected in evidence['racnet_source_sha256'].items():
        actual = sha(source / name)
        assert actual == expected, ('official source mismatch', name)
        checks[name] = actual
    for name in ('model.py', 'dataset/gen_gt.py', 'dataset/cal_count.py', 'dataset/adaptive_loader.py'):
        dest = work / 'vendor' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == checks[name], ('science copy changed', name)
        else:
            shutil.copy2(source / name, dest)
    savejson(work / 'PLAN.json', {'status': 'SOURCE_PREPARATION_ONLY', 'source_sha256': checks,
             'contract_sha256': sha(PACKET / 'RUN_CONTRACT.json'), 'base_commit': '59bec947fbe7a6afc63efe847b868c6b355df0c6',
             'populations': c['inputs']['populations'], 'training_released': False, 'gpu_extraction_released': False})


def runtime(work):
    import torch
    torch.set_num_threads(1)
    packages = {}
    for distribution, module in [('torch', 'torch'), ('torchvision', 'torchvision'), ('numpy', 'numpy'),
                                 ('scipy', 'scipy'), ('opencv-python-headless', 'cv2'), ('mmcv', 'mmcv'),
                                 ('timm', 'timm'), ('einops', 'einops'), ('PyYAML', 'yaml'), ('matplotlib', 'matplotlib')]:
        obj = __import__(module)
        packages[distribution] = {'version': obj.__version__ if distribution in ('torch', 'torchvision') else importlib.metadata.version(distribution),
                                  'distribution_version': importlib.metadata.version(distribution), 'path': obj.__file__,
                                  'module_sha256': sha(obj.__file__)}
    expected = {'torch': '2.5.1+cu124', 'torchvision': '0.20.1+cu124', 'numpy': '1.26.4', 'scipy': '1.14.1',
                'opencv-python-headless': '4.10.0.84', 'mmcv': '1.4.0', 'timm': '0.4.12', 'einops': '0.3.2'}
    assert platform.python_version() == '3.10.22'
    assert all(packages[k]['version'] == v for k, v in expected.items()), {k: packages[k]['version'] for k in expected}
    receipt = {'status': 'CPU_IMPORT_ONLY', 'python': platform.python_version(), 'executable': sys.executable,
               'packages': packages, 'sys_path': sys.path, 'gpu_kernel_validation': 'PENDING_FORMAL_FULL_WORK',
               'model_forward': False, 'cpu_threads': torch.get_num_threads()}
    savejson(Path(work) / 'RUNTIME.json', receipt)
    return receipt


def cpu_check(work):
    import contextlib
    import io
    import numpy as np
    import torch
    from native_common import atomic, science, rng_state, restore_rng, module
    from native_inputs import supervised_collate
    from native_extract import backbone
    from native_engine import save_epoch, committed
    work = Path(work)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU check requires CUDA hidden'
    assert not torch.cuda.is_initialized()
    runtime(work)
    model_module, gt, decoder = science(work)
    # Import and construction only; never call the model.
    native = model_module.RACnet(4, 10, 64)
    native_parameters = sum(p.numel() for p in native.parameters())
    del native
    ready = load(work / 'inputs/READINESS.json')
    targets = load(work / 'inputs/targets.json')
    assert len(targets) == 886
    for r in targets:
        assert sha(r['target_path']) == r['target_sha256']
        with np.load(r['target_path'], allow_pickle=False) as data:
            assert data['start'].shape == data['end'].shape == (r['length'],)
            assert np.isfinite(data['start']).all() and np.isfinite(data['end']).all()
    # Border and constant semantics use the untouched author decoder.
    seq = torch.tensor([[1., 0., 1., 0., 1.], [0., 0., 0., 0., 0.]])
    assert decoder.count_peaks(seq).tolist() == [3., 0.]
    assert decoder.count_peaks_scaling(seq, pro=0.2).tolist() == [3., 0.]
    # Exact native collate correspondence, with masks independent from labels.
    rows = [(torch.ones(5, 2), torch.ones(5, 5), (torch.zeros(5), torch.ones(5)), 'a'),
            (torch.ones(3, 2), torch.ones(3, 3), (torch.zeros(3), torch.ones(3)), 'b')]
    ours = supervised_collate(rows)
    original = module('native_original_collate', work / 'vendor/dataset/adaptive_loader.py').collateT_fn(rows)
    assert torch.equal(ours['masks'], original['masks'].squeeze(-1))
    for k in ('features', 'target_s', 'target_e', 'gt_tsm', 'tsm_masks'):
        assert torch.equal(ours[k], original[k]), k
    _, strict = backbone(work)
    # Full state persistence primitives, using one CPU scalar (no neural forward).
    primitive = work / 'cpu_state_primitive'
    assert not primitive.exists(), 'CPU primitive already exists; preserve prior receipt and use targeted closure'
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.Adam([p], lr=6.4e-5)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [40, 80, 120, 160], gamma=.9)
    scaler = torch.amp.GradScaler('cpu')
    scaler.scale(p.square()).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()
    state = {'identity': {'primitive': True}, 'next_epoch': 0, 'model': {'scalar': p.detach()},
             'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'scaler': scaler.state_dict(),
             'rng': rng_state(cuda=False), 'loader_generator': None, 'sampler': 'global Torch RNG',
             'attempted_steps': 1, 'actual_steps': 1, 'amp_skips': 0, 'epochs': []}
    save_epoch(primitive, state)
    restored, ref = committed(primitive)
    assert restored['scheduler'] == state['scheduler'] and restored['scaler'] == state['scaler']
    q = torch.nn.Parameter(torch.tensor(0.))
    opt2 = torch.optim.Adam([q], lr=6.4e-5)
    opt2.load_state_dict(restored['optimizer'])
    assert torch.equal(opt2.state[q]['exp_avg'], opt.state[p]['exp_avg'])
    restore_rng(restored['rng'])
    witness = torch.rand(4)
    restore_rng(restored['rng'])
    assert torch.equal(witness, torch.rand(4))
    assert len(list((primitive / 'checkpoints').glob('epoch_*.pt'))) == 1
    metric_path = ROOT / 'source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py'
    assert sha(metric_path) == contract()['evaluation']['common_metric_sha256']
    metric = module('frozen_common_metrics', metric_path)
    assert metric.count_metrics([0, 2], [0, 2])['mae_p'] == 0
    assert not torch.cuda.is_initialized()
    report = {'status': 'PASS' if ready['status'] == 'PASS' else 'PENDING',
              'candidate_identity': identity(work), 'readiness_sha256': sha(work / 'inputs/READINESS.json'),
              'passed': ['native imports and parameter construction without forward', '886 native targets',
                         'original decoder border+constant semantics', 'feature-length masks/native padding equality',
                         'strict complete official Swin backbone parameters+buffers', 'atomic full-state primitives',
                         'RNG roundtrip', 'frozen metric API'],
              'native_parameter_count': native_parameters, 'backbone': strict,
              'state_primitive_checkpoint_sha256': ref['sha256'], 'no_model_forward': True,
              'cuda_initialized': False, 'pending': {'author_missing': [r for r in ready['missing'] if r['reason'] == 'author member pending'],
                                                    'QUVA_missing': [r for r in ready['missing'] if 'QUVA' in r['reason']],
                                                    'fresh_engineering_review': 'root-owned after full inputs ready',
                                                    'GPU_ownership': 'NOT_RELEASED'}}
    savejson(work / 'CPU_CHECK.json', report)
    return report


def refresh_cpu(work):
    """Reuse completed checks after new input receipts; no repeated model/key primitives."""
    import torch
    work = Path(work)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    previous = load(work / 'CPU_CHECK.json')
    assert len(previous['passed']) == 8 and previous['no_model_forward'] and not previous['cuda_initialized']
    current = identity(work)
    for field in ('contract', 'env_spec', 'runtime', 'science'):
        assert current[field] == previous['candidate_identity'][field], ('previous check not reusable', field)
    # Compile all actual candidate sources in memory: no pyc or model invocation.
    for source in sorted(work.glob('*.py')):
        compile(source.read_text(), str(source), 'exec')
    ready = load(work / 'inputs/READINESS.json')
    for name, expected in ready['hashes'].items():
        assert sha(work / 'inputs' / (name + '.json')) == expected
    from native_common import storage_guard
    storage_guard(work)
    # Retain the complete prior receipt; every update names its exact inherited evidence.
    old_sha = sha(work / 'CPU_CHECK.json')
    archive = work / ('CPU_CHECK_' + old_sha + '.json')
    if not archive.exists():
        shutil.copy2(work / 'CPU_CHECK.json', archive)
    previous.update(status='PASS' if ready['status'] == 'PASS' else 'PENDING',
                    candidate_identity=current, readiness_sha256=sha(work / 'inputs/READINESS.json'),
                    inherited_cpu_check={'path': str(archive), 'sha256': old_sha},
                    targeted_closure=['in-memory syntax compile', 'full manifest receipt hashes', 'storage guard'],
                    pending={'author_missing': [r for r in ready['missing'] if r['reason'] == 'author member pending'],
                             'QUVA_missing': [r for r in ready['missing'] if 'QUVA' in r['reason']],
                             'fresh_engineering_review': 'root-owned after full inputs ready', 'GPU_ownership': 'NOT_RELEASED'})
    savejson(work / 'CPU_CHECK.json', previous)
    return previous


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['plan', 'runtime', 'prepare', 'refresh-inputs', 'cpu-check', 'cpu-refresh', 'extract-quva', 'train', 'evaluate'])
    p.add_argument('--work', type=Path, default=ROOT / 'baselines/racnet_native_v1')
    p.add_argument('--ownership-receipt', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    assert args.work.resolve() == (ROOT / 'baselines/racnet_native_v1').resolve(), 'candidate path fixed'
    if args.action == 'plan':
        plan(args.work)
    elif args.action == 'runtime':
        runtime(args.work)
    elif args.action == 'prepare':
        from native_inputs import prepare
        report = prepare(args.work)
        print({k: v for k, v in report.items() if k not in ('missing', 'index_snapshots')})
    elif args.action == 'cpu-check':
        report = cpu_check(args.work)
        print({'status': report['status'], 'passed': report['passed'], 'cuda_initialized': False})
    elif args.action == 'refresh-inputs':
        from native_inputs import refresh_inputs
        report = refresh_inputs(args.work)
        print({k: v for k, v in report.items() if k not in ('missing', 'index_snapshots', 'inherited_supervision')})
    elif args.action == 'cpu-refresh':
        report = refresh_cpu(args.work)
        print({'status': report['status'], 'targeted_closure': report['targeted_closure']})
    else:
        assert args.ownership_receipt, 'formal GPU work requires actual root ownership receipt'
        import fcntl
        lock = open(args.work / 'ACTIVE.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'extract-quva':
            from native_extract import extract
            extract(args.work, args.ownership_receipt)
        else:
            from native_engine import train, evaluate
            if args.action == 'train':
                train(args.work, args.ownership_receipt, args.resume)
            else:
                evaluate(args.work, args.ownership_receipt)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        # Includes allocation/import/gate failures before an epoch can be entered.
        import time
        work = ROOT / 'baselines/racnet_native_v1'
        if work.is_dir():
            savejson(work / ('ENTRY_FAILURE_%d.json' % time.time_ns()),
                     {'status': 'FAILED', 'error': repr(exc), 'argv': sys.argv,
                      'traceback': traceback.format_exc(), 'no_automatic_retry': True})
        raise
