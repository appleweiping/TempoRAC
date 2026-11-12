"""Native start SSE + masked tReCo, with durable outer execution plumbing."""
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from native_common import (ROOT, atomic, contract, digest, gpu_gate, identity, journal, load,
                           module, restore_rng, rng_state, runtime_flags, savejson, science, seed, sha, syncdir, storage_guard, timing_summary)
from native_inputs import SupervisedInputs, PredictionInputs, supervised_collate


def save_epoch(work, payload, improved_mae=False, improved_obo=False):
    """Payload then one atomic commit pointer; uncommitted saves are never resumed."""
    work = Path(work)
    folder = work / 'checkpoints'
    name = 'epoch_%03d_%s.pt' % (payload['next_epoch']-1, uuid.uuid4().hex)
    dest = folder / name
    atomic(dest, lambda f: torch.save(payload, f))
    ref = {'file': str(dest), 'sha256': sha(dest), 'next_epoch': payload['next_epoch']}
    old = load(folder / 'COMMIT.json') if (folder / 'COMMIT.json').exists() else {}
    commit = {'identity': payload['identity'], 'latest': ref,
              'initial': old.get('initial', ref),
              'first_epoch': old.get('first_epoch') or (ref if payload['next_epoch'] == 1 else None),
              'best_mae': ref if improved_mae else old.get('best_mae'),
              'best_obo': ref if improved_obo else old.get('best_obo')}
    savejson(folder / 'COMMIT.json', commit)
    # Only this candidate's unreferenced generation payloads are reclaimed.
    retained = {v['file'] for v in commit.values() if isinstance(v, dict) and 'file' in v}
    for p in folder.glob('epoch_*.pt'):
        if str(p) not in retained:
            p.unlink()
    syncdir(folder)
    return commit


def committed(work, which='latest'):
    c = load(Path(work) / 'checkpoints/COMMIT.json')
    ref = c[which]
    assert ref and sha(ref['file']) == ref['sha256'], 'checkpoint generation corrupt'
    return torch.load(ref['file'], map_location='cpu', weights_only=False), ref


def batch_forward(batch, model, decoder, training):
    data = batch['features'].type(torch.FloatTensor).to('cuda')
    start = batch['target_s'].type(torch.FloatTensor).to('cuda')
    target_tsm = batch['gt_tsm'].type(torch.FloatTensor).to('cuda')
    masks = [batch['tsm_masks'].to('cuda'), batch['masks'].to('cuda')]
    output, _, out_tsm = model(data, masks, 'H-norm', 'None')
    # This is the published loss computation, including native padded probabilities.
    sse = torch.nn.functional.mse_loss(output, start, reduction='sum')
    treco = torch.nn.functional.mse_loss(out_tsm * (target_tsm == 0.0).logical_not().float(), target_tsm, reduction='sum')
    loss = sse + 1e-5 * treco
    if not torch.isfinite(loss).item() or not torch.isfinite(output).all().item():
        raise FloatingPointError('nonfinite native loss/output; stop without method change')
    count = decoder.count_peaks(batch['target_s']).to('cuda')
    pred = decoder.count_peaks(output).to('cuda')
    native_mae = torch.sum(torch.div(torch.abs(pred-count), count+1e-1)) / pred.flatten().shape[0]
    gaps = (pred-count).reshape(-1).cpu().detach().numpy()
    obo = sum(abs(x) <= 1 for x in gaps.tolist()) / len(gaps)
    return loss, {'loss': float(loss.detach()), 'SSE': float(sse.detach()), 'tReCo': float(treco.detach()),
                  'MAE': native_mae.item(), 'OBO': obo,
                  'counts': count.detach().cpu().tolist(), 'predictions': pred.detach().cpu().tolist(),
                  'keys': list(batch['keys']), 'lengths': batch['lengths']}


def validate(loader, model, decoder):
    model.eval()
    batches = []
    with torch.no_grad():
        for batch in loader:
            # No autocast in the source validation path.
            _, log = batch_forward(batch, model, decoder, False)
            batches.append(log)
    assert len(batches) == 9 and sum(len(b['keys']) for b in batches) == 131
    errors = [abs(p-g)/(g+0.1) for b in batches for p, g in zip(b['predictions'], b['counts'])]
    return {'MAE': float(np.mean([b['MAE'] for b in batches])),
            'OBO': float(np.mean([b['OBO'] for b in batches])),
            'pooled_MAE_secondary': float(np.mean(errors)), 'batches': batches}


def training_identity(work):
    work = Path(work)
    return {'candidate': identity(work), 'readiness': sha(work / 'inputs/READINESS.json'),
            'input_hashes': load(work / 'inputs/READINESS.json')['hashes']}


def train(work, receipt, resume=False):
    work = Path(work)
    gpu_gate(work, receipt, 'train')
    ready = load(work / 'inputs/READINESS.json')
    assert ready['status'] == 'PASS', 'all1243 inputs and61 author objects required'
    check = load(work / 'CPU_CHECK.json')
    assert check['status'] == 'PASS' and check['candidate_identity'] == identity(work)
    assert check['readiness_sha256'] == sha(work / 'inputs/READINESS.json'), 'CPU check stale'
    runid = training_identity(work)
    model_module, _, decoder = science(work)
    flags = runtime_flags()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    seed(2333)
    model = torch.nn.DataParallel(model_module.RACnet(4, 10, 64).to('cuda'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=6.4e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.9)
    scaler = torch.cuda.amp.GradScaler()
    # No explicit separate loader generator: preserve native global RNG semantics.
    trainloader = DataLoader(SupervisedInputs(work, 'rc_train'), batch_size=16, shuffle=True,
                             num_workers=1, pin_memory=False, drop_last=False, collate_fn=supervised_collate)
    validloader = DataLoader(SupervisedInputs(work, 'rc_valid'), batch_size=16, shuffle=False,
                             num_workers=1, pin_memory=False, drop_last=False, collate_fn=supervised_collate)
    assert len(trainloader) == 48
    has_commit = (work / 'checkpoints/COMMIT.json').exists()
    assert resume == has_commit, 'fresh/resume must be explicit; existing trajectory cannot be overwritten'
    state = {'identity': runid, 'next_epoch': 0, 'best_mae': None, 'best_obo': None,
             'epochs': [], 'attempted_steps': 0, 'actual_steps': 0, 'amp_skips': 0,
             'replayed_attempts': 0, 'replayed_known_updates': 0, 'replayed_uncertain_steps': 0,
             'runtime_flags': flags, 'sampler': 'RandomSampler/global Torch RNG', 'loader_generator': None}
    if resume:
        state, _ = committed(work)
        assert state['identity'] == runid
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
    attempt_id = uuid.uuid4().hex
    attempts = work / 'attempts'
    attempts.mkdir(exist_ok=True)
    # Count abandoned attempts since the committed epoch boundary, including uncertain starts.
    replay_starts, replay_ends = {}, {}
    previous_accounted = set(state.get('accounted_attempts', []))
    for p in attempts.glob('*.jsonl'):
        if p.stem in previous_accounted:
            continue
        for line in p.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                # Preserve an interrupted final journal write; its durable start is counted uncertain.
                continue
            row = json.loads(line)
            if row.get('epoch', -1) < state['next_epoch']:
                continue
            key = (p.stem, row.get('epoch'), row.get('batch'))
            if row['event'] == 'step_start':
                replay_starts[key] = row
            elif row['event'] == 'step_end':
                replay_ends[key] = row
        previous_accounted.add(p.stem)
    state['replayed_attempts'] += len(replay_starts)
    state['replayed_known_updates'] += sum(r['actual_update'] for r in replay_ends.values())
    state['replayed_uncertain_steps'] += len(set(replay_starts)-set(replay_ends))
    state['accounted_attempts'] = sorted(previous_accounted)
    step_log = attempts / (attempt_id + '.jsonl')
    def capture():
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), rng=rng_state())
    if not resume:
        capture()
        save_epoch(work, state)
    try:
        for epoch in range(state['next_epoch'], 200):
            storage_guard(work)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            model.train()
            batches, seen = [], []
            for batch_index, batch in enumerate(trainloader):
                journal(step_log, {'event': 'step_start', 'epoch': epoch, 'batch': batch_index,
                                   'keys': list(batch['keys']), 'time': time.time()})
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, log = batch_forward(batch, model, decoder, True)
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                actual = scale_after >= scale_before
                state['attempted_steps'] += 1
                state['actual_steps'] += int(actual)
                state['amp_skips'] += int(not actual)
                log.update(event='step_end', epoch=epoch, batch=batch_index, actual_update=actual,
                           scaled_gradients_finite_by_native_scaler=actual,
                           scale_before=scale_before, scale_after=scale_after,
                           attempted_steps=state['attempted_steps'], time=time.time())
                journal(step_log, log)
                batches.append(log)
                seen.extend(batch['keys'])
            assert len(batches) == 48 and len(seen) == len(set(seen)) == 755
            train_seconds = time.monotonic() - start
            val_start = time.monotonic()
            validation = validate(validloader, model, decoder) if epoch > 10 else None
            validation_seconds = time.monotonic() - val_start
            improved_mae = validation is not None and (state['best_mae'] is None or validation['MAE'] < state['best_mae']['value'])
            improved_obo = validation is not None and (state['best_obo'] is None or validation['OBO'] > state['best_obo']['value'])
            if validation:
                assert math.isfinite(validation['MAE']) and math.isfinite(validation['OBO'])
            if improved_mae:
                state['best_mae'] = {'epoch': epoch, 'value': validation['MAE']}
            if improved_obo:
                state['best_obo'] = {'epoch': epoch, 'value': validation['OBO']}
            scheduler.step()
            record = {'epoch': epoch, 'training_rows': 755, 'attempts': 48,
                      'actual_updates': sum(b['actual_update'] for b in batches),
                      'amp_skips': sum(not b['actual_update'] for b in batches),
                      'order_sha256': digest(seen), 'batches': batches, 'validation': validation,
                      'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                      'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'next_lr': scheduler.get_last_lr()}
            state['epochs'].append(record)
            state['next_epoch'] = epoch + 1
            capture()
            commit = save_epoch(work, state, improved_mae, improved_obo)
            savejson(work / 'epochs' / ('epoch_%03d.json' % epoch), record)
            savejson(work / 'PROGRESS.json', {'status': 'training', 'completed_epochs': epoch+1,
                     'attempted_steps': state['attempted_steps'], 'actual_steps': state['actual_steps'],
                     'amp_skips': state['amp_skips'], 'best_mae': state['best_mae'], 'latest': commit['latest'],
                     'attempt_id': attempt_id})
            storage_guard(work)
        assert state['attempted_steps'] == 9600 and len(state['epochs']) == 200
        assert sum(r['validation'] is not None for r in state['epochs']) == 189
        assert state['best_mae'] is not None, 'no finite eligible validation checkpoint'
        c = load(work / 'checkpoints/COMMIT.json')
        savejson(work / 'TRAINING_COMPLETE.json', {'identity': runid, 'epochs': 200,
                 'attempted_steps': 9600, 'actual_steps': state['actual_steps'], 'amp_skips': state['amp_skips'],
                 'selected': c['best_mae'], 'selection': state['best_mae'], 'tie_break': 'earliest epoch'})
    except BaseException as exc:
        savejson(work / ('FAILURE_%s.json' % attempt_id), {'status': 'FAILED', 'error': repr(exc),
                 'attempt_id': attempt_id, 'next_epoch': state['next_epoch'], 'no_automatic_retry': True})
        raise


def evaluate(work, receipt):
    invocation_start = time.monotonic()
    work = Path(work)
    gpu_gate(work, receipt, 'evaluate')
    complete = load(work / 'TRAINING_COMPLETE.json')
    assert complete['identity'] == training_identity(work)
    state, ref = committed(work, 'best_mae')
    assert ref == complete['selected'] and state['best_mae'] == complete['selection']
    seed(42)
    runtime_flags()
    model_module, _, decoder = science(work)
    model = torch.nn.DataParallel(model_module.RACnet(4, 10, 64).to('cuda'))
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    rows = load(work / 'inputs/prediction_inputs.json')
    dataset = PredictionInputs(rows)
    binding = {'identity': complete['identity'], 'checkpoint_sha256': ref['sha256'],
               'prediction_inputs_sha256': sha(work / 'inputs/prediction_inputs.json')}
    outputs, reused_keys = [], set()
    with torch.no_grad():
        for i, r in enumerate(rows):
            storage_guard(work)
            dest = work / 'predictions' / (r['key'].replace('/', '__') + '.npy')
            marker = Path(str(dest) + '.json')
            if marker.exists():
                result = load(marker)
                assert result['binding'] == binding and result['key'] == r['key']
                assert sha(dest) == result['probability_sha256']
                outputs.append(result)
                reused_keys.add(r['key'])
                continue
            start = time.monotonic()
            x, key = dataset[i]
            mask = torch.ones(1, len(x), device='cuda', dtype=torch.bool)
            tsm_mask = mask[:, :, None] & mask[:, None, :]
            torch.cuda.synchronize()
            infer_start = time.monotonic()
            output, _, _ = model(x[None].to('cuda'), [tsm_mask, mask], 'H-norm', 'None')
            torch.cuda.synchronize()
            infer_seconds = time.monotonic() - infer_start
            assert torch.isfinite(output).all().item(), 'nonfinite model probability'
            count = decoder.count_peaks_scaling(output, pro=0.2).item()
            prob = output[0].detach().cpu().numpy()
            atomic(dest, lambda f: np.save(f, prob, allow_pickle=False))
            result = {'key': key, 'population': r['population'], 'count': count, 'constant': bool(np.ptp(prob) == 0),
                      'nonfinite': False, 'probability_path': str(dest), 'probability_sha256': sha(dest),
                      'input_sha256': r['sha256'], 'binding': binding, 'infer_seconds': infer_seconds,
                      'total_seconds': time.monotonic()-start}
            savejson(marker, result)
            outputs.append(result)
    assert len(outputs) == 357
    savejson(work / 'PREDICTIONS_357.json', outputs)
    # Labels become accessible only after every prediction has been persisted.
    score(work, outputs, binding, invocation_start, reused_keys)


def score(work, outputs, binding, invocation_start, reused_keys):
    work = Path(work)
    metric_path = ROOT / 'source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py'
    assert sha(metric_path) == contract()['evaluation']['common_metric_sha256']
    metric = module('temporac_common_metrics', metric_path)
    labels = {r['key']: r for r in load(work / 'inputs/score_labels.json')}
    records = [dict(r, **labels[r['key']]) for r in outputs]
    assert len(records) == 357 and len({r['key'] for r in records}) == 357
    summaries = {}
    for pop in ('rc_test', 'ucf_test', 'quva_test'):
        rows = [r for r in records if r['population'] == pop]
        pred = np.asarray([r['count'] for r in rows])
        target = np.asarray([r['scalar'] for r in rows])
        native = np.asarray([r['native_gt'] for r in rows])
        # Frozen scorer API is bound and inspected by CPU_CHECK before launch.
        summaries[pop] = {'common': metric.count_metrics(pred, target),
                          'native_all_NMAE': float(np.mean(abs(pred-native)/(native+0.1))),
                          'native_all_OBO': float(np.mean(abs(pred-native) <= 1)),
                          'native_protocol': 'official decoder transfer' if pop == 'quva_test' else ('ROI-adapted' if pop == 'ucf_test' else 'native start GT')}
    savejson(work / 'SCORED_357.json', records)
    timing = timing_summary(outputs, time.monotonic()-invocation_start, reused_keys,
                            total_field='total_seconds', inference_field='infer_seconds')
    savejson(work / 'METRICS_357.json', {'binding': binding, 'populations': summaries,
             'zero_rows': [r for r in records if r['scalar'] == 0], 'scorer_sha256': sha(metric_path), 'timing': timing})
    savejson(work / 'COMPLETE.json', {'status': 'COMPLETE', 'predictions': 357,
             'prediction_sha256': sha(work / 'PREDICTIONS_357.json'), 'score_sha256': sha(work / 'SCORED_357.json'),
             'metric_sha256': sha(work / 'METRICS_357.json'), 'binding': binding, 'timing': timing})
