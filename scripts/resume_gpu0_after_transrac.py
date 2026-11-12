"""One-shot dependency gate for the frozen GPU0 lane; default explicit check, no watcher."""
import argparse, contextlib, datetime, fcntl, hashlib, importlib.util, json, math, os
from pathlib import Path
import subprocess, sys
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True
R = Path('/root/autodl-tmp/TempoRAC')
A = R/'resume20261112/ablation-continuation'
T = R/'baselines/transrac_native_v1'
D = R/'resume20261112/live-continuation/ablation-successor'
SOURCE = R/'source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb'
OUTPUTS = R/'runs/frozen_known_source_v1_ablation20261112'
TRAIN = '489dc77b91a21d21215784ff49961edb20dab4fb'
CONTRACT = 'dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167'
WORKER = 'd23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144'
TC = '1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a'
HOLD_SHA = '69fe13edb64e15ece22f8987318fe4a6b0a8879204cc261f1f7ba7206d8998e2'
HELD_SHA = 'bd4665f016c009b7488805342982520f9754b4ddf0e6b8a1ce4d1e3291edd121'
LAUNCH_SHA = '9fded1a672e4e92509bcf4c888614a5eae5577944ea00a4a5afd7ee45ce03e11'
PREFLIGHT_SHA = 'ca79905b34591587ec9899660140550dd6b18e815942e8f1c87587875777955d'
REVIEW_SHA = 'e547b9493f5657df140e7b8d08047e3f5cd8f90103559b3fc53c370f83736a14'
RESIDUAL = ['wo_pyramid_s2', 'wo_depth_s0', 'wo_depth_s1', 'wo_depth_s2',
            'wo_balance_s0', 'wo_balance_s1', 'wo_balance_s2',
            'wo_joint_rc_s0', 'wo_joint_rc_s1', 'wo_joint_rc_s2']
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(ok, reason):
    if not ok: raise RuntimeError(reason)


def out(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), creationflags=FLAGS).strip()


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def live(pid): return Path('/proc', str(pid)).exists()


def atomic(path, value):
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def sync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def assess(stack):
    require(os.name != 'nt' and R.resolve() == R, 'Server-only verified TempoRAC root required')
    for path in [A, T, SOURCE, OUTPUTS]:
        require(path.resolve().is_relative_to(R), 'Project path escaped root')
    require(not (D/'one_shot.json').exists(), 'Successor claim exists: inspect it; no automatic repeat')
    for name, digest in [('ablation_worker.py', WORKER), ('RUN_CONTRACT.json', CONTRACT),
                         ('HOLD_AFTER_CURRENT_GPU0', HOLD_SHA), ('lane_gpu0.json', HELD_SHA),
                         ('launch_gpu0.json', LAUNCH_SHA)]:
        require(sha(A/name) == digest, 'Frozen GPU0 control changed: '+name)
    require(out(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']) == TRAIN, 'Training source commit changed')
    require(not out(['git', '-C', str(SOURCE), 'status', '--porcelain']), 'Frozen training tree is dirty')
    require(sha(SOURCE/'temporac_cvpr/train.py') == '33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4', 'Training runtime changed')
    contract = read(A/'RUN_CONTRACT.json'); held = read(A/'lane_gpu0.json')
    launch = read(A/'launch_gpu0.json'); hold = read(A/'HOLD_AFTER_CURRENT_GPU0')
    require(held['status'] == 'held_after_complete' and held['remaining'] == RESIDUAL and held['gpu'] == 0, 'Not the frozen ten-arm hold')
    require(held['contract_sha256'] == CONTRACT and held['worker_sha256'] == WORKER and held['source_commit'] == TRAIN, 'Held provenance mismatch')
    require(hold['baseline_contract_sha256'] == TC and hold['training_source_unchanged'] == TRAIN, 'Handoff dependency mismatch')
    require(not live(int((A/'worker_gpu0.pid').read_text())), 'GPU0 ablation worker still exists')
    spec_path = Path(contract['frozen_spec']['path']); manifest = Path(contract['manifest']['path'])
    require(sha(spec_path) == contract['frozen_spec']['sha256'] and sha(manifest) == contract['manifest']['sha256'], 'Frozen spec/manifest changed')
    module_spec = importlib.util.spec_from_file_location('frozen_gpu0_ablation_worker', A/'ablation_worker.py')
    worker = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(worker)
    plans = worker.checked_plans(contract, read(spec_path)); lane = contract['lanes']['gpu0']
    require(lane == launch['lane'] and lane[-10:] == RESIDUAL, 'Frozen lane changed')
    command = [str(R/'env/bin/python'), '-u', '-B', str(A/'ablation_worker.py'), '--gpu', '0',
               '--source', str(SOURCE), '--commit', TRAIN, '--contract', str(A/'RUN_CONTRACT.json')]
    require(launch['command'] == command, 'Native worker command changed')

    def complete(name):
        path = OUTPUTS/name/'complete_known_source.json'
        if not path.exists(): return False
        got = read(path); plan = plans[name]; r = plan['run']
        expected = dict(run=name, source=str(SOURCE), source_commit=TRAIN, worker_sha256=WORKER,
            contract_sha256=CONTRACT, spec_sha256=contract['frozen_spec']['sha256'],
            manifest_sha256=contract['manifest']['sha256'], seed=r['seed'], cfg=r['cfg'],
            cfg_sha256=worker.cfg_sha(r['cfg']), features=r['features'], train_split=r['train_split'],
            val_split=r['val_split'], test_split=r['test_split'], evaluations=plan['evaluations'])
        require(got['provenance'] == expected and got['epochs'] == r['cfg']['epochs'], 'Completed arm provenance/epoch mismatch: '+name)
        required = {'config.json', 'log.jsonl', 'result.json'}
        for job in plan['evaluations']: required.update([job['checkpoint'], 'eval_'+job['tag']+'.json'])
        require(set(got['file_sha256']) == required, 'Incomplete arm output digest set: '+name)
        for filename, digest in got['file_sha256'].items():
            require(sha(OUTPUTS/name/filename) == digest, 'Completed arm output changed: '+name+'/'+filename)
        return True

    pending = worker.validate_residual(lane, held['remaining'], complete)
    require(pending == RESIDUAL, 'Held residual advanced outside this frozen one-shot; root must reconcile')
    require(all(not (OUTPUTS/name).exists() for name in pending), 'Pending run directory exists: no automatic failed/partial experiment restart')
    events = [json.loads(x) for x in (A/'worker_gpu0.events.jsonl').read_text().splitlines() if x.strip()]
    require(events[-1]['status'] == 'held_after_complete' and events[-1]['remaining'] == RESIDUAL, 'GPU0 history no longer ends at frozen hold')
    require(sha(T/'RUN_CONTRACT.json') == TC and sha(T/'checks/CPU_PREFLIGHT.json') == PREFLIGHT_SHA and sha(T/'ENGINEERING_REVIEW.json') == REVIEW_SHA, 'TransRAC approved bindings changed')
    tl = read(T/'LAUNCH_RECEIPT_AFTER_IDENTITY_FIX.json'); pre = read(T/'checks/CPU_PREFLIGHT.json'); review = read(T/'ENGINEERING_REVIEW.json')
    identity = pre['identity']
    require(pre['status'] == 'PASS' and review['complete'] and review['code'] == identity['code'] == tl['code'] and identity['contract_sha256'] == TC, 'TransRAC preflight/review/launch mismatch')
    for name, digest in identity['code'].items(): require(sha(T/name) == digest, 'TransRAC runtime changed: '+name)
    transrac_contract = read(T/'RUN_CONTRACT.json')
    official = Path(transrac_contract['identity']['source_path'])
    require(official.resolve().is_relative_to(R) and out(['git', '-C', str(official), 'rev-parse', 'HEAD']) == identity['official_source']['commit'], 'TransRAC official source identity changed')
    for name, digest in identity['official_source']['files'].items():
        require(sha(official/name) == digest, 'TransRAC official source file changed: '+name)
    require(sha(T/'inputs/IDENTITIES.json') == identity['input_receipt_sha256'], 'TransRAC prepared input identity changed')
    scorer = transrac_contract['final_evaluation']['canonical_scorer']
    require(sha(scorer['server_path']) == scorer['sha256'] == identity['canonical_scorer_sha256'], 'Frozen scorer changed')
    proof = dict(utc=utc(), completed_prefix=lane[:-len(pending)], pending=pending, ablation_contract_sha256=CONTRACT,
                 worker_sha256=WORKER, hold_sha256=HOLD_SHA, held_lane_sha256=HELD_SHA,
                 events_sha256=sha(A/'worker_gpu0.events.jsonl'), command=command+['--resume'])
    if live(tl['pid']):
        cmdline = (Path('/proc')/str(tl['pid'])/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        require(cmdline == tl['command'], 'TransRAC PID reused or command changed')
        return dict(proof, status='WAIT', reason='TransRAC full200/full357 process is still alive', dependency_pid=tl['pid'])
    lock = stack.enter_context((T/'run.lock').open('rb'))
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return dict(proof, status='WAIT', reason='TransRAC run lock remains held')
    require((T/'COMPLETION.json').is_file(), 'TransRAC exited without full completion; do not restart it here')
    done = read(T/'COMPLETION.json')
    require(done['status'] == 'COMPLETE' and done['completed_epochs'] == 200 and done['final_population_total'] == 357, 'Incomplete TransRAC full run')
    require(done['logical_optimizer_opportunities'] == 4800 and done['successful_optimizer_updates']+done['AMP_overflow_skips'] == 4800, 'TransRAC optimizer accounting incomplete')
    require(done['identity'] == identity, 'TransRAC completion identity mismatch')
    attempt = Path(done['attempt']); require(attempt.resolve().is_relative_to(T/'attempts'), 'Completion attempt escaped project')
    require(not (attempt/'FAILURE.json').exists(), 'Completion attempt has failure receipt')
    start = read(attempt/'START.json'); require(start['identity'] == identity and start['resource_gate']['physical_gpu'] == 0, 'Completion start identity/GPU mismatch')
    require(not live(start['pid']), 'TransRAC completion writer has not exited')
    require(sha(T/'checkpoints/best.pt') == done['best_checkpoint_sha256'] and sha(T/'checkpoints/latest.pt') == done['latest_sha256'], 'Completed checkpoint bytes changed')
    require(set(done['candidate_identities']) == {f'epoch{e:03d}.json' for e in range(51,199,3)}, 'Validation candidates incomplete')
    for name, digest in done['candidate_identities'].items(): require(sha(T/'candidates'/name) == digest, 'Validation candidate changed')
    inputs = R/'baselines/transrac-native-assets20261112/inputs/manifest.json'
    require(sha(inputs) == '61cbbd8c98c7a9764284dd2f032d85d2cfb9403fedabf695fe90271f31fa2333', 'TransRAC input manifest changed')
    rows = read(inputs); counts = {'rc_test':152, 'ucf_test':105, 'test':100}; verified = 0
    require(set(done['final']) == set(counts), 'TransRAC final population set mismatch')
    for pop, n in counts.items():
        path = T/f'final/{pop}_predictions.json'; item = done['final'][pop]
        require(item['n'] == n and sha(path) == item['predictions_sha256'], 'Prediction count/hash mismatch: '+pop)
        aggregate = read(path); expected = {r['population']+'/'+r['video_id']:r for r in rows if r['population'] == pop}
        predictions = aggregate['rows']
        require(aggregate['identity'] == identity and aggregate['best'] == done['best'] and aggregate['checkpoint_sha256'] == done['best_checkpoint_sha256'], 'Prediction identity mismatch')
        require(len(predictions) == len(expected) == n and {r['key'] for r in predictions} == set(expected), 'Prediction membership mismatch')
        require(read(T/f'final/{pop}_metrics.json') == item['metrics'], 'Final metric receipt mismatch')
        for row in predictions:
            ref = expected[row['key']]
            require(row['source_video_sha256'] == ref['source_video_sha256'] and row['scalar_gt'] == ref['count'] and row['frame_range'] == ref['frame_range'], 'Prediction/input identity mismatch')
            require(row['checkpoint_sha256'] == done['best_checkpoint_sha256'] and row['selected_epoch'] == done['best']['epoch'] and row['input_receipt_sha256'] == identity['input_receipt_sha256'], 'Prediction checkpoint identity mismatch')
            require(math.isfinite(row['continuous_count']) and len(row['density']) == 64 and all(math.isfinite(x) for x in row['density']), 'Invalid saved prediction')
            member = read(T/'final/records'/pop/(ref['video_id']+'.json'))
            require(member['row'] == row and member['identity'] == identity and member['best'] == done['best'] and member['checkpoint_sha256'] == done['best_checkpoint_sha256'], 'Per-video final record mismatch')
            verified += 1
    require(verified == 357, 'Not full357')
    gpu = out(['nvidia-smi','-i','0','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits']).split(',')
    require(len(gpu) == 3 and int(gpu[0]) == 0 and gpu[1].strip() == start['resource_gate']['uuid'], 'Physical GPU0 identity changed')
    xml = ET.fromstring(out(['nvidia-smi','-i','0','-q','-x']))
    require(len(xml.findall('gpu')) == 1, 'Ambiguous GPU0 process query')
    processes = xml.findall('.//processes/process_info')
    if int(gpu[2]) >= 500 or processes:
        return dict(proof, status='WAIT', reason='GPU0 is occupied; no process is signalled', gpu0_memory_mib=int(gpu[2]), gpu0_process_count=len(processes))
    return dict(proof, status='READY', reason='Full200/full357 complete, writers exited, exact residual and GPU0 idle',
                transrac_completion_sha256=sha(T/'COMPLETION.json'), prediction_members_verified=verified,
                gpu0_uuid=gpu[1].strip(), gpu0_memory_mib=int(gpu[2]), gpu0_process_count=0)


def main():
    parser = argparse.ArgumentParser(__doc__); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true'); mode.add_argument('--execute', action='store_true'); args = parser.parse_args()
    claimed = False; state = {}; process = None
    try:
        with contextlib.ExitStack() as stack:
            if args.execute:
                require(os.name != 'nt' and R.resolve() == R, 'Server-only')
                D.mkdir(parents=True, exist_ok=True)
                own = stack.enter_context((D/'successor.lock').open('a'))
                fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
            proof = assess(stack)
            if args.check or proof['status'] != 'READY': print(json.dumps(proof)); return 0
            state = dict(proof, status='CLAIMED', entry_sha256=sha(__file__), no_automatic_retry=True)
            with (D/'one_shot.json').open('x') as f:
                json.dump(state, f, indent=2); f.flush(); os.fsync(f.fileno())
            sync_dir(D); claimed = True
            archive = D/'HOLD_ORIGINAL_GPU0.archived.json'; require(not archive.exists(), 'HOLD archive already exists')
            os.rename(A/'HOLD_AFTER_CURRENT_GPU0', archive); sync_dir(A); sync_dir(D)
            require(sha(archive) == HOLD_SHA, 'Archived HOLD changed')
            state.update(status='HOLD_ARCHIVED', hold_archive=str(archive)); atomic(D/'one_shot.json', state)
            with (D/'handoff.events.jsonl').open('x') as event:
                event.write(json.dumps(dict(utc=utc(), event='original_hold_archived', sha256=HOLD_SHA, old_events_sha256=proof['events_sha256']))+'\n'); event.flush(); os.fsync(event.fileno())
            state.update(status='SPAWN_COMMITTED_NO_REPEAT'); atomic(D/'one_shot.json', state)
            log = stack.enter_context((D/'worker_gpu0.successor.stdout.log').open('x'))
            process = subprocess.Popen(proof['command'], cwd=SOURCE, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1'),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, creationflags=FLAGS)
            state.update(status='WORKER_SPAWNED', worker_pid=process.pid); atomic(D/'one_shot.json', state)
        # One ordinary child wait, no dependency polling, retry or scheduler.
        code = process.wait(); state.update(status='WORKER_EXITED_OK' if code == 0 else 'FAILED_NO_RETRY', worker_returncode=code, finished_utc=utc())
        atomic(D/'one_shot.json', state); print(json.dumps(state)); return code
    except Exception as error:
        failure = dict(state, status='BLOCKED' if not claimed else 'FAILED_MANUAL_RECOVERY_ONLY', error_type=type(error).__name__, error=str(error), utc=utc(), no_automatic_retry=True)
        if claimed:
            try: atomic(D/'one_shot.json', failure)
            except Exception: pass
        print(json.dumps(failure)); return 2


if __name__ == '__main__': raise SystemExit(main())
