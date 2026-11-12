"""Official TransRAC native full-run adapter; server-only entry point."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # preserve read-only inherited base/source trees

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--contract', type=Path, default=HERE/'RUN_CONTRACT.json')
    parser.add_argument('--phase', choices=['spine','primitives','prepare','cpu-preflight','train-evaluate'], required=True)
    parser.add_argument('--build-rgb-cache', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert os.name != 'nt', 'Experimental execution is server-only'
    assert Path('/root/autodl-tmp/TempoRAC') in HERE.parents
    if args.phase=='spine':
        atomic_json(HERE/'checks/spine.json', {'phase':'spine','status':'PASS','driver_sha256':digest(__file__)})
        return
    # Hide every GPU for all engineering-only phases before importing torch.
    if args.phase!='train-evaluate':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        assert args.gpu in (0,1), '--gpu must identify root-released physical lane'
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]='2'
    os.environ['PYTHONNOUSERSITE']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PIP_REQUIRE_VIRTUALENV']='true'
    if args.phase=='primitives':
        from native_engine import primitive_checks
        atomic_json(HERE/'checks/primitives.json',primitive_checks())
        return
    assert digest(args.contract)=='1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a', 'frozen contract mismatch'
    contract=json.load(open(args.contract))
    if args.phase=='prepare':
        from native_inputs import prepare_inputs
        prepare_inputs(HERE,contract,digest,atomic_json,build_cache=args.build_rgb_cache)
    elif args.phase=='cpu-preflight':
        from native_run import cpu_preflight
        cpu_preflight(HERE,contract,digest,atomic_json)
    elif args.phase=='train-evaluate':
        from native_run import train_evaluate
        train_evaluate(HERE,contract,digest,atomic_json,args)

if __name__ == '__main__':
    main()
