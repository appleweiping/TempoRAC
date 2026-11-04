"""Explicit NTR1 cached validation/train/predict; parent scientific plan is reused."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate-cached', 'train-cached', 'predict-cached'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dev-manifest', type=Path)
    parser.add_argument('--dev-manifest-sha256')
    args = parser.parse_args(argv)
    if bool(args.dev_manifest) != bool(args.dev_manifest_sha256):
        parser.error('--dev-manifest and --dev-manifest-sha256 must be supplied together')
    from pams.temporac.native_cached_training_runtime_v1 import execute
    ref = None if args.dev_manifest is None else {'path': str(args.dev_manifest), 'sha256': args.dev_manifest_sha256}
    try:
        return execute(json.loads(args.config.read_text(encoding='utf-8-sig')), args.command, dev_manifest=ref)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
