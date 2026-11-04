"""NTR1 full-native response runtime: explicit plan/validate/train/predict."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'validate', 'train', 'predict'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    from pams.temporac.native_training_runtime_v1 import execute
    try:
        return execute(json.loads(args.config.read_text(encoding='utf-8-sig')), args.command)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
