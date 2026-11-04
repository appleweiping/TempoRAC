"""Isolated native target diagnostics; proposed metrics, never official results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pams.temporac.native_target_diagnostics_v1 import run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    try:
        report = run(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
        print(json.dumps({'schema': report['schema'], 'output': report['receipt']['identity_rows_sha256'],
                          'official_metric_status': report['official_metric_status']}))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
