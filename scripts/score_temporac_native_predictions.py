"""CPU-only isolated scorer for complete NTR1 development predictions."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('native_prediction_score_v1',
    ROOT / 'src/pams/temporac/native_prediction_score_v1.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = scorer.run(scorer.read_json(args.config))
        print(json.dumps({'status': report['status'], 'full_population': report['full_population'],
                          'scientific_acceptance': report['scientific_acceptance']}))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, type(exc).__name__ + ': ' + str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
