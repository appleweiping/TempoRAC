"""Collect quantitative citations from the completed frozen group, without judging support."""
import json
from pathlib import Path

R = Path('/root/autodl-tmp/TempoRAC')
D = R / 'resume20261112/group-completion'
summary = json.loads((D / 'group_summary.json').read_text())
claims = []
def add(identifier, claim, value, source, pointer):
    claims.append({'id': identifier, 'claim': claim, 'value': value, 'source': source,
                   'json_pointer': pointer, 'support_verdict': 'not_assessed_by_collector'})
source = 'resume20261112/group-completion/group_summary.json'
add('fixed_training_identity', 'This group uses immutable training source489dc77.', summary['training_commit'], source, '/training_commit')
add('fixed_manifest_identity', 'This group uses the known-source-v1 corrected training manifest.', summary['manifest_sha256'], source, '/manifest_sha256')
for seed in [0, 1, 2]:
    raw = f'runs/frozen_known_source_v1_resume20261112/full_s{seed}/complete_known_source.json'
    completion = json.loads((R / raw).read_text())
    add(f'seed{seed}_epochs', f'Frozen seed{seed} has50 completed epochs in its raw completion receipt.', completion['epochs'], raw, '/epochs')
for tag, benchmark in summary['benchmarks'].items():
    aggregate = benchmark['views']['two_view_mean']['aggregate']
    pointer = f'/benchmarks/{tag}/views/two_view_mean/aggregate'
    for metric in ['mae_p', 'obo_p']:
        for statistic in ['mean', 'sample_sd']:
            add(f'{tag}_{metric}_{statistic}', f'{tag}: fixed3seed two-view {metric} {statistic}.',
                aggregate[metric][statistic], source, pointer + f'/{metric}/{statistic}')
    add(f'{tag}_n_all', f'{tag}: all original evaluation rows retained.', aggregate['n'], source, pointer + '/n')
    add(f'{tag}_n_positive', f'{tag}: fixed primary positive-count population.', aggregate['n_pos'], source, pointer + '/n_pos')
add('all_seed_video_rows', 'The group output retains every seed/evaluation-video row, including zero and disputed inputs.',
    8280, 'resume20261112/group-completion/COMPLETION_RECEIPT.json', '/per_video_rows')
add('rc_zero_row_retained', 'The RepCount zero-count row is retained in each fixed view/seed output.',
    summary['seed_results']['full_s0']['rc_test']['views']['two_view_mean']['zero_metrics']['n'],
    source, '/seed_results/full_s0/rc_test/views/two_view_mean/zero_metrics/n')
(D / 'claims.json').write_text(json.dumps(claims, indent=2))
print(json.dumps({'claims': len(claims), 'support_verdict': 'not_assessed', 'file': str(D / 'claims.json')}))
