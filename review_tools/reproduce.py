"""Read-only data verification / figure reproduction. Never runs a quantum circuit.

Outputs go to review_generated; delivered data and figures stay unchanged.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
OUT = ROOT / 'review_generated'


def table(name):
    with (ROOT / 'analysis/outputs_unified/tables' / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def verify():
    import numpy as np
    from summarize_unified_table import collect_pairs
    from unified_variational_n16p2 import summarize_runs
    pairs = collect_pairs(ROOT)
    saved = {(r['id'], int(r['p'])): r for r in table('T1_paired_records_unified.csv')}
    assert len(pairs) == len(saved) == 32000
    largest = 0.
    for row in pairs:
        target = saved[(row['id'], row['p'])]
        for key, value in row.items():
            if isinstance(value, (int, float)):
                err = abs(float(target[key]) - value)
                largest = max(largest, err)
                assert np.isclose(float(target[key]), value, rtol=1e-12, atol=1e-12), (row['id'], key)
            else:
                assert target[key] == value, (row['id'], key)
    opt = ROOT / 'results/unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500'
    metadata = json.loads((opt/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((opt/'summary.json').read_text(encoding='utf-8'))
    counts = {}
    for method in metadata['methods']:
        traces = [json.loads((opt/'runs'/method/f'{gid}_start{start}.json').read_text(encoding='utf-8'))
                  for gid in metadata['test_graph_ids'] for start in range(metadata['starts_per_graph'])]
        assert len(traces) == 240
        rebuilt = summarize_runs(traces, metadata['budget_evaluations'])
        target = summary[method]
        assert rebuilt.keys() == target.keys()
        for key, value in rebuilt.items():
            if key in ('run_hits', 'mean_trajectory_hit') or value is None:
                assert value == target[key], (method, key)
            else:
                assert np.allclose(value, target[key], rtol=1e-12, atol=1e-12), (method, key)
        counts[method] = sum(hit is not None for hit in rebuilt['run_hits'])
    result = {'graph_depth_pairs': len(pairs), 'unique_graphs': len({r['id'] for r in pairs}),
              'max_paired_table_difference': largest, 'optimizer_runs': 720,
              'optimizer_hits': counts, 'quantum_simulations_run': 0}
    OUT.mkdir(exist_ok=True)
    (OUT/'data_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


def figures():
    import matplotlib.pyplot as plt
    import plot_quality_depth_and_size_unified_v3 as quality
    import plot_optimizer_comparison_n12p2_v3 as optimizer
    import plot_weight_distributions_unified_v2 as weight
    import summarize_unified_table as summary
    OUT.mkdir(exist_ok=True)
    quality.draw(OUT/'fig02_quality_depth_and_size_unified_v3')
    opt = ROOT/'results/unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500/summary.json'
    optimizer.draw(json.loads(opt.read_text(encoding='utf-8')), OUT/'fixed_vs_variational_unified_n12p2_optimizers_pennylane_starts20_eval500_v3')
    records = table('T1_paired_records_unified.csv')
    weight.plot(records, OUT/'fig05_weight_distributions_unified_v2')
    def typed(rows):
        for row in rows:
            for key in list(row):
                if key in ('cohort', 'family', 'weight_kind', 'id'):
                    continue
                try:
                    row[key] = int(row[key]) if key in ('p', 'n', 'pairs') else float(row[key])
                except ValueError:
                    pass
        return rows
    model = json.loads((ROOT/'exports/cd-qaoa-gpu-transfer/assets/unified_gpu_model.json').read_text(encoding='utf-8'))
    summary.FIGURES = OUT
    summary.make_figures(typed(table('T2_same_depth_unified.csv')),
                         typed(table('T4_by_family_unified.csv')),
                         typed(table('T5_by_weight_unified.csv')), model, typed(records))
    plt.close('all')
    print('Figures regenerated in', OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'figures'))
    args = parser.parse_args()
    verify() if args.action == 'verify' else figures()
