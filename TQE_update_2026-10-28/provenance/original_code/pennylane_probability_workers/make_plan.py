"""Prepare a fixed, disjoint graph-level worker plan for the active campaign."""
import json, sys
from pathlib import Path
STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY / 'code' / 'pennylane_probability_1050'))
import campaign
from core import atomic_json
OUT = STUDY / 'results' / 'pennylane_probability_1050'
b = campaign.load_manifest()
status = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
last = status.get('last')
current = last.rsplit('-p', 1)[0] if last else None
ids = [g['id'] for g in b['graphs'] if g['n'] == 22]
if current not in ids:
    raise RuntimeError(f'Cannot locate current graph in frozen order: {current}')
tail = ids[ids.index(current) + 1:]
workers = 11
assignments = [tail[i::workers] for i in range(workers)]
plan = {'campaign': b['campaign'], 'created_after': last, 'current_graph': current,
        'worker_count': workers, 'future_graph_count': len(tail),
        'assignments': assignments, 'backend': 'pennylane.lightning.qubit',
        'per_worker_openmp_threads': 1, 'LQ_ENABLE_KERNEL_OMP': 'ON'}
atomic_json(OUT / 'parallel_worker_plan.json', plan)
print(json.dumps({'current': last, 'future_graphs': len(tail),
                  'workers': workers, 'graphs_per_worker': [len(x) for x in assignments],
                  'campaign': b['campaign']}), flush=True)
