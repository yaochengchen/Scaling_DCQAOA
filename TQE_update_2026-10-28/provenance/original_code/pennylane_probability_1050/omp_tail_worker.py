"""Run a disjoint future tail of the active campaign with Lightning OpenMP."""
import os
os.environ['LQ_ENABLE_KERNEL_OMP'] = 'ON'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
import argparse
import json
import os as _os
from pathlib import Path
import msvcrt
import core
import campaign

OUT = campaign.OUT
LOCK_PATH = OUT / 'omp_worker.lock'

class WorkerLock:
    def __enter__(self):
        self.f = open(LOCK_PATH, 'a+b')
        self.f.seek(0)
        try:
            msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.f.close()
            raise RuntimeError('An OpenMP tail worker is already active') from exc
        return self
    def __exit__(self, *exc):
        self.f.seek(0)
        msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        self.f.close()


def main():
    with WorkerLock():
        b = campaign.load_manifest()
        status = json.loads((OUT / 'status.json').read_text(encoding='utf-8'))
        last = status.get('last')
        current_id = last.rsplit('-p', 1)[0] if last else None
        graph_ids = [g['id'] for g in b['graphs'] if g['n'] == 22]
        if current_id not in graph_ids:
            raise RuntimeError(f'Cannot locate active graph in frozen order: {current_id}')
        start = graph_ids.index(current_id) + 1
        tail_ids = graph_ids[start:]
        if not tail_ids:
            print('No future 22-qubit graphs remain; main campaign continues.', flush=True)
            return

        original_atomic_json = core.atomic_json
        def tagged_atomic_json(path, obj):
            path = Path(path)
            if path.parent.resolve() == (OUT / 'records').resolve() and 'methods' in obj:
                record = dict(obj)
                record.pop('content_sha256', None)
                record['execution_config'] = {
                    'worker': 'lightning_openmp_tail',
                    'pid': _os.getpid(),
                    'LQ_ENABLE_KERNEL_OMP': 'ON',
                    'OMP_NUM_THREADS': '4',
                }
                record['content_sha256'] = core.object_digest(record)
                return original_atomic_json(path, record)
            return original_atomic_json(path, obj)

        core.atomic_json = tagged_atomic_json
        campaign.status = lambda *args, **kwargs: None
        print(json.dumps({
            'event': 'worker_started', 'pid': _os.getpid(),
            'campaign': b['campaign'], 'current_graph': current_id,
            'future_graph_count': len(tail_ids), 'backend': core.BACKEND,
            'LQ_ENABLE_KERNEL_OMP': os.environ['LQ_ENABLE_KERNEL_OMP'],
            'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
        }), flush=True)
        args = argparse.Namespace(sizes=[22], depths=list(campaign.DEPTHS),
                                  graph_ids=tail_ids, max_records=None)
        campaign.run(b, args)
        print(json.dumps({'event': 'worker_finished', 'pid': _os.getpid(),
                          'campaign': b['campaign']}), flush=True)

if __name__ == '__main__':
    main()
