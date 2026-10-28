"""Single-thread PennyLane worker; run concurrently on a disjoint graph shard."""
import os, sys, json, argparse, msvcrt
from pathlib import Path
os.environ['LQ_ENABLE_KERNEL_OMP'] = 'ON'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
PACKAGE = STUDY / 'code' / 'pennylane_probability_1050'
sys.path.insert(0, str(PACKAGE))

# Load a local core copy that requests one OpenMP thread, while retaining the
# frozen core/campaign source files and campaign identity used by the main run.
import importlib.util
spec = importlib.util.spec_from_file_location('core', HERE / 'core.py')
core = importlib.util.module_from_spec(spec)
sys.modules['core'] = core
spec.loader.exec_module(core)
import campaign

OUT = campaign.OUT
WORKER_ID = int(sys.argv[1])
PLAN_PATH = OUT / 'parallel_worker_plan.json'
LOCK_DIR = OUT / 'parallel_worker_locks'
LOCK_DIR.mkdir(parents=True, exist_ok=True)
LOCK_PATH = LOCK_DIR / f'worker_{WORKER_ID:02d}.lock'

class WorkerLock:
    def __enter__(self):
        self.f = open(LOCK_PATH, 'a+b')
        if self.f.seek(0, os.SEEK_END) == 0:
            self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try: msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.f.close(); raise RuntimeError(f'Worker {WORKER_ID} is already active') from exc
        return self
    def __exit__(self, *exc):
        self.f.seek(0); msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1); self.f.close()

def main():
    with WorkerLock():
        b = campaign.load_manifest()
        plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        if plan['campaign'] != b['campaign'] or plan['worker_count'] != 11:
            raise RuntimeError('Worker plan does not match the frozen campaign')
        ids = plan['assignments'][WORKER_ID]
        original_atomic_json = core.atomic_json
        def tagged_atomic_json(path, obj):
            path = Path(path)
            if path.parent.resolve() == (OUT / 'records').resolve() and 'methods' in obj:
                record = dict(obj)
                record.pop('content_sha256', None)
                record['threads'] = 1
                record['execution_config'] = {
                    'worker': 'graph_parallel_single_thread', 'worker_id': WORKER_ID,
                    'pid': os.getpid(), 'worker_count': 11,
                    'LQ_ENABLE_KERNEL_OMP': 'ON', 'OMP_NUM_THREADS': '1',
                }
                record['content_sha256'] = core.object_digest(record)
                return original_atomic_json(path, record)
            return original_atomic_json(path, obj)
        core.atomic_json = tagged_atomic_json
        campaign.status = lambda *args, **kwargs: None
        print(json.dumps({'event':'worker_started','worker_id':WORKER_ID,
                          'pid':os.getpid(),'graphs':len(ids),'campaign':b['campaign'],
                          'backend':core.BACKEND,'OMP_NUM_THREADS':'1',
                          'LQ_ENABLE_KERNEL_OMP':'ON'}),flush=True)
        args = argparse.Namespace(sizes=[22], depths=list(campaign.DEPTHS),
                                  graph_ids=ids, max_records=None)
        campaign.run(b, args)
        print(json.dumps({'event':'worker_finished','worker_id':WORKER_ID,
                          'pid':os.getpid(),'campaign':b['campaign']}),flush=True)

if __name__ == '__main__': main()

