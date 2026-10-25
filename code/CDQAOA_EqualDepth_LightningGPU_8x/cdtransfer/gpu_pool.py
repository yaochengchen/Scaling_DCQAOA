"""Slot scheduler: several worker processes per GPU, small circuits packed, large circuits isolated.

Every GPU has `slots_per_gpu` persistent worker processes (each pinned with
CUDA_VISIBLE_DEVICES before PennyLane/CUDA is imported).  A task of size n
occupies weight(n) slots of its GPU, so e.g. with 4 slots and weights
{18:1, 20:1, 22:2, 24:4}: four n<=20 instances share a card, two n=22 share a
card, an n=24 instance has the card alone.  Tasks are dispatched heaviest
first; when the head task does not fit anywhere, the GPU with the most free
slots is reserved for it and smaller tasks back-fill the other GPUs only.

Small statevectors are limited by Python/PennyLane launch overhead rather than
by GPU memory bandwidth, so running several of them per card is what keeps all
eight cards busy.  Optionally start CUDA MPS (scripts/mps_start.sh) so kernels
of different processes on one card can overlap instead of time-slicing.
"""
from __future__ import annotations
import multiprocessing as mp
import os
import queue
import traceback


def parse_weights(text: str | None) -> dict[int, int]:
    if not text:
        return {}
    out = {}
    for part in str(text).split(","):
        if part.strip():
            n, w = part.split(":")
            out[int(n)] = int(w)
    return out


def default_weight(n: int, slots: int) -> int:
    """1 slot up to n=21, then doubling per qubit (statevector size), capped at the whole GPU."""
    return int(min(slots, max(1, 2 ** (n - 21))))


def resolve_execution(config: dict, runtime: dict | None) -> dict:
    """Merge config `execution` with command-line overrides into one plain dict."""
    runtime = runtime or {}
    ex = dict(config.get("execution", {}))
    gpus = runtime.get("gpus") or os.environ.get("CDQAOA_GPUS") or ex.get("gpus", "0")
    ex["gpus"] = [g.strip() for g in str(gpus).split(",") if g.strip()]
    ex["slots_per_gpu"] = int(runtime.get("slots_per_gpu") or ex.get("slots_per_gpu", 1))
    weights = {int(k): int(v) for k, v in (ex.get("task_weight_by_n") or {}).items()}
    weights.update(parse_weights(runtime.get("task_weights")))
    ex["task_weight_by_n"] = weights
    if not ex["gpus"] or len(ex["gpus"]) != len(set(ex["gpus"])) or ex["slots_per_gpu"] < 1:
        raise ValueError("Need unique GPU ids and slots_per_gpu >= 1")
    return ex


def task_weight(n: int, ex: dict) -> int:
    w = ex["task_weight_by_n"].get(int(n), default_weight(int(n), ex["slots_per_gpu"]))
    return int(min(max(w, 1), ex["slots_per_gpu"]))


def set_process_name(name: str):
    """Linux: set the short process name shown by `ps -e` / matched by `pkill NAME`."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(15, name.encode()[:15], 0, 0, 0)  # PR_SET_NAME
    except Exception:
        pass


def die_with_parent():
    """Linux: get SIGTERM when the launching process dies (e.g. dropped SSH session), so no
    orphaned worker keeps a GPU busy or blocks a rerun."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)   # PR_SET_PDEATHSIG
    except Exception:
        pass


def _worker(gpu, inbox, results, handler_name, context):
    die_with_parent()
    set_process_name("cdqaoa-worker")
    parent = os.getppid()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    import importlib
    module, _, name = handler_name.rpartition(":")
    handler = getattr(importlib.import_module(module or "cdtransfer.eqdepth"), name)
    while True:
        try:
            item = inbox.get(timeout=30)
        except queue.Empty:
            if os.getppid() != parent:      # launcher gone
                break
            continue
        if item is None:
            break
        tid, task = item
        try:
            results.put(("ok", tid, handler(task, context, str(gpu))))
        except Exception:
            results.put(("error", tid, traceback.format_exc()))


def run_tasks(tasks, sizes, ex, handler_name, context):
    """Yield (status, index, payload) as tasks finish. `sizes[i]` is the qubit number of task i."""
    order = sorted(range(len(tasks)), key=lambda i: -sizes[i])
    if not order:
        return
    gpus, slots = ex["gpus"], ex["slots_per_gpu"]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    workers = []      # (gpu, process, inbox)
    max_needed = min(len(tasks), len(gpus) * slots)
    for k in range(slots):
        for g in gpus:
            if len(workers) >= max_needed:
                break
            inbox = ctx.Queue()
            proc = ctx.Process(target=_worker, args=(g, inbox, results, handler_name, context), daemon=True)
            proc.start()
            workers.append((g, proc, inbox))
    idle = {g: [w for w in workers if w[0] == g] for g in gpus}
    free = {g: slots for g in gpus}
    running = {}      # task index -> (worker, weight)
    pending = list(order)
    try:
        while pending or running:
            reserved = None
            placed = True
            while placed and pending:
                placed = False
                for pos, i in enumerate(pending):
                    w = task_weight(sizes[i], ex)
                    fits = [g for g in gpus if (pos == 0 or g != reserved) and free[g] >= w and idle[g]]
                    if fits:
                        g = max(fits, key=lambda q: free[q])
                        worker = idle[g].pop()
                        free[g] -= w
                        running[i] = (worker, w)
                        worker[2].put((i, tasks[i]))
                        pending.pop(pos)
                        placed = True
                        break
                    if reserved is None and pos == 0:
                        reserved = max(gpus, key=lambda q: free[q])
            try:
                status, i, payload = results.get(timeout=60)
            except queue.Empty:
                dead = [w for w in workers if not w[1].is_alive()]
                if dead:
                    raise RuntimeError(f"{len(dead)} worker process(es) died (e.g. GPU OOM); "
                                       "rerun the same command to resume")
                continue
            worker, w = running.pop(i)
            free[worker[0]] += w
            idle[worker[0]].append(worker)
            yield status, i, payload
    finally:
        for _, proc, inbox in workers:
            try:
                inbox.put(None)
            except Exception:
                pass
        for _, proc, _ in workers:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def _sleep_probe(task, ctx, gpu):
    """Test helper: report which GPU ran the task and when."""
    import time
    t0 = time.time()
    time.sleep(task["seconds"])
    return {"gpu": gpu, "n": task["n"], "start": t0, "end": time.time()}
