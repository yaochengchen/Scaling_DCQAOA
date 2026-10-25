"""Quick concurrency benchmark (a few minutes): how many processes per GPU are fastest?

    python scripts/20_quick_bench.py --gpus 0,1,2,3,4,5,6,7
    python scripts/20_quick_bench.py --gpus 0,1,2,3,4,5,6,7 --n 18 20 22 24 --levels 1 2 4 8 --seconds 30
    python scripts/20_quick_bench.py --dtype complex64 --n 22 --levels 1      # single precision

No training pipeline: for each n and each c, starts c processes on every GPU; each builds one
graph (regular3 / er05 alternately, CD p=3 by default), warms up, then all processes call
energy+adjoint-gradient in a loop for --seconds at the same time (start-up excluded).
Reports total calls per second over all GPUs and the --slots-per-gpu / --task-weights to use.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _bench(gpu, k, n, p, seconds, barrier, out, device, dtype="complex128"):
    from cdtransfer.gpu_pool import die_with_parent, set_process_name
    die_with_parent()
    set_process_name("cdqaoa-worker")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    from cdtransfer.instances import make_instance
    from cdtransfer.lightning import LightningTapeSimulator
    inst = make_instance(n, ("regular3", "er05")[k % 2], "uniform", 1000 + k)
    sim = LightningTapeSimulator(inst, device=device, dtype=dtype)
    x = np.random.default_rng(k).uniform(-.3, .3, 3 * p)
    for _ in range(2):
        sim.value_and_grad(x, p, True)
    barrier.wait(timeout=900)
    t0, calls = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        sim.value_and_grad(x, p, True)
        calls += 1
    out.put((calls, time.perf_counter() - t0))


def run(gpus, n, c, p, seconds, device, dtype="complex128"):
    ctx = mp.get_context("spawn")
    total = len(gpus) * c
    barrier, out = ctx.Barrier(total), ctx.Queue()
    procs = [ctx.Process(target=_bench, args=(g, k * len(gpus) + i, n, p, seconds, barrier, out, device, dtype))
             for k in range(c) for i, g in enumerate(gpus)]
    for pr in procs:
        pr.start()
    res = [out.get(timeout=seconds + 1200) for _ in procs]
    for pr in procs:
        pr.join()
    return sum(cn / el for cn, el in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n", type=int, nargs="+", default=[18, 22])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--device", default="lightning.gpu", choices=["lightning.gpu", "lightning.qubit"])
    ap.add_argument("--dtype", default="complex128", choices=["complex128", "complex64"])
    ap.add_argument("--out", default="results/quick_bench.json")
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    rows, best = [], {}
    for n in args.n:
        rate = {}
        for c in args.levels:
            rate[c] = run(gpus, n, c, args.p, args.seconds, args.device, args.dtype)
            rows.append({"n": n, "processes_per_gpu": c, "calls_per_second_all_gpus": rate[c]})
            print(f"n={n} processes/GPU={c}: {rate[c]:.1f} energy+gradient calls/s on {len(gpus)} GPUs "
                  f"({rate[c] / rate[min(rate)]:.2f}x vs {min(rate)}/GPU)", flush=True)
        choice = min(args.levels)            # more processes only if >= 10% faster
        for c in sorted(args.levels):
            if rate[c] > 1.10 * rate[choice]:
                choice = c
        best[n] = choice
    slots = max(best.values())
    weights = {n: max(1, slots // c) for n, c in best.items()}
    line = f"--slots-per-gpu {slots}" + ("" if slots == 1 else " --task-weights " +
                                          ",".join(f"{k}:{v}" for k, v in sorted(weights.items())))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"measurements": rows, "best_processes_per_gpu": best,
                                          "options": line, "p": args.p, "seconds": args.seconds, "dtype": args.dtype}, indent=2))
    print(f"\nUse: OPTS=\"{line}\"")


if __name__ == "__main__":
    main()
