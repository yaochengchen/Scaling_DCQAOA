"""Measure how many concurrent processes per GPU give the highest throughput for each n.

    python scripts/19_tune_gpu_workers.py --gpus 0,1,2,3,4,5,6,7 --out results/tune_slots

For every n and concurrency c (--levels), the same equal-depth workload is trained with
c processes per GPU (chain p=1,2,3, 1 restart, 8*max(levels) instances per family).
Throughput is measured in the STEADY STATE: process start-up (spawn + PennyLane/CUDA
initialisation of up to 8*c processes) is excluded by timing from the moment every
worker has finished its first optimization to the end of the run.
An interrupted measurement is deleted and redone on the next call; finished ones are kept.
Run once more inside `source scripts/mps_start.sh` (with another --out) to compare with CUDA MPS.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdtransfer.gpu_pool import die_with_parent


def steady_rate(recs, workers):
    t = sorted(r["finished_unix"] for r in recs)
    k = min(workers, len(t) - 2)
    if k < 0 or t[-1] <= t[k]:
        return None
    return (len(t) - 1 - k) / (t[-1] - t[k]) * 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n", type=int, nargs="+", default=[18, 22])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--out", default="results/tune_slots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ngpu = len(args.gpus.split(","))
    per_family = ngpu * max(args.levels)
    rows, best = [], {}
    for n in args.n:
        tput = {}
        for c in args.levels:
            dest = out / f"n{n}_c{c}"
            result = dest / "TUNE_RESULT.json"
            if not result.exists():
                if dest.exists():
                    shutil.rmtree(dest)          # interrupted earlier: redo cleanly
                cfg = {"comparison": "same_depth", "seed": 4242, "depths": [1, 2, 3], "shots": 0,
                       "exact_max_n": 0,
                       "execution": {"backend": "lightning.gpu", "dtype": "complex128", "gpus": args.gpus,
                                     "slots_per_gpu": c, "task_weight_by_n": {str(n): 1}},
                       "optimization": {"restarts": 1, "maxiter": args.maxiter},
                       "datasets": {s: {"n": [n], "families": ["regular3", "er05"], "weights": ["uniform"],
                                        "samples_per_cell": per_family if s == "train" else 1}
                                    for s in ("train", "val", "test")}}
                path = out / f"n{n}_c{c}.json"
                path.write_text(json.dumps(cfg))
                subprocess.run([sys.executable, "-m", "cdtransfer.cli", "train", "--config", str(path),
                                "--out", str(dest)], check=True, preexec_fn=die_with_parent)
                recs = [json.loads(l) for l in (dest / "offline_optima.jsonl").open() if l.strip()]
                rate = steady_rate(recs, ngpu * c)
                result.write_text(json.dumps({"optimizations": len(recs), "steady_rate_per_hour": rate}))
            res = json.loads(result.read_text())
            tput[c] = res["steady_rate_per_hour"] or 0.0
            rows.append({"n": n, "processes_per_gpu": c, **res})
            print(f"n={n} processes/GPU={c}: {tput[c]:.0f} optimizations/hour (steady state)", flush=True)
        choice = min(args.levels)          # more processes only if >= 10% faster
        for c in sorted(args.levels):
            if tput[c] > 1.10 * tput[choice]:
                choice = c
        best[n] = choice
    slots = max(best.values())
    weights = {n: max(1, slots // c) for n, c in best.items()}
    rec = {"measurements": rows, "best_processes_per_gpu": best, "slots_per_gpu": slots,
           "task_weight_by_n": {str(k): v for k, v in weights.items()},
           "note": "sizes not measured (e.g. n=24) default to a whole GPU when slots_per_gpu > 1",
           "command_line": f"--slots-per-gpu {slots}" + ("" if slots == 1 else " --task-weights " +
                           ",".join(f"{k}:{v}" for k, v in sorted(weights.items())))}
    (out / "recommendation.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\nUse: {rec['command_line']}")


if __name__ == "__main__":
    main()
