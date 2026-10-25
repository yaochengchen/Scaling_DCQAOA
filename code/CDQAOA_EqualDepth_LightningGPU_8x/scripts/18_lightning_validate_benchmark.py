"""GPU preflight: Lightning state / energy / adjoint gradient vs the NumPy reference. Exits non-zero on mismatch.

    python scripts/18_lightning_validate_benchmark.py --gpu 0 --out results/preflight.json
    python scripts/18_lightning_validate_benchmark.py --gpu 0 --benchmark-n 18 20 22 24   # optional timings
"""
import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="lightning.gpu", choices=["lightning.gpu", "lightning.qubit"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="complex128", choices=["complex128", "complex64"])
    ap.add_argument("--benchmark-n", type=int, nargs="*", default=[])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/preflight_validation.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from cdtransfer.instances import make_instance, dump_json
    from cdtransfer.simulator import StatevectorSimulator, Angles
    from cdtransfer.lightning import LightningSimulator, LightningTapeSimulator
    tol = 2e-7 if args.dtype == "complex128" else 3e-3
    checks = []
    for n in (8, 10, 12):
        inst = make_instance(n, "er05", "lognormal", 100 + n)
        ref = StatevectorSimulator(inst)
        sim = LightningTapeSimulator(inst, device=args.device, dtype=args.dtype)      # engine used in runs
        slow = LightningSimulator(inst, device=args.device, dtype=args.dtype)         # plain QNode cross-check
        rng = np.random.default_rng(n)
        for p in (1, 2, 4):
            for cd in (False, True):
                x = rng.uniform(-.3, .3, (3 if cd else 2) * p)
                if cd:
                    x[2 * p] = 0.0          # derivative at alpha = 0 must exist
                a = Angles.unpack(x, p, cd)
                e, g = sim.value_and_grad(x, p, cd)
                fd = [(ref.evaluate(Angles.unpack(x + d, p, cd)) - ref.evaluate(Angles.unpack(x - d, p, cd))) / 2e-6
                      for d in np.eye(len(x)) * 1e-6]
                errs = {"state": float(np.max(np.abs(sim.state(a) - ref.state(a)))),
                        "energy": abs(e - ref.evaluate(a)), "forward_energy": abs(sim.evaluate(a) - ref.evaluate(a)),
                        "gradient": float(np.max(np.abs(g - np.asarray(fd)))),
                        "qnode_gradient": float(np.max(np.abs(g - slow.value_and_grad(x, p, cd)[1])))}
                if max(errs.values()) > tol:
                    raise SystemExit(f"PREFLIGHT FAILED n={n} p={p} cd={cd}: {errs}")
                checks.append({"n": n, "p": p, "cd": cd, **errs})
                print(f"PASS n={n} p={p} cd={cd} max error {max(errs.values()):.1e}", flush=True)
    timings = []
    for n in args.benchmark_n:
        for family in ("regular3", "er05"):
            inst = make_instance(n, family, "uniform", 312 + n)
            sim = LightningTapeSimulator(inst, device=args.device, dtype=args.dtype)
            for p in args.depths:
                for cd in (False, True):
                    x = Angles(np.full(p, -.1), np.full(p, .2), np.full(p, -.04)).pack(include_cd=cd)
                    sim.value_and_grad(x, p, cd)
                    t = time.perf_counter()
                    for _ in range(args.repeats):
                        sim.value_and_grad(x, p, cd)
                    timings.append({"n": n, "family": family, "m": inst.m, "p": p, "cd": cd,
                                    "energy_plus_gradient_seconds": (time.perf_counter() - t) / args.repeats})
                    print(timings[-1], flush=True)
    dump_json(args.out, {"device": args.device, "gpu": args.gpu, "dtype": args.dtype,
                         "checks": checks, "timings": timings})
    print(f"Preflight passed; saved {args.out}")


if __name__ == "__main__":
    main()
