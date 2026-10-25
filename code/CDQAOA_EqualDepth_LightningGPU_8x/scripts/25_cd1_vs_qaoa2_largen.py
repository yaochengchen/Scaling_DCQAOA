"""Quick look beyond n=22: CD-QAOA p=1 versus QAOA p=2 (equal two-qubit count after level-3 compilation).

Small side check for the paper, not part of the main protocol.  For every new instance:
  QAOA p=1 : 1 start (ramp)                       -> only used as warm start
  CD   p=1 : 2 starts (from QAOA p=1 optimum with alpha=0, ramp)
  QAOA p=2 : 2 starts (INTERP of QAOA p=1, ramp)
(the same structured starts that won most often in the main runs; no random starts).
L-BFGS-B with adjoint gradients in complex64, final energies re-evaluated in complex128.
The frozen fixed rule of the large run (FROZEN_MODEL.json) is also evaluated on the same graphs.
gain = (E_QAOA,p=2 - E_CD,p=1) / sum|J|  (positive = CD p=1 better).

Default: n = 20 22 24 26 28, all 10 families, new seeds; 10 graphs per family for n <= 26
(5 unit + 5 lognormal weights) and 4 per family for n = 28 (--per-family-override 28:4).
Expected wall time on 8 x RTX PRO 6000 (one process per GPU): about 2-2.5 h in total,
dominated by n=26 and n=28 (n=20/22 take minutes and anchor the trend under the same reduced
protocol).  Tasks run in ascending n, so --summary shows complete smaller sizes early.

    cd code/CDQAOA_EqualDepth_LightningGPU_8x        # from the repository root
    python scripts/25_cd1_vs_qaoa2_largen.py --gpus 0,1,2,3,4,5,6,7
    python scripts/25_cd1_vs_qaoa2_largen.py --summary          # any time, also while running

Results are appended per instance to results/cd1_vs_qaoa2_largen/records.jsonl; rerunning the
same command skips finished instances, so it can be stopped and resumed at any time.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAMILIES = ["tree", "ring", "regular3", "regular4", "regular5", "regular6", "er3", "er05", "community", "bipartite"]
WEIGHTS = ["unit", "lognormal"]


def parse_override(text):
    out = {}
    for item in (text or "").split(","):
        if item.strip():
            n, k = item.split(":")
            out[int(n)] = int(k)
    return out


def tasks_for(ns, per_family, seed, override=None):
    override = override or {}
    out = []
    for n in sorted(ns):
        for fam in FAMILIES:
            for j in range(override.get(n, per_family)):
                w = WEIGHTS[j % len(WEIGHTS)]
                tok = f"{seed}|largen|{n}|{fam}|{w}|{j}".encode()
                s = int.from_bytes(hashlib.sha256(tok).digest()[:4], "big")
                out.append({"id": f"largen-n{n}-{fam}-{w}-{j:03d}", "n": n, "family": fam, "weight_kind": w, "seed": s})
    return out


def _worker(gpu_queue, task_queue, result_queue, args):
    gpu = gpu_queue.get()
    if args["backend"] == "lightning.gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    from cdtransfer.instances import make_instance
    from cdtransfer.lightning import make_simulator
    from cdtransfer.optimization import ramp_start, interp_start, run_lbfgs
    from cdtransfer.transfer import predict
    model = json.load(open(args["model"]))
    settings = {"backend": args["backend"], "dtype": args["dtype"]}
    ref_settings = {"backend": args["backend"], "dtype": "complex128"}
    while True:
        t = task_queue.get()
        if t is None:
            break
        t0 = time.time()
        try:
            inst = make_instance(t["n"], t["family"], t["weight_kind"], t["seed"], split="largen")
            z = float(np.sum(np.abs(inst.weights)))
            sim = make_simulator(inst, settings)
            ref = make_simulator(inst, ref_settings) if args["dtype"] != "complex128" else sim
            kw = dict(maxiter=args["maxiter"], ftol=args["ftol"], gtol=args["gtol"])

            def opt(p, cd, x0):
                r = run_lbfgs(inst, p, cd, x0, sim, **kw)
                e = ref.evaluate(r.angles)
                x = np.concatenate([r.angles.gamma, r.angles.beta] + ([r.angles.alpha] if cd else []))
                return e, x, r.evaluations, bool(r.success)

            calls = 0
            e_q1, x_q1, c, _ = opt(1, False, ramp_start(inst, 1, False)); calls += c
            cd_runs = [opt(1, True, np.concatenate([x_q1, np.zeros(1)])), opt(1, True, ramp_start(inst, 1, True))]
            q2_runs = [opt(2, False, interp_start(x_q1, 1, 2, False)), opt(2, False, ramp_start(inst, 2, False))]
            calls += sum(r[2] for r in cd_runs + q2_runs)
            e_cd1 = min(r[0] for r in cd_runs)
            e_q2 = min(r[0] for r in q2_runs)
            fx_cd1 = ref.evaluate(predict(inst, 1, "cd_proposed", model))
            fx_q2 = ref.evaluate(predict(inst, 2, model["qaoa_fixed_best_by_depth"]["2"], model))
            result_queue.put({**t, "m": int(inst.m), "z": z, "gpu": str(gpu),
                              "E_qaoa1_opt": e_q1, "E_cd1_opt": e_cd1, "E_qaoa2_opt": e_q2,
                              "E_cd1_fixed": fx_cd1, "E_qaoa2_fixed": fx_q2,
                              "gain_opt": (e_q2 - e_cd1) / z, "gain_fixed": (fx_q2 - fx_cd1) / z,
                              "eval_calls": calls, "seconds": time.time() - t0})
        except Exception as exc:  # keep going; error is recorded
            result_queue.put({**t, "error": repr(exc), "seconds": time.time() - t0})


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in open(rec_path):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["id"])
    todo = [t for t in tasks_for(args.n, args.per_family, args.seed, parse_override(args.per_family_override)) if t["id"] not in done]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"[largen] {len(todo)} instances to run (n={args.n}, {args.per_family}/family, "
          f"override {args.per_family_override or '-'}), "
          f"{len(done)} already done, {len(gpus)} worker(s)", flush=True)
    if not todo:
        return
    ctx = mp.get_context("spawn")
    gq, tq, rq = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for g in gpus:
        gq.put(g)
    for t in todo:          # ascending n: smaller sizes finish first
        tq.put(t)
    for _ in gpus:
        tq.put(None)
    wargs = {"backend": args.backend, "dtype": args.dtype, "maxiter": args.maxiter, "ftol": args.ftol,
             "gtol": args.gtol, "model": str(Path(args.model).resolve())}
    procs = [ctx.Process(target=_worker, args=(gq, tq, rq, wargs), daemon=True) for _ in gpus]
    for p in procs:
        p.start()
    t0 = time.time()
    with open(rec_path, "a") as f:
        for k in range(1, len(todo) + 1):
            r = rq.get()
            f.write(json.dumps(r) + "\n")
            f.flush()
            msg = f"ERROR {r['error']}" if "error" in r else f"gain_opt {r['gain_opt']:+.4f}"
            print(f"[largen] {k}/{len(todo)} {r['id']} {r['seconds']:.0f}s {msg} "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    for p in procs:
        p.join(timeout=10)
    summary(args)


def summary(args):
    rec_path = Path(args.out) / "records.jsonl"
    if not rec_path.exists():
        print("no records yet")
        return
    rows = [json.loads(line) for line in open(rec_path)]
    err = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    rng = np.random.default_rng(1441)

    def ci(x):
        x = np.asarray(x)
        m = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
        return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
    lines = ["# CD p=1 vs QAOA p=2 beyond n=22 (reduced protocol: 1+2+2 structured starts)", "",
             "gain = (E_QAOA,p=2 - E_CD,p=1)/sum|J|, positive = CD better. Optimized = best of the starts; "
             "fixed = frozen large-run rule.", "",
             "| n | graphs | optimized gain [95% CI] | CD wins (opt) | fixed gain [95% CI] | CD wins (fixed) | mean s/graph |",
             "|---|---:|---|---:|---|---:|---:|"]
    ns = sorted({r["n"] for r in rows})
    for n in ns:
        s = [r for r in rows if r["n"] == n]
        go, lo, hi = ci([r["gain_opt"] for r in s])
        gf, lf, hf = ci([r["gain_fixed"] for r in s])
        lines.append(f"| {n} | {len(s)} | {go:+.4f} [{lo:+.4f}, {hi:+.4f}] | {np.mean([r['gain_opt'] > 1e-9 for r in s]):.0%} | "
                     f"{gf:+.4f} [{lf:+.4f}, {hf:+.4f}] | {np.mean([r['gain_fixed'] > 0 for r in s]):.0%} | "
                     f"{np.mean([r['seconds'] for r in s]):.0f} |")
    lines += ["", "Optimized gain by family:", "", "| family | " + " | ".join(f"n={n}" for n in ns) + " |",
              "|---|" + "---:|" * len(ns)]
    for fam in FAMILIES:
        cells = []
        for n in ns:
            s = [r["gain_opt"] for r in rows if r["n"] == n and r["family"] == fam]
            cells.append(f"{np.mean(s):+.4f}" if s else "–")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    if err:
        lines += ["", f"{len(err)} instance(s) failed, e.g. {err[0]['id']}: {err[0]['error']}"]
    text = "\n".join(lines) + "\n"
    (Path(args.out) / "SUMMARY.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[20, 22, 24, 26, 28])
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--per-family-override", default="28:4",
                    help="graphs per family for specific n, e.g. '28:4' or '26:6,28:4'; '' = none")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--backend", default="lightning.gpu", choices=["lightning.gpu", "lightning.qubit"])
    ap.add_argument("--dtype", default="complex64")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--ftol", type=float, default=1e-6)
    ap.add_argument("--gtol", type=float, default=1e-3)
    ap.add_argument("--model", default="results/eqdepth_large_N20/FROZEN_MODEL.json")
    ap.add_argument("--out", default="results/cd1_vs_qaoa2_largen")
    ap.add_argument("--summary", action="store_true", help="only print/write SUMMARY.md from existing records")
    args = ap.parse_args()
    summary(args) if args.summary else run(args)


if __name__ == "__main__":
    main()
