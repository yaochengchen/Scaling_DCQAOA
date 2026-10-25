#!/usr/bin/env python3
"""Formula-initialized refinement on the 12 held-out graphs of the paper's Fig. 3 (n=12, p=2).

Self-contained inside CDQAOA_EqualDepth_LightningGPU_8x: all inputs come from
assets/prr_formula_init_inputs.json (the 12 graphs, their exact extrema, the archived
unified-table CD energy, and the frozen unified angle table).  Uses this repo's own
cdtransfer (Lightning simulator, adjoint gradients, run_lbfgs, random_start); modifies nothing.

Per graph:
  (a) one L-BFGS-B chain started from the frozen-table ("formula") angles,
  (b) --random-starts chains from the campaign's instance-scaled random box (paired control).
Every value-and-gradient call is recorded.

Run from the repo root with the cdqaoa conda env active (see scripts/run_formula_init.sh):
  python scripts/26_formula_init_refine.py --dtype complex128 --ftol 1e-10 --gtol 1e-5 \
      --random-starts 20 --gpu 0 --out results/prr_formula_init_c128
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "assets" / "prr_formula_init_inputs.json"
P = 2


def degree_factor(degree: float) -> float:
    return 1.0 if degree <= 1.0 else float(np.arctan(1.0 / np.sqrt(degree - 1.0)))


def formula_angles(model: dict, n: int, edges, weights, p: int):
    """The frozen rule of the paper (identical to exports/cd-qaoa-gpu-transfer/scripts/transfer.py)."""
    degree = 2.0 * len(edges) / n
    rms = math.sqrt(float(np.mean(np.asarray(weights, float) ** 2)))
    d0, b, c = float(model["reference_degree"]), float(model["alpha_degree_power"]), float(model["alpha_multiplier"])
    t = model["templates"][str(p)]
    gamma = [float(x) * degree_factor(degree) / rms for x in t["gamma_unit"]]
    beta = [float(x) for x in t["beta"]]
    alpha = [float(x) / rms * (degree / d0) ** b * c for x in t["alpha_unit"]]
    return gamma, beta, alpha, {"degree": degree, "rms": rms}


def best_so_far(trace, ratio):
    best, out = -np.inf, []
    for ev in trace:
        best = max(best, ratio(ev["energy"]))
        out.append(best)
    return out


def calls_to_fraction(curve, start, final, frac):
    if final <= start:
        return 0
    target = start + frac * (final - start)
    for k, v in enumerate(curve, 1):
        if v >= target - 1e-12:
            return k
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="lightning.gpu", choices=["lightning.gpu", "lightning.qubit"])
    ap.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--ftol", type=float, default=1e-10)
    ap.add_argument("--gtol", type=float, default=1e-5)
    ap.add_argument("--random-starts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for this run")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "prr_formula_init_c128")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    sys.path.insert(0, str(ROOT))
    from cdtransfer.instances import IsingInstance
    from cdtransfer.lightning import make_simulator
    from cdtransfer.optimization import random_start, run_lbfgs
    from cdtransfer.simulator import Angles

    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    model = inputs["unified_model"]
    settings = {"backend": a.backend, "dtype": a.dtype, "engine": "tape"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "runs").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tol = 1e-4 if a.dtype == "complex64" else 1e-7
    summary = []
    t_all = time.perf_counter()
    for g in inputs["graphs"]:
        inst = IsingInstance.from_dict({k: g[k] for k in ("n", "edges", "weights", "family", "weight_kind", "seed", "split")})
        if inst.fingerprint != g["fingerprint"]:
            raise ValueError(f"{g['id']}: instance fingerprint mismatch (instances.py differs from the campaign?)")
        gamma, beta, alpha, feat = formula_angles(model, inst.n, inst.edges, inst.weights, P)
        formula = Angles(gamma, beta, alpha)
        sim = make_simulator(inst, settings)
        span = g["max_energy"] - g["ground_energy"]
        ratio = lambda e, g=g, span=span: (g["max_energy"] - e) / span

        fixed_energy = sim.evaluate(formula)
        drift = abs(fixed_energy - g["archived_fixed_energy_p2"])
        if drift > tol:
            raise ValueError(f"{g['id']}: fixed-angle energy {fixed_energy:.9f} differs from the archived "
                             f"{g['archived_fixed_energy_p2']:.9f} by {drift:.2e} (> {tol}); conventions differ, stop.")

        chains = [("formula", formula.pack(True))]
        chains += [(f"random{k:02d}", random_start(inst, P, True, rng)) for k in range(a.random_starts)]
        rec = {"id": g["id"], "family": inst.family, "weight_kind": inst.weight_kind, "n": inst.n, "p": P,
               "features": feat, "fixed_angles": formula.to_dict(), "fixed_energy": float(fixed_energy),
               "fixed_ratio": float(ratio(fixed_energy)), "archived_fixed_energy": g["archived_fixed_energy_p2"],
               "chains": {}}
        for name, x0 in chains:
            trace = []
            t0 = time.perf_counter()
            res = run_lbfgs(inst, P, True, x0, sim, maxiter=a.maxiter, trace_callback=trace.append,
                            ftol=a.ftol, gtol=a.gtol)
            curve = best_so_far(trace, ratio)
            start_ratio = float(ratio(trace[0]["energy"]))
            rec["chains"][name] = {
                "initial_angles": Angles.unpack(x0, P, True).to_dict(), "final_angles": res.angles.to_dict(),
                "start_ratio": start_ratio, "final_ratio": float(curve[-1]), "final_energy": float(res.energy),
                "calls": res.evaluations, "success": res.success, "message": res.message,
                "calls_to_99pct": calls_to_fraction(curve, start_ratio, curve[-1], 0.99),
                "calls_to_999pct": calls_to_fraction(curve, start_ratio, curve[-1], 0.999),
                "seconds": time.perf_counter() - t0,
                "best_so_far_ratio": curve,
                "energy_trace": [ev["energy"] for ev in trace],
                "grad_norm_trace": [ev["grad_norm"] for ev in trace]}
        (a.out / "runs" / f"{g['id']}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")

        f = rec["chains"]["formula"]
        rnd = [c for k, c in rec["chains"].items() if k.startswith("random")]
        row = {"id": g["id"], "family": inst.family, "weight_kind": inst.weight_kind,
               "fixed_ratio": rec["fixed_ratio"], "formula_refined_ratio": f["final_ratio"],
               "formula_gain": f["final_ratio"] - rec["fixed_ratio"], "formula_calls": f["calls"],
               "formula_calls_to_99pct": f["calls_to_99pct"], "formula_message": f["message"]}
        if rnd:
            finals = np.array([c["final_ratio"] for c in rnd])
            row.update({"random_mean_final": float(finals.mean()), "random_best_final": float(finals.max()),
                        "random_worst_final": float(finals.min()),
                        "random_mean_calls": float(np.mean([c["calls"] for c in rnd])),
                        "random_reaching_fixed_1e-3": int((finals >= rec["fixed_ratio"] - 1e-3).sum()),
                        "random_reaching_refined_1e-3": int((finals >= f["final_ratio"] - 1e-3).sum()),
                        "random_chains": len(rnd)})
        summary.append(row)
        msg = (f"{g['id']:32s} fixed {rec['fixed_ratio']:.4f} -> refined {f['final_ratio']:.4f} "
               f"(+{row['formula_gain']:.4f}; {f['calls']} calls, 99% at {f['calls_to_99pct']})")
        if rnd:
            msg += (f" | random mean {row['random_mean_final']:.4f}, best {row['random_best_final']:.4f}, "
                    f"{row['random_reaching_fixed_1e-3']}/{len(rnd)} reach fixed")
        print(msg, flush=True)

    with open(a.out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    gains = np.array([s["formula_gain"] for s in summary])
    meta = {"graphs": [s["id"] for s in summary], "p": P, "backend": a.backend, "dtype": a.dtype,
            "maxiter": a.maxiter, "ftol": a.ftol, "gtol": a.gtol, "random_starts": a.random_starts, "seed": a.seed,
            "unified_model_sha256": inputs["unified_model_sha256"],
            "mean_fixed_ratio": float(np.mean([s["fixed_ratio"] for s in summary])),
            "mean_refined_ratio": float(np.mean([s["formula_refined_ratio"] for s in summary])),
            "mean_gain": float(gains.mean()), "max_gain": float(gains.max()), "min_gain": float(gains.min()),
            "mean_formula_calls": float(np.mean([s["formula_calls"] for s in summary])),
            "total_seconds": time.perf_counter() - t_all}
    if a.random_starts:
        meta.update({"random_mean_final_over_graphs": float(np.mean([s["random_mean_final"] for s in summary])),
                     "random_reaching_fixed_total": int(sum(s["random_reaching_fixed_1e-3"] for s in summary)),
                     "random_chains_total": int(sum(s["random_chains"] for s in summary))})
    try:
        import pennylane as qml
        meta["pennylane_version"] = qml.__version__
    except Exception:
        pass
    (a.out / "summary.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nmean fixed {meta['mean_fixed_ratio']:.4f} -> refined {meta['mean_refined_ratio']:.4f}; "
          f"mean gain {meta['mean_gain']:.4f} (min {meta['min_gain']:.4f}, max {meta['max_gain']:.4f}); "
          f"mean {meta['mean_formula_calls']:.0f} value-and-gradient calls; {meta['total_seconds']:.0f} s total")
    if a.random_starts:
        print(f"random chains: mean final {meta['random_mean_final_over_graphs']:.4f}; "
              f"{meta['random_reaching_fixed_total']}/{meta['random_chains_total']} reach the formula value")


if __name__ == "__main__":
    main()
