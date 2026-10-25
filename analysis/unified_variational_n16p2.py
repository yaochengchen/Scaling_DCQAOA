"""Small held-out n=16,p=2 variational pilot against the locked unified GPU table.

Each optimizer objective call is recorded, including finite-difference calls.
The archived test graphs and unified model are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from evaluate_unified_table import (EXPECTED_MODEL_SHA256, ROOT, _instance,
                                    energy_qulacs, load_archived_inputs,
                                    predict_unified)
from cdtransfer.simulator import Angles

GROUPS = (("regular3", "unit"), ("er3", "unit"),
          ("regular3", "uniform"), ("er3", "uniform"))
BASE_SEED = 20261022
TOLERANCE = 0.001
OUT = ROOT / "results" / "unified_variational_n16p2"
FIGURE = ROOT / "paper" / "figures" / "fixed_vs_variational_unified_n16p2"


class BudgetStop(Exception):
    pass


def optimizer_options(budget: int) -> dict:
    """Do not cap iterations at the old 80 when extending evaluations."""
    return {"maxiter": 80 if budget <= 200 else budget, "ftol": 1e-10}


def mean_best_trajectory(traces: list[dict], budget: int) -> list[float]:
    """Equal weight per run; carry forward its last best value after termination."""
    values = []
    for count in range(1, budget + 1):
        observed = []
        for trace in traces:
            events = trace["evaluations"]
            if not events:
                raise ValueError("Empty objective trace")
            if [event["evaluation"] for event in events] != list(range(1, len(events) + 1)):
                raise ValueError("Nonconsecutive objective evaluation numbers")
            observed.append(events[min(count, len(events)) - 1]["best_ratio"])
        values.append(float(np.mean(observed)))
    return values


def summarize_runs(traces: list[dict], budget: int) -> dict:
    if not traces:
        raise ValueError("No traces")
    targets = [trace["fixed_ratio"] for trace in traces]
    mean_fixed = float(np.mean(targets))
    curve = mean_best_trajectory(traces, budget)
    per_run_curves = np.asarray([
        [trace["evaluations"][min(count, len(trace["evaluations"])) - 1]["best_ratio"]
         for count in range(1, budget + 1)]
        for trace in traces], dtype=float)
    p10, p90 = np.quantile(per_run_curves, [0.1, 0.9], axis=0)
    hits = [next((event["evaluation"] for event in trace["evaluations"]
                  if event["best_ratio"] >= trace["fixed_ratio"] - TOLERANCE), None)
            for trace in traces]
    mean_hit = next((index for index, ratio in enumerate(curve, 1)
                     if ratio >= mean_fixed - TOLERANCE), None)
    return {
        "runs": len(traces), "graphs": len({trace["fingerprint"] for trace in traces}),
        "budget_evaluations": budget, "tolerance": TOLERANCE,
        "mean_fixed_ratio": mean_fixed, "mean_best_curve": curve,
        "p10_best_curve": p10.tolist(), "p90_best_curve": p90.tolist(),
        "mean_trajectory_hit": mean_hit, "run_hits": hits,
        "run_hit_fraction": sum(hit is not None for hit in hits) / len(hits),
        "mean_hit_among_successful": (float(np.mean([hit for hit in hits if hit is not None]))
                                      if any(hit is not None for hit in hits) else None),
        "mean_last_best_ratio": curve[-1],
        "errors": sum(trace.get("error") is not None for trace in traces),
    }


def grouped_attainment(traces: list[dict], budget: int) -> list[dict]:
    """Per-group first-hit evaluations, with nonattainment retained as censored."""
    groups = []
    for family, weight in GROUPS:
        subset = [trace for trace in traces if trace["family"] == family
                  and trace["weight_kind"] == weight]
        if not subset:
            continue
        hits = [next((event["evaluation"] for event in trace["evaluations"]
                      if event["best_ratio"] >= trace["fixed_ratio"] - TOLERANCE), None)
                for trace in subset]
        cumulative = [sum(hit is not None and hit <= count for hit in hits) / len(hits)
                      for count in range(budget + 1)]
        groups.append({"family": family, "weight_kind": weight, "runs": len(subset),
                       "hits": hits, "not_reached": hits.count(None),
                       "cumulative": cumulative})
    return groups


def select_rows(cohort: dict, per_group: int, n: int = 16) -> list[dict]:
    chosen = []
    for family, weight in GROUPS:
        candidates = [row for row in cohort["rows"] if row["n"] == n
                      and row["family"] == family and row["weight_kind"] == weight]
        if len(candidates) < per_group:
            raise ValueError(f"Insufficient held-out graphs: {family}/{weight}")
        chosen.extend(candidates[:per_group])
    return chosen


def run_one(row: dict, model: dict, old: dict, start: int, budget: int,
            method: str = "L-BFGS-B", backend: str = "qulacs") -> dict:
    if method not in ("L-BFGS-B", "COBYLA", "Adam"):
        raise ValueError(f"Unsupported optimizer: {method}")
    if backend == "pennylane":
        from pennylane_n12p2 import make_pennylane_evaluator
    elif backend != "qulacs":
        raise ValueError(f"Unsupported circuit backend: {backend}")
    inst = _instance(row)
    if row["split"] != "test" or inst.fingerprint != row["fingerprint"]:
        raise ValueError("Test split or graph fingerprint mismatch")
    archived = ROOT / "results" / "unified_table_heldout" / "small" / f"{row['id']}.json"
    fixed_record = next(entry for entry in json.loads(archived.read_text(encoding="utf-8"))
                        if entry["p"] == 2)
    if fixed_record["fingerprint"] != inst.fingerprint or fixed_record["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("Archived unified result mismatch")
    fixed_angles = predict_unified(inst, 2, model)
    evaluate_energy = (make_pennylane_evaluator(inst) if backend == "pennylane"
                       else lambda angles: energy_qulacs(inst, angles))
    fixed_energy = evaluate_energy(fixed_angles)
    if abs(fixed_energy - fixed_record["energy_cd_unified"]) > 1e-7:
        raise ValueError("Fixed-angle cross-check failed")
    optimum = (old["max_energy"] - old["ground_energy"]) / 2
    if optimum <= 0:
        raise ValueError("Invalid MaxCut optimum")
    def ratio(energy):
        return (old["max_energy"] - energy) / (2 * optimum)
    fixed_ratio = ratio(fixed_energy)
    bounds = [(-math.pi, math.pi)] * 2 + [(-math.pi / 2, math.pi / 2)] * 2 + [(-math.pi, math.pi)] * 2
    seed = int.from_bytes(hashlib.sha256(f"{BASE_SEED}:{inst.fingerprint}:{start}".encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    x0 = np.array([rng.uniform(low, high) for low, high in bounds])
    evaluations = []
    best = -np.inf
    error = None
    optimizer_result = None
    began = time.perf_counter()

    def objective(x):
        nonlocal best
        if len(evaluations) >= budget:
            raise BudgetStop()
        energy = evaluate_energy(Angles.unpack(x, 2, True))
        value = ratio(energy)
        if not np.isfinite(value) or not -1e-7 <= value <= 1 + 1e-7:
            raise ValueError("Invalid approximation ratio")
        best = max(best, value)
        evaluations.append({"evaluation": len(evaluations) + 1,
                            "ratio": float(value), "best_ratio": float(best)})
        return energy

    try:
        if method == "Adam":
            x = x0.copy()
            first_moment = np.zeros_like(x)
            second_moment = np.zeros_like(x)
            step = 0
            while True:
                fx = objective(x)
                gradient = np.zeros_like(x)
                for coordinate, (lower, upper) in enumerate(bounds):
                    probe = x.copy()
                    probe[coordinate] = (x[coordinate] + 1e-4 if x[coordinate] + 1e-4 <= upper
                                         else x[coordinate] - 1e-4)
                    gradient[coordinate] = (objective(probe) - fx) / (probe[coordinate] - x[coordinate])
                step += 1
                first_moment = .9 * first_moment + .1 * gradient
                second_moment = .999 * second_moment + .001 * gradient * gradient
                correction = (.08 * (first_moment / (1 - .9 ** step))
                              / (np.sqrt(second_moment / (1 - .999 ** step)) + 1e-8))
                x = np.clip(x - correction, [bound[0] for bound in bounds],
                            [bound[1] for bound in bounds])
        elif method == "COBYLA":
            optimizer_result = minimize(objective, x0, method="COBYLA", bounds=bounds,
                                        options={"maxiter": budget, "rhobeg": .5, "tol": 1e-4})
        else:
            optimizer_result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                        options=optimizer_options(budget))
    except BudgetStop:
        pass
    except Exception as exc:
        error = repr(exc)
    return {"id": row["id"], "fingerprint": inst.fingerprint, "family": inst.family,
            "weight_kind": inst.weight_kind, "n": inst.n, "p": 2, "start": start, "seed": seed,
            "method": method, "backend": backend, "initial_angles": x0.tolist(),
            "fixed_angles": fixed_angles.to_dict(), "fixed_ratio": float(fixed_ratio),
            "fixed_energy": float(fixed_energy), "optimum": float(optimum),
            "evaluations": evaluations, "nfev": len(evaluations),
            "nit": (int(optimizer_result.nit) if optimizer_result is not None
                    and getattr(optimizer_result, "nit", None) is not None else None),
            "optimizer_status": int(optimizer_result.status) if optimizer_result is not None else None,
            "optimizer_success": bool(optimizer_result.success) if optimizer_result else False,
            "optimizer_message": str(optimizer_result.message) if optimizer_result else None,
            "budget_stop": len(evaluations) >= budget, "error": error,
            "seconds": time.perf_counter() - began}


def plot(traces: list[dict], summary: dict) -> None:
    budget = summary["budget_evaluations"]
    x = np.arange(1, budget + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.9), layout="constrained")
    for trace in traces:
        values = np.array([event["best_ratio"] for event in trace["evaluations"]])
        ax.plot(np.arange(1, len(values) + 1), values, color="#a79bc9", alpha=.17, lw=.7)
    ax.plot(x, summary["mean_best_curve"], color="#6651a7", lw=2.6,
            label="Variational DC-QAOA: mean best-so-far")
    ax.axhline(summary["mean_fixed_ratio"], color="#e0a11d", lw=2.7,
               label="Formula-based DC-QAOA (unified GPU table)")
    hit = summary["mean_trajectory_hit"]
    if hit is not None:
        ax.axvline(hit, color="#555555", ls=":", lw=1.2)
        ax.annotate(f"Mean catch-up: {hit} evaluations", (hit, summary["mean_fixed_ratio"]),
                    xytext=(6, 12), textcoords="offset points", fontsize=9)
    ax.set(xlim=(0, budget), xlabel="Target-instance objective-function evaluations",
           ylabel="Mean MaxCut approximation ratio",
           title="Held-out MaxCut, n=16, p=2: fixed rule vs variational search")
    ax.grid(alpha=.25)
    ax.legend(loc="lower right", frameon=True)
    fig.text(.5, -.02, f"12 graphs × 3 random starts; L-BFGS-B; at most {budget} evaluations/run. "
             "Faint curves: individual runs; early-stop curves are not extended.",
             ha="center", fontsize=8, color="#555555")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-per-group", type=int, default=3)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--budget", type=int, default=200)
    args = parser.parse_args()
    if min(args.graphs_per_group, args.starts, args.budget) < 1:
        parser.error("All counts must be positive")
    model, cohorts = load_archived_inputs()
    cohort = next(item for item in cohorts if item["name"] == "small")
    rows = select_rows(cohort, args.graphs_per_group)
    meta = {"n": 16, "p": 2, "groups": GROUPS, "graphs_per_group": args.graphs_per_group,
            "starts": args.starts, "budget": args.budget, "optimizer": "L-BFGS-B",
            "base_seed": BASE_SEED, "model_sha256": EXPECTED_MODEL_SHA256,
            "cohort": "eqdepth_small_N20/test"}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(meta)):
        raise ValueError("Existing output has different experiment settings")
    manifest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    traces = []
    for index, row in enumerate(rows, 1):
        historical = cohort["old_results"][(row["id"], 2)]
        for start in range(args.starts):
            destination = OUT / "runs" / f"{row['id']}_start{start}.json"
            if destination.exists():
                trace = json.loads(destination.read_text(encoding="utf-8"))
                if trace["fingerprint"] != row["fingerprint"] or trace["nfev"] != len(trace["evaluations"]):
                    raise ValueError("Invalid saved trajectory")
            else:
                trace = run_one(row, model, historical, start, args.budget)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(trace) + "\n", encoding="utf-8")
            traces.append(trace)
        print(f"{index}/{len(rows)} graphs, {len(traces)} trajectories", flush=True)
    result = summarize_runs(traces, args.budget)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot(traces, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("mean_best_curve", "run_hits")}, indent=2))


if __name__ == "__main__":
    main()
