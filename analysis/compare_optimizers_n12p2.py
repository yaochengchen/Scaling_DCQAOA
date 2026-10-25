"""Compare three optimizers with PennyLane on held-out n=12,p=2 graphs."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from evaluate_unified_table import EXPECTED_MODEL_SHA256, ROOT, load_archived_inputs
from run_extended_variational_n16p2 import _save
from unified_variational_n16p2 import BASE_SEED, GROUPS, run_one, select_rows, summarize_runs

METHODS = ("L-BFGS-B", "COBYLA", "Adam")
STARTS = 20
BUDGET = 500
WORKERS = 3
PREVIOUS = ROOT / "results" / "unified_variational_n12p2_optimizer_compare_pennylane_eval500"
OUT = ROOT / "results" / "unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500"
FIGURE = ROOT / "paper" / "figures" / "fixed_vs_variational_unified_n12p2_optimizers_pennylane_starts20_eval500"


def reusable_manifest(previous: dict, current: dict) -> bool:
    """Only the start count and added provenance may differ from the six-start run."""
    return (previous.get("starts_per_graph") == 6
            and current.get("starts_per_graph") == 20
            and all(json.loads(json.dumps(current.get(key))) == value
                    for key, value in previous.items()
                    if key != "starts_per_graph"))


def summarize_methods(traces: list[dict], budget: int,
                      methods: tuple[str, ...] = METHODS) -> dict:
    return {method: summarize_runs([trace for trace in traces if trace["method"] == method], budget)
            for method in methods}


def _valid(trace: dict, row: dict, start: int, method: str) -> bool:
    events = trace["evaluations"]
    return (trace["id"] == row["id"] and trace["fingerprint"] == row["fingerprint"]
            and trace["n"] == 12 and trace["p"] == 2 and trace["start"] == start
            and trace["method"] == method and trace.get("backend") == "pennylane"
            and trace["error"] is None
            and 1 <= trace["nfev"] == len(events) <= BUDGET
            and [event["evaluation"] for event in events] == list(range(1, len(events) + 1)))


def main():
    import pennylane as qml

    model, cohorts = load_archived_inputs()
    cohort = next(item for item in cohorts if item["name"] == "small")
    rows = select_rows(cohort, 3, n=12)
    metadata = {"n": 12, "p": 2, "test_graph_ids": [row["id"] for row in rows],
                "starts_per_graph": STARTS, "budget_evaluations": BUDGET,
                "methods": METHODS, "groups": GROUPS, "base_seed": BASE_SEED,
                "backend": "pennylane", "pennylane_device": "lightning.qubit",
                "pennylane_version": qml.__version__,
                "reused_runs_from": str(PREVIOUS.relative_to(ROOT)),
                "model_sha256": EXPECTED_MODEL_SHA256,
                "adam": {"gradient": "one-sided finite difference", "step": 1e-4,
                         "learning_rate": .08, "beta1": .9, "beta2": .999, "epsilon": 1e-8},
                "cobyla": {"rhobeg": .5, "tol": 1e-4}, "workers": WORKERS}
    manifest = OUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(metadata)):
        raise ValueError("Existing comparison settings differ")
    prior_manifest = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))
    if not reusable_manifest(prior_manifest, metadata):
        raise ValueError("Previous PennyLane settings differ; cannot reuse runs")
    _save(manifest, metadata)

    jobs = []
    pending = []
    for method in METHODS:
        for row in rows:
            historical = cohort["old_results"][(row["id"], 2)]
            for start in range(STARTS):
                path = OUT / "runs" / method / f"{row['id']}_start{start}.json"
                jobs.append((method, row, start, path))
                if path.exists():
                    trace = json.loads(path.read_text(encoding="utf-8"))
                    if not _valid(trace, row, start, method):
                        raise ValueError(f"Invalid saved run: {path}")
                elif start < prior_manifest["starts_per_graph"]:
                    prior = PREVIOUS / "runs" / method / path.name
                    trace = json.loads(prior.read_text(encoding="utf-8"))
                    if not _valid(trace, row, start, method):
                        raise ValueError(f"Invalid prior run: {prior}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(prior, path)
                else:
                    pending.append((method, row, historical, start, path))
    print(f"{len(jobs)} total runs; {len(pending)} new runs; {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, row, model, historical, start, BUDGET, method, "pennylane"):
                   (method, row, start, path)
                   for method, row, historical, start, path in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            method, row, start, path = futures[future]
            trace = future.result()
            if not _valid(trace, row, start, method):
                raise ValueError(f"Invalid computed run: {path}")
            _save(path, trace)
            if completed % 12 == 0 or completed == len(pending):
                print(f"{completed}/{len(pending)} new runs saved", flush=True)

    traces = [json.loads(path.read_text(encoding="utf-8"))
              for _, _, _, path in jobs]
    summary = summarize_methods(traces, BUDGET)
    _save(OUT / "summary.json", summary)
    from plot_optimizer_comparison_n12p2 import plot_comparison
    plot_comparison(summary, FIGURE)
    for method in METHODS:
        item = summary[method]
        print(f"{method}: {sum(hit is not None for hit in item['run_hits'])}/{item['runs']} hits; "
              f"mean final={item['mean_last_best_ratio']:.6f}; "
              f"mean hit={item['mean_trajectory_hit']}", flush=True)


if __name__ == "__main__":
    main()
