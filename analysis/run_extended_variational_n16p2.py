"""Rerun the exact 12 held-out graphs and 36 seeds with 500 evaluations.

The original 200-evaluation experiment remains untouched. Each extended trace
is checked against its saved old prefix to ensure comparability.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_unified_table import EXPECTED_MODEL_SHA256, ROOT, load_archived_inputs
from plot_unified_variational_three_panel import plot_three_panel
from unified_variational_n16p2 import (BASE_SEED, GROUPS, OUT, run_one,
                                       select_rows, summarize_runs)

BUDGET = 500
DESTINATION = ROOT / "results" / "unified_variational_n16p2_eval500"
FIGURE = ROOT / "paper" / "figures" / "fixed_vs_variational_unified_n16p2_eval500_three_panel"


def prefix_matches(old: dict, new: dict) -> bool:
    earlier = old["evaluations"]
    later = new["evaluations"]
    return (len(later) >= len(earlier)
            and all(a["evaluation"] == b["evaluation"]
                    and np.isclose(a["ratio"], b["ratio"], atol=1e-9)
                    and np.isclose(a["best_ratio"], b["best_ratio"], atol=1e-9)
                    for a, b in zip(earlier, later)))


def _save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    model, cohorts = load_archived_inputs()
    cohort = next(item for item in cohorts if item["name"] == "small")
    rows = select_rows(cohort, 3)
    metadata = {"n": 16, "p": 2, "graphs": 12, "starts_per_graph": 3,
                "budget_evaluations": BUDGET, "optimizer": "L-BFGS-B",
                "base_seed": BASE_SEED, "groups": GROUPS,
                "model_sha256": EXPECTED_MODEL_SHA256,
                "old_run": str(OUT.relative_to(ROOT))}
    manifest = DESTINATION / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(metadata)):
        raise ValueError("Existing extended-run metadata mismatch")
    _save(manifest, metadata)
    traces = []
    original = []
    for index, row in enumerate(rows, 1):
        historical = cohort["old_results"][(row["id"], 2)]
        for start in range(3):
            filename = f"{row['id']}_start{start}.json"
            old = json.loads((OUT / "runs" / filename).read_text(encoding="utf-8"))
            path = DESTINATION / "runs" / filename
            new = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                   else run_one(row, model, historical, start, BUDGET))
            if (new["fingerprint"] != old["fingerprint"] or new["seed"] != old["seed"]
                    or new["nfev"] != len(new["evaluations"]) or new["nfev"] > BUDGET
                    or not prefix_matches(old, new)):
                raise ValueError(f"Extended trajectory diverged from original: {filename}")
            if not path.exists():
                _save(path, new)
            traces.append(new)
            original.append(old)
        print(f"{index}/{len(rows)} graphs, {len(traces)} trajectories", flush=True)
    summary = summarize_runs(traces, BUDGET)
    original_hits = [next((event["evaluation"] for event in trace["evaluations"]
                           if event["best_ratio"] >= trace["fixed_ratio"] - .001), None)
                     for trace in original]
    termination = Counter("budget" if trace["budget_stop"] else
                          "optimizer_converged" if trace["optimizer_success"] else
                          "error_or_other" for trace in traces)
    summary["termination_counts"] = dict(termination)
    summary["old_budget_200_hits"] = sum(hit is not None for hit in original_hits)
    summary["prefixes_match_old"] = True
    _save(DESTINATION / "summary.json", summary)
    plot_three_panel(traces, summary, FIGURE)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("mean_best_curve", "run_hits")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
