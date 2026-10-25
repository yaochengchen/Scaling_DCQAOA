"""Optimizer comparisons count every actual objective call on a held-out graph."""
from evaluate_unified_table import load_archived_inputs
from unified_variational_n16p2 import run_one, select_rows
from compare_optimizers_n12p2 import summarize_methods
import compare_optimizers_n12p2 as comparison
from plot_optimizer_comparison_n12p2 import plot_comparison, attainment_ylim
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest


def test_three_methods_share_start_and_count_every_call():
    model, cohorts = load_archived_inputs()
    cohort = next(item for item in cohorts if item["name"] == "small")
    row = select_rows(cohort, 1, n=12)[0]
    historical = cohort["old_results"][(row["id"], 2)]
    runs = [run_one(row, model, historical, 0, 14, method=method)
            for method in ("L-BFGS-B", "COBYLA", "Adam")]
    assert len({tuple(trace["initial_angles"]) for trace in runs}) == 1
    assert all(trace["method"] == method for trace, method in
               zip(runs, ("L-BFGS-B", "COBYLA", "Adam")))
    assert all(1 <= trace["nfev"] <= 14 and trace["error"] is None for trace in runs)
    assert all([event["evaluation"] for event in trace["evaluations"]]
               == list(range(1, trace["nfev"] + 1)) for trace in runs)
    assert all(abs(trace["fixed_ratio"] - runs[0]["fixed_ratio"]) < 1e-12
               for trace in runs)


def test_adam_gradient_probes_are_counted_in_budget():
    model, cohorts = load_archived_inputs()
    cohort = next(item for item in cohorts if item["name"] == "small")
    row = select_rows(cohort, 1, n=12)[0]
    trace = run_one(row, model, cohort["old_results"][(row["id"], 2)],
                    1, 20, method="Adam")
    assert trace["nfev"] == 20
    assert trace["budget_stop"]


def test_method_summary_counts_censored_runs_without_excluding_them():
    traces = [
        {"method": "Adam", "fingerprint": "a", "fixed_ratio": .8,
         "evaluations": [{"evaluation": 1, "best_ratio": .7},
                         {"evaluation": 2, "best_ratio": .81}],
         "nfev": 2, "budget_stop": True, "optimizer_success": False, "error": None},
        {"method": "Adam", "fingerprint": "b", "fixed_ratio": .9,
         "evaluations": [{"evaluation": 1, "best_ratio": .6},
                         {"evaluation": 2, "best_ratio": .7}],
         "nfev": 2, "budget_stop": True, "optimizer_success": False, "error": None},
    ]
    result = summarize_methods(traces, 2, methods=("Adam",))["Adam"]
    assert result["runs"] == 2
    assert result["run_hit_fraction"] == .5
    assert result["run_hits"] == [2, None]
    assert result["mean_trajectory_hit"] is None


def test_twenty_starts_reuse_prior_pennylane_runs_without_overwriting_them():
    assert comparison.STARTS == 20
    assert comparison.OUT != comparison.PREVIOUS
    assert "starts20" in str(comparison.OUT)
    prior = {"n": 12, "p": 2, "test_graph_ids": ["a"], "starts_per_graph": 6,
             "budget_evaluations": 500, "backend": "pennylane"}
    current = dict(prior, starts_per_graph=20)
    assert comparison.reusable_manifest(prior, current)
    prior["groups"] = [["regular3", "unit"]]
    current["groups"] = (("regular3", "unit"),)
    assert comparison.reusable_manifest(prior, current)
    assert not comparison.reusable_manifest(dict(prior, backend="qulacs"), current)


def test_summary_exposes_tenth_and_ninetieth_percentiles_of_individual_runs():
    traces = [
        {"fingerprint": "a", "fixed_ratio": .8, "evaluations": [
            {"evaluation": 1, "best_ratio": .2}, {"evaluation": 2, "best_ratio": .4}]},
        {"fingerprint": "b", "fixed_ratio": .8, "evaluations": [
            {"evaluation": 1, "best_ratio": .6}]},
    ]
    from unified_variational_n16p2 import summarize_runs
    item = summarize_runs(traces, 2)
    assert item["p10_best_curve"] == pytest.approx([.24, .42])
    assert item["p90_best_curve"] == pytest.approx([.56, .58])


def test_comparison_plot_exports_png_and_pdf():
    traces = [{"method": method, "fingerprint": method, "fixed_ratio": .8,
               "evaluations": [{"evaluation": 1, "best_ratio": .6},
                               {"evaluation": 2, "best_ratio": .81}],
               "nfev": 2, "budget_stop": True, "optimizer_success": False, "error": None}
              for method in ("L-BFGS-B", "COBYLA", "Adam")]
    summary = summarize_methods(traces, 2)
    with TemporaryDirectory() as folder:
        target = Path(folder) / "comparison"
        plot_comparison(summary, target)
        assert target.with_suffix(".png").is_file()
        assert target.with_suffix(".pdf").is_file()


def test_attainment_axis_starts_at_zero_and_shows_small_rates():
    summary = {method: {"run_hits": [None] * 70 + [155, 449], "runs": 72}
               for method in ("L-BFGS-B", "COBYLA", "Adam")}
    assert attainment_ylim(summary) == (0, .05)
