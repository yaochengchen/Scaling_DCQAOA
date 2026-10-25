import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "15_audit_maxcut_approximation.py"
CONFIG = ROOT / "configs" / "maxcut_approximation_current_formula.json"
PRODUCTION = ROOT / "results" / "maxcut_approximation_current_formula"


def _load_audit():
    assert SCRIPT.exists(), "audit script must exist"
    spec = importlib.util.spec_from_file_location("maxcut_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_production_artifacts_pass_independent_audit(tmp_path):
    audit = _load_audit()
    report = tmp_path / "audit.md"

    result = audit.main(CONFIG, PRODUCTION, report)

    assert result["status"] == "PASS"
    assert result["record_count"] == 350
    assert result["unique_id_count"] == 350
    assert result["unique_fingerprint_count"] == 350
    assert result["evaluation_count"] == 840
    assert result["cell_n_sample_slots"] == 350
    assert result["p1_evaluations"] == 700
    assert result["p2_n12_evaluations"] == 140
    records = json.loads((PRODUCTION / "records.json").read_text(encoding="utf-8"))
    ratios = []
    weights = []
    objective_diffs = []
    energy_cut_residuals = []
    ratio_residuals = []
    for row in records:
        weights.extend(row["instance"]["weights"])
        exact = row["exact"]
        objective_diffs.append(abs(exact["scip_objective"] - exact["gurobi_objective"]))
        optimum = exact["optimum"]
        for depth in row["evaluations"].values():
            for evaluation in depth.values():
                ratio = evaluation["approximation_ratio"]
                cut = evaluation["expected_cut"]
                energy = evaluation["ising_energy"]
                ratios.append(ratio)
                energy_cut_residuals.append(
                    abs(cut - 0.5 * (sum(row["instance"]["weights"]) - energy))
                )
                ratio_residuals.append(abs(ratio - cut / optimum))
    expected_extrema = {
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "minimum_weight": min(weights),
        "max_solver_objective_diff": max(objective_diffs),
        "max_energy_cut_residual": max(energy_cut_residuals),
        "max_ratio_identity_residual": max(ratio_residuals),
    }
    for key, expected in expected_extrema.items():
        assert math.isclose(result["extrema"][key], expected, rel_tol=0, abs_tol=0)
    text = report.read_text(encoding="utf-8")
    assert "## Numeric extrema and residuals" in text
    assert "Ratio minimum / maximum" in text
    assert "Minimum positive edge weight" in text
    assert "Maximum |SCIP − Gurobi objective|" in text
    assert "Maximum |expected cut − energy-derived cut|" in text
    assert "Maximum |ratio − expected cut / optimum|" in text
    assert "Tolerance for all identity/agreement checks: `1e-8`" in text
    assert "FINAL: PASS" in text


def test_failed_audit_removes_stale_pass_and_raises(tmp_path):
    audit = _load_audit()
    copied = tmp_path / "run"
    shutil.copytree(PRODUCTION, copied)
    records_path = copied / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records[0]["evaluations"]["1"]["qaoa_csv"]["approximation_ratio"] = 1.5
    records_path.write_text(json.dumps(records), encoding="utf-8")
    report = tmp_path / "audit.md"
    report.write_text("FINAL: PASS\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ratio"):
        audit.main(CONFIG, copied, report)

    assert not report.exists()


@pytest.mark.parametrize("field", ["median", "std", "ci95"])
def test_summary_statistics_are_independently_recomputed(tmp_path, field):
    audit = _load_audit()
    copied = tmp_path / "run"
    shutil.copytree(PRODUCTION, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cell = summary["cells"][0]
    if field == "ci95":
        cell[field] = [cell["mean"] - 0.0001, cell["mean"] + 0.0001]
    else:
        cell[field] += 0.0001
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = tmp_path / "audit.md"

    with pytest.raises(ValueError, match="summary"):
        audit.main(CONFIG, copied, report)

    assert not report.exists()


@pytest.mark.parametrize("tamper", ["solver_diff", "energy_cut_residual"])
def test_absolute_audit_tolerance_is_one_e_minus_eight(tmp_path, tamper):
    audit = _load_audit()
    copied = tmp_path / "run"
    shutil.copytree(PRODUCTION, copied)
    records_path = copied / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if tamper == "solver_diff":
        records[0]["exact"]["scip_objective"] += 2e-8
    else:
        records[0]["evaluations"]["1"]["qaoa_csv"]["expected_cut"] += 2e-8
    records_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="(solver objectives|cut/energy identity)"):
        audit.main(CONFIG, copied, tmp_path / "audit.md")
