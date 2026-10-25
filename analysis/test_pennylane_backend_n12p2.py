"""PennyLane and Qulacs must evaluate the same project circuit."""
import numpy as np

from evaluate_unified_table import _instance, energy_qulacs, load_archived_inputs
from unified_variational_n16p2 import run_one, select_rows
from cdtransfer.simulator import Angles
from compare_optimizers_n12p2 import OUT, FIGURE, _valid


def _sample():
    model, cohorts = load_archived_inputs()
    cohort = next(item for item in cohorts if item["name"] == "small")
    row = select_rows(cohort, 1, n=12)[0]
    return model, cohort, row, _instance(row)


def test_pennylane_matches_qulacs_for_fixed_and_random_angles():
    from pennylane_n12p2 import energy_pennylane

    _, _, _, inst = _sample()
    vectors = [np.zeros(6), np.array([.31, -.27, .4, -.22, .15, -.38])]
    for vector in vectors:
        angles = Angles.unpack(vector, 2, True)
        assert abs(energy_pennylane(inst, angles) - energy_qulacs(inst, angles)) < 1e-8


def test_optimizer_trace_can_use_pennylane_without_mixing_backends():
    model, cohort, row, _ = _sample()
    trace = run_one(row, model, cohort["old_results"][(row["id"], 2)],
                    0, 14, method="COBYLA", backend="pennylane")
    assert trace["backend"] == "pennylane"
    assert trace["error"] is None
    assert 1 <= trace["nfev"] <= 14
    assert [event["evaluation"] for event in trace["evaluations"]] == list(range(1, trace["nfev"] + 1))


def test_comparison_keeps_pennylane_runs_separate_from_old_qulacs_runs():
    assert "pennylane" in str(OUT)
    assert "pennylane" in str(FIGURE)
    _, _, row, _ = _sample()
    trace = {"id": row["id"], "fingerprint": row["fingerprint"], "n": 12, "p": 2,
             "start": 0, "method": "COBYLA", "error": None, "nfev": 1,
             "evaluations": [{"evaluation": 1}], "backend": "qulacs"}
    assert not _valid(trace, row, 0, "COBYLA")
