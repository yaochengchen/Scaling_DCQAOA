import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "CDQAOA_EqualDepth_LightningGPU_8x"))

from cdtransfer.instances import IsingInstance
from cdtransfer.simulator import Angles, StatevectorSimulator
from cdtransfer.transfer import predict
from evaluate_unified_table import energy_qulacs, load_archived_inputs, predict_unified


def test_qulacs_energy_matches_existing_reference():
    inst = IsingInstance.from_dict({
        "n": 3, "edges": [[0, 1], [1, 2]], "weights": [1.0, 1.7],
        "family": "test", "weight_kind": "test", "seed": 1, "split": "test",
    })
    angles = Angles(np.array([-0.41, -0.65]), np.array([0.53, 0.17]),
                    np.array([-0.11, -0.03]))
    assert energy_qulacs(inst, angles) == pytest.approx(
        StatevectorSimulator(inst).evaluate(angles), abs=1e-10)


def test_unified_predictor_uses_one_reference_degree():
    inst = IsingInstance.from_dict({
        "n": 3, "edges": [[0, 1], [1, 2]], "weights": [1.0, 1.0],
        "family": "test", "weight_kind": "test", "seed": 1, "split": "test",
    })
    model = {
        "reference_degree": 3.5, "alpha_degree_power": -0.5,
        "alpha_multiplier": 1.0, "templates": {
            "1": {"gamma_unit": [-0.4], "beta": [0.5], "alpha_unit": [-0.1]}
        },
    }
    a = predict_unified(inst, 1, model)
    d = 4 / 3
    assert a.gamma[0] == pytest.approx(-0.4 * np.arctan(1 / np.sqrt(d - 1)))
    assert a.beta[0] == pytest.approx(0.5)
    assert a.alpha[0] == pytest.approx(-0.1 * (d / 3.5) ** -0.5)


def test_archived_test_splits_have_complete_join():
    model, cohorts = load_archived_inputs(ROOT)
    assert model["reference_degree"] == pytest.approx(3.4545454545454546)
    assert [(c["name"], len(c["rows"]), len(c["depths"])) for c in cohorts] == [
        ("small", 4000, 5), ("large", 3000, 4)
    ]
    for cohort in cohorts:
        assert set(cohort["depths"]) == set(model["depths"]) & set(cohort["depths"])
        assert len(cohort["old_results"]) == len(cohort["rows"]) * len(cohort["depths"])
        for row in cohort["rows"]:
            for p in cohort["depths"]:
                old = cohort["old_results"][(row["id"], p)]
                assert old["energy_qaoa_best"] is not None
                assert old["ground_energy"] is not None
                assert old["max_energy"] is not None


@pytest.mark.parametrize("cohort_index", [0, 1])
def test_qulacs_reproduces_archived_gpu_energy(cohort_index):
    _, cohorts = load_archived_inputs(ROOT)
    cohort = cohorts[cohort_index]
    row = cohort["rows"][0]
    p = cohort["depths"][0]
    old_model = json.loads((cohort["source"] / "FROZEN_MODEL.json").read_text())
    inst = IsingInstance.from_dict({key: row[key] for key in
                                    ("n", "edges", "weights", "family", "weight_kind", "seed", "split")})
    angles = predict(inst, p, "cd_proposed", old_model)
    actual = energy_qulacs(inst, angles)
    expected = cohort["old_results"][(row["id"], p)]["energy_cd_proposed"]
    assert actual == pytest.approx(expected, abs=1e-9)
