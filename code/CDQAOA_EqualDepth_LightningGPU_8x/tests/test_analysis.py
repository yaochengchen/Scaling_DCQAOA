from __future__ import annotations

import numpy as np

from cdtransfer.analysis import paired_method_summary, concentration_by_depth


def test_paired_method_summary_uses_instance_normalization():
    rows = [
        {"norm_weights": 2.0, "energy_ref": -2.0, "energy_candidate": -2.5},
        {"norm_weights": 1.0, "energy_ref": -1.0, "energy_candidate": -0.5},
    ]
    result = paired_method_summary(rows, "energy_ref", "energy_candidate", n_boot=100)
    assert result["n"] == 2
    assert result["mean_gain"] == -0.125
    assert result["median_gain"] == -0.125
    assert result["win_rate"] == 0.5
    assert result["loss_rate"] == 0.5


def test_concentration_selects_best_restart_and_scales_angles():
    base = {
        "p": 1,
        "cd": True,
        "features": {"rms": 2.0, "degree": 3.0},
        "optimizer_success": True,
    }
    records = [
        {**base, "id": "a", "restart": 0, "energy": -1.0,
         "angles": {"gamma": [0.2], "beta": [0.3], "alpha": [0.4]}},
        {**base, "id": "a", "restart": 1, "energy": -2.0,
         "angles": {"gamma": [0.1], "beta": [0.5], "alpha": [0.2]}},
        {**base, "id": "b", "restart": 0, "energy": -3.0,
         "angles": {"gamma": [0.3], "beta": [0.7], "alpha": [0.4]}},
    ]
    result = concentration_by_depth(records)["1"]
    assert result["n_instances"] == 2
    np.testing.assert_allclose(result["alpha_unit_median"], [0.6])
    np.testing.assert_allclose(result["beta_median"], [0.6])
    assert result["all_restart_success_rate"] == 1.0
