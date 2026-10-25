import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_unified_table import attach_training_reference, score_pair, summarize_group


def test_score_pair_uses_saved_extrema_and_qaoa():
    historical = {
        "ground_energy": -3.0, "max_energy": 3.0,
        "energy_qaoa_best": -1.0, "norm_weights": 3.0,
        "ratio_cd_opt": 0.9, "ratio_qaoa_opt": 0.8,
    }
    scored = score_pair(1.0, historical)
    assert scored["ratio_cd_unified"] == pytest.approx(1 / 3)
    assert scored["ratio_qaoa_fixed"] == pytest.approx(2 / 3)
    assert scored["gain_fixed_norm"] == pytest.approx(-2 / 3)
    assert scored["transfer_loss_cd"] == pytest.approx(0.9 - 1 / 3)


def test_score_pair_accepts_archived_test_row_without_optimized_reference():
    historical = {
        "ground_energy": -3.0, "max_energy": 3.0,
        "energy_qaoa_best": -1.0, "norm_weights": 3.0,
    }
    scored = score_pair(-2.0, historical)
    assert scored["ratio_cd_unified"] == pytest.approx(5 / 6)
    assert "transfer_loss_cd" not in scored


def test_summary_is_paired_and_reports_wins():
    rows = [
        {"ratio_cd_unified": 0.8, "ratio_qaoa_fixed": 0.7, "gain_fixed_norm": 0.3},
        {"ratio_cd_unified": 0.5, "ratio_qaoa_fixed": 0.6, "gain_fixed_norm": -0.2},
    ]
    out = summarize_group(rows)
    assert out["pairs"] == 2
    assert out["ratio_cd_unified"] == pytest.approx(0.65)
    assert out["ratio_qaoa_fixed"] == pytest.approx(0.65)
    assert out["gain_fixed_norm"] == pytest.approx(0.05)
    assert out["cd_win"] == pytest.approx(0.5)


def test_transfer_loss_uses_training_mean_not_missing_test_field():
    groups = [{"cohort": "small", "family": "tree", "p": 1,
               "ratio_cd_unified": 0.80, "ratio_qaoa_fixed": 0.70}]
    reference = [{"run": "small", "family": "tree", "p": "1",
                  "opt_cd": "0.90", "opt_qaoa": "0.75"}]
    attach_training_reference(groups, reference)
    assert groups[0]["transfer_loss_cd"] == pytest.approx(0.10)
    assert groups[0]["transfer_loss_qaoa"] == pytest.approx(0.05)
