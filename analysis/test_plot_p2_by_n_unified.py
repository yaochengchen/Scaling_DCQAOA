"""Checks for the single-panel p=2 size comparison."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_p2_by_n_unified import summarize_p2_by_n


def _write_rows(path, *, omit_n=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("n", "p", "ratio_qaoa_fixed", "ratio_cd_unified"))
        writer.writeheader()
        for n in (10, 12, 14, 16, 18, 20, 22):
            if n == omit_n:
                continue
            for qaoa, cd in ((0.7, 0.8), (0.8, 0.9)):
                writer.writerow({"n": n, "p": 2, "ratio_qaoa_fixed": qaoa,
                                 "ratio_cd_unified": cd})
        writer.writerow({"n": 10, "p": 1, "ratio_qaoa_fixed": 0.1,
                         "ratio_cd_unified": 0.2})


def test_p2_grouping_ignores_other_depths_and_preserves_counts(tmp_path):
    source = tmp_path / "paired.csv"
    _write_rows(source)
    rows = summarize_p2_by_n(source, expected_per_n=2)
    assert [row["n"] for row in rows] == [10, 14, 18, 22]
    assert all(row["count"] == 2 for row in rows)
    assert all(row["qaoa_mean"] == pytest.approx(0.75) for row in rows)
    assert all(row["dc_mean"] == pytest.approx(0.85) for row in rows)
    assert all(row["qaoa_ci_lo"] <= row["qaoa_mean"] <= row["qaoa_ci_hi"]
               for row in rows)


def test_p2_grouping_rejects_incomplete_size_series(tmp_path):
    source = tmp_path / "paired.csv"
    _write_rows(source, omit_n=18)
    with pytest.raises(ValueError, match="Expected 2 p=2 graphs"):
        summarize_p2_by_n(source, expected_per_n=2)
