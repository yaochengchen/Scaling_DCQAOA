"""Checks for the all-size, equal-layer comparison."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_pooled_depth_unified import summarize_pooled_depths


def _write_rows(path, missing=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "p", "ratio_qaoa_fixed", "ratio_cd_unified"))
        writer.writeheader()
        for p in (1, 2, 3, 4):
            for graph_id, offset in (("small-a", 0.0), ("large-b", 0.1)):
                if (p, graph_id) == missing:
                    continue
                writer.writerow({"id": graph_id, "p": p,
                                 "ratio_qaoa_fixed": 0.6 + 0.05 * p + offset,
                                 "ratio_cd_unified": 0.7 + 0.05 * p + offset})
        writer.writerow({"id": "small-a", "p": 6,
                         "ratio_qaoa_fixed": 0.99, "ratio_cd_unified": 1.0})


def test_all_four_depths_use_the_same_graphs(tmp_path):
    source = tmp_path / "pairs.csv"
    _write_rows(source)
    summary = summarize_pooled_depths(source, expected_graphs=2)
    assert [row["p"] for row in summary] == [1, 2, 3, 4]
    assert all(row["count"] == 2 for row in summary)
    assert summary[0]["qaoa_mean"] == pytest.approx(0.70)
    assert summary[0]["dc_mean"] == pytest.approx(0.80)


def test_missing_graph_at_one_depth_is_rejected(tmp_path):
    source = tmp_path / "pairs.csv"
    _write_rows(source, missing=(3, "large-b"))
    with pytest.raises(ValueError, match="same 2 graphs"):
        summarize_pooled_depths(source, expected_graphs=2)
