import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from transfer import build_model, predict_angles


def write_rows(path, rows):
    import csv

    fields = ["id", "n", "family", "weight_kind", "p", "cd", "degree", "rms",
              "gamma", "beta", "alpha"]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(instance, p, gamma, degree=2.0):
    return {
        "id": instance, "n": "10", "family": "ring", "weight_kind": "unit",
        "p": str(p), "cd": "True", "degree": str(degree), "rms": "1.0",
        "gamma": json.dumps([gamma] * p), "beta": json.dumps([0.2] * p),
        "alpha": json.dumps([-0.1] * p),
    }


class TransferTests(unittest.TestCase):
    def test_predict_uses_one_table_and_exact_weight_scaling(self):
        model = {
            "depths": [1],
            "reference_degree": 2.0,
            "alpha_degree_power": -0.5,
            "alpha_multiplier": 1.0,
            "templates": {
                "1": {"gamma_unit": [-0.5], "beta": [0.3], "alpha_unit": [-0.2]}
            },
        }
        graph = [[0, 1, 1.0], [1, 2, 2.0]]
        first = predict_angles(model, 3, graph, 1)
        scaled = predict_angles(model, 3, [[i, j, 3 * w] for i, j, w in graph], 1)
        self.assertAlmostEqual(first["gamma"][0] / 3, scaled["gamma"][0])
        self.assertAlmostEqual(first["alpha"][0] / 3, scaled["alpha"][0])
        self.assertEqual(first["beta"], scaled["beta"])
        self.assertEqual(first["features"]["degree"], 4 / 3)

    def test_unsupported_depth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            predict_angles({"depths": [1], "templates": {}}, 3, [[0, 1, 1.0]], 5)

    def test_aggregate_deduplicates_training_ids_and_unifies_d0(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            small = directory / "small.csv.gz"
            large = directory / "large.csv.gz"
            extra = directory / "extra.csv.gz"
            write_rows(small, [row("train-a", 1, -0.4, 2),
                               row("train-a", 6, -0.5, 2)])
            write_rows(large, [row("train-b", 1, -0.6, 4)])
            write_rows(extra, [row("train-b", 1, -0.6, 4),
                               row("train-c", 6, -0.7, 6)])
            model = build_model(small, large, extra, depths=(1, 6))
            self.assertEqual(model["reference_degree"], 3.0)
            self.assertEqual(model["training_counts"], {"1": 2, "6": 2})
            self.assertEqual(model["depths"], [1, 6])
            self.assertEqual(len(model["templates"]["6"]["gamma_unit"]), 6)
            self.assertEqual(model["alpha_degree_power"], -0.5)
            self.assertEqual(model["alpha_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
