from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

from cdtransfer.instances import features, make_instance


ROOT = Path(__file__).resolve().parents[1]
PLOTTER_PATH = ROOT / "scripts" / "14_plot_maxcut_approximation.py"


def _load_plotter():
    spec = importlib.util.spec_from_file_location("maxcut_plotter", PLOTTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evaluation(value: float, depths=(1, 2), optimum=1.0, weight_sum=1.0) -> dict:
    return {
        str(p): {
            method: {
                "approximation_ratio": value + (0.01 if method == "cd_proposed" else 0.0),
                "expected_cut": optimum * (value + (0.01 if method == "cd_proposed" else 0.0)),
                "ising_energy": weight_sum - 2.0 * optimum * (value + (0.01 if method == "cd_proposed" else 0.0)),
                "angles": {"gamma": [0.1] * p, "beta": [0.2] * p, "alpha": [0.0] * p},
            }
            for method in ("qaoa_csv", "cd_proposed")
        }
        for p in depths
    }


def _row(value: float, *, n=12, family="regular3", weight="unit", sample=0) -> dict:
    token = f"20261012:{n}:{family}:{weight}:{sample}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
    inst = make_instance(n, family, weight, seed, split="comparison")
    optimum = float(inst.weights.sum())
    row = {
        "id": f"{family}-{weight}-n{n}-sample{sample:02d}",
        "instance": inst.to_dict(),
        "features": features(inst),
        "exact": {
            "optimum": optimum,
            "scip_objective": optimum,
            "gurobi_objective": optimum,
            "scip_status": "optimal",
            "gurobi_status": "optimal",
            "scip_version": "test",
            "gurobi_version": "test",
            "scip_seconds": 0.01,
            "gurobi_seconds": 0.01,
        },
        "evaluations": _evaluation(value, (1, 2) if n == 12 else (1,), optimum, optimum),
    }
    payload = {key: row["instance"][key] for key in ("n", "edges", "weights")}
    row["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return row


def _complete_config() -> dict:
    return {
        "version": "maxcut-approx-current-v1",
        "seed": 20261012,
        "n_values": [8, 10, 12, 14, 16],
        "layer_n": 12,
        "depths": [1, 2],
        "samples_per_cell": 10,
        "structure_cells": [["tree", "unit"], ["regular3", "unit"], ["er3", "unit"], ["er6", "unit"]],
        "weight_cells": [["regular3", "unit"], ["regular3", "uniform"], ["regular3", "bimodal"], ["regular3", "lognormal"]],
        "methods": ["qaoa_csv", "cd_proposed"],
        "frozen_model": "assets/CD_TRANSFER_V1_FROZEN_MODEL.json",
        "bootstrap_resamples": 5000,
    }


def _write_saved_run(base: Path, config: dict, records: list[dict]) -> Path:
    config_path = base / "config.json"
    config_text = json.dumps(config, indent=2) + "\n"
    config_path.write_text(config_text, encoding="utf-8")
    base.mkdir(exist_ok=True)
    manifest = {
        "config": config,
        "config_text": config_text,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "frozen_model_sha256": "a" * 64,
        "qaoa_source_csv_sha256": "b" * 64,
        "expected_records": len(records),
    }
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (base / "records.json").write_text(json.dumps(records), encoding="utf-8")
    return config_path


def _complete_records(config: dict) -> list[dict]:
    cells = list(dict.fromkeys(tuple(x) for x in config["structure_cells"] + config["weight_cells"]))
    return [
        _row(0.70 + 0.002 * sample, n=n, family=family, weight=weight, sample=sample)
        for family, weight in cells
        for n in config["n_values"]
        for sample in range(10)
    ]


def test_bootstrap_and_summary_are_deterministic_and_complete():
    plotter = _load_plotter()
    rows = [_row(0.70 + i / 100, sample=i) for i in range(10)]
    summary = plotter.summarize_cell(rows, method="cd_proposed", p=1, n_boot=200, seed=7)
    assert summary["n"] == 10
    assert summary["mean"] == pytest.approx(0.755)
    assert summary["ci95"][0] <= summary["mean"] <= summary["ci95"][1]
    assert summary == plotter.summarize_cell(rows, method="cd_proposed", p=1, n_boot=200, seed=7)
    assert plotter.bootstrap_mean_ci([0.2, 0.4, 0.6], n_boot=50, seed=3) == plotter.bootstrap_mean_ci(
        [0.2, 0.4, 0.6], n_boot=50, seed=3
    )
    constant = plotter.bootstrap_mean_ci([0.42] * 10, n_boot=50, seed=3)
    assert constant == pytest.approx([0.42, 0.42])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, 1.01])
def test_summary_rejects_nonfinite_or_out_of_range(bad):
    plotter = _load_plotter()
    rows = [_row(0.8, sample=i) for i in range(10)]
    rows[0]["evaluations"]["1"]["qaoa_csv"]["approximation_ratio"] = bad
    with pytest.raises(ValueError, match="finite approximation ratios"):
        plotter.summarize_cell(rows, method="qaoa_csv", p=1, n_boot=20, seed=1)


def test_summary_rejects_missing_row():
    plotter = _load_plotter()
    with pytest.raises(ValueError, match="requires exactly 10"):
        plotter.summarize_cell([_row(0.8, sample=i) for i in range(9)], method="qaoa_csv", p=1, n_boot=20, seed=1)


def test_complete_saved_records_create_both_figures_summary_and_report(tmp_path, monkeypatch):
    plotter = _load_plotter()
    config = _complete_config()
    records = _complete_records(config)
    figure_texts = []
    original_savefig = matplotlib.figure.Figure.savefig

    def capture_savefig(figure, *args, **kwargs):
        figure_texts.append([text.get_text() for text in figure.texts])
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)
    out = tmp_path / "run"
    out.mkdir()
    config_path = _write_saved_run(out, config, records)
    plotter.main(config_path, out)
    for relative in (
        "figures/fig01_graph_structures_same_layer.png",
        "figures/fig02_weight_distributions_same_layer.png",
        "summary.json",
        "EXPERIMENT_REPORT.md",
    ):
        artifact = out / relative
        assert artifact.stat().st_size > 0
        if artifact.suffix == ".png":
            assert plt.imread(artifact).shape[:2] == (2340, 2070)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["cells"]) == 7 * 5 * 2 + 7 * 1 * 2
    assert summary["complete"] is True
    assert summary["figures"] == {
        "dpi": 180,
        "layout": [4, 2],
        "colors": {"qaoa_csv": "tab:blue", "cd_proposed": "tab:orange"},
        "offsets": {"qaoa_csv": -0.1, "cd_proposed": 0.1},
        "n_values": [8, 10, 12, 14, 16],
        "depths": [1, 2],
        "shared_ylim": summary["figures"]["shared_ylim"],
        "warning": "Same-layer comparison; not matched two-qubit-gate budget",
        "watermark": None,
        "structure_rows": ["tree/unit", "regular3/unit", "er3/unit", "er6/unit"],
        "weight_rows": ["regular3/unit", "regular3/uniform", "regular3/bimodal", "regular3/lognormal"],
    }
    assert summary["figures"]["shared_ylim"][1] == 1.01
    assert all(cell["n"] == 10 and cell["ci95"][0] <= cell["mean"] <= cell["ci95"][1] for cell in summary["cells"])
    assert all(cell["mean"] - cell["ci95"][0] >= 0 and cell["ci95"][1] - cell["mean"] >= 0 for cell in summary["cells"])
    report = (out / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8").lower()
    assert "same-layer" in report and "not a matched or equal two-qubit-gate-budget" in report
    assert all("INCOMPLETE SMOKE" not in texts for texts in figure_texts)


def test_smoke_flag_is_watermarked_and_rejected_for_production(tmp_path, monkeypatch):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    out = tmp_path / "smoke"
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [_row(0.8, n=8)])
    figure_texts = []
    original_savefig = matplotlib.figure.Figure.savefig

    def capture_savefig(figure, *args, **kwargs):
        figure_texts.append([text.get_text() for text in figure.texts])
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)
    plotter.main(config_path, out, allow_incomplete_smoke=True)
    assert "INCOMPLETE SMOKE" in (out / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert all("INCOMPLETE SMOKE" in texts for texts in figure_texts)

    no_flag = tmp_path / "smoke-no-flag"
    no_flag.mkdir()
    no_flag_config = _write_saved_run(no_flag, smoke, [_row(0.8, n=8)])
    with pytest.raises(ValueError, match="requires --allow-incomplete-smoke"):
        plotter.main(no_flag_config, no_flag)

    production_out = tmp_path / "production"
    production_out.mkdir()
    production = _complete_config()
    production_path = _write_saved_run(production_out, production, [])
    with pytest.raises(ValueError, match="production config"):
        plotter.main(production_path, production_out, allow_incomplete_smoke=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(version="maxcut-approx-current-v2"),
        lambda c: c.update(n_values=[8, 10, 12, 14]),
        lambda c: c.update(layer_n=10),
        lambda c: c.update(depths=[1]),
        lambda c: c.update(samples_per_cell=9),
        lambda c: c.update(structure_cells=c["structure_cells"][:-1]),
        lambda c: c.update(weight_cells=c["weight_cells"][:-1]),
        lambda c: c.update(methods=list(reversed(c["methods"]))),
        lambda c: c.update(seed=20261013),
        lambda c: c.update(frozen_model="assets/other.json"),
        lambda c: c.update(bootstrap_resamples=4999),
    ],
)
def test_production_config_is_exactly_locked(mutation):
    plotter = _load_plotter()
    config = _complete_config()
    mutation(config)
    with pytest.raises(ValueError, match="approved production config"):
        plotter._validate_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(depths=[True, 2]),
        lambda c: c.update(depths=[1.0, 2]),
        lambda c: c.update(n_values=[8.0, 10, 12, 14, 16]),
        lambda c: c.update(seed=True),
        lambda c: c.update(seed=20261012.0),
        lambda c: c.update(layer_n=True),
        lambda c: c.update(layer_n=12.0),
        lambda c: c.update(samples_per_cell=True),
        lambda c: c.update(samples_per_cell=10.0),
        lambda c: c.update(bootstrap_resamples=True),
        lambda c: c.update(bootstrap_resamples=5000.0),
    ],
)
def test_config_integer_types_are_strict(mutation):
    plotter = _load_plotter()
    config = _complete_config()
    mutation(config)
    with pytest.raises(ValueError, match="JSON integer|approved production config"):
        plotter._validate_config(config)


def test_production_rejects_one_missing_record(tmp_path):
    plotter = _load_plotter()
    config = _complete_config()
    records = _complete_records(config)
    records.pop()
    out = tmp_path / "missing"
    out.mkdir()
    config_path = _write_saved_run(out, config, records)
    with pytest.raises(ValueError, match="incomplete"):
        plotter.main(config_path, out)
    assert not (out / "summary.json").exists()


def test_forged_fingerprint_is_fatal(tmp_path):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    row = _row(0.8, n=8)
    row["fingerprint"] = "f" * 64
    out = tmp_path / "forged"
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [row])
    with pytest.raises(ValueError, match="fingerprint"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)


@pytest.mark.parametrize("tamper", ["energy", "cut_and_ratio"])
def test_energy_cut_ratio_identity_is_enforced(tmp_path, tamper):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    row = _row(0.8, n=8)
    result = row["evaluations"]["1"]["qaoa_csv"]
    if tamper == "energy":
        result["ising_energy"] += 0.5
    else:
        result["expected_cut"] += 0.5
        result["approximation_ratio"] = result["expected_cut"] / row["exact"]["optimum"]
    out = tmp_path / tamper
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [row])
    with pytest.raises(ValueError, match="Ising energy"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)


@pytest.mark.parametrize("extra", ["depth", "method", "nonfinite_method"])
def test_evaluation_keys_are_exact(tmp_path, extra):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    row = _row(0.8, n=8)
    if extra == "depth":
        row["evaluations"]["2"] = row["evaluations"]["1"]
    else:
        row["evaluations"]["1"]["target_optimized"] = {
            "approximation_ratio": float("nan") if extra == "nonfinite_method" else 0.5
        }
    out = tmp_path / extra
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [row])
    with pytest.raises(ValueError, match="evaluation (depths|methods)"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", 123), ("split", "train"), ("family", "tree")],
)
def test_saved_instance_must_match_deterministic_regeneration(tmp_path, field, value):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    row = _row(0.8, n=8)
    row["instance"][field] = value
    out = tmp_path / field
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [row])
    with pytest.raises(ValueError, match="deterministic instance"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)


@pytest.mark.parametrize("field", ["degree", "rms"])
def test_saved_features_must_match_recomputation(tmp_path, field):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    row = _row(0.8, n=8)
    row["features"][field] += 0.25
    out = tmp_path / field
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [row])
    with pytest.raises(ValueError, match="features"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)


def test_smoke_config_is_exactly_locked():
    plotter = _load_plotter()
    smoke = dict(plotter.APPROVED_SMOKE)
    smoke["bootstrap_resamples"] += 1
    with pytest.raises(ValueError, match="approved production config or approved smoke config"):
        plotter._validate_config(smoke)


def test_savefig_failure_still_closes_figure(tmp_path, monkeypatch):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    out = tmp_path / "save-failure"
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [_row(0.8, n=8)])
    before = set(plt.get_fignums())
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)
    assert set(plt.get_fignums()) == before


def test_plotter_rejects_tampered_config_hash(tmp_path):
    plotter = _load_plotter()
    smoke = _complete_config()
    smoke.update(version="maxcut-approx-smoke-v1", n_values=[8], layer_n=8, depths=[1], samples_per_cell=1, bootstrap_resamples=200)
    smoke["structure_cells"] = smoke["weight_cells"] = [["regular3", "unit"]]
    out = tmp_path / "tampered"
    out.mkdir()
    config_path = _write_saved_run(out, smoke, [_row(0.8, n=8)])
    config_path.write_text(config_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Config hash"):
        plotter.main(config_path, out, allow_incomplete_smoke=True)
    assert not (out / "summary.json").exists()


def test_plotter_source_has_no_solver_simulator_or_runner_imports():
    tree = ast.parse(PLOTTER_PATH.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(
        name.startswith(("cdtransfer.simulator", "cdtransfer.maxcut", "networkx")) or "13_compare_maxcut" in name
        for name in imported
    )
    assert not any(
        (isinstance(node, ast.Name) and node.id == "make_instance")
        or (isinstance(node, ast.Attribute) and node.attr == "make_instance")
        for node in ast.walk(tree)
    )
