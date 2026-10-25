from __future__ import annotations

import json
import importlib.util
import importlib.metadata
import copy
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

import cdtransfer.instances as instances_module
from cdtransfer.instances import IsingInstance, make_instance
from cdtransfer.maxcut import (
    VerifiedMaxCut,
    expected_cut_from_ising,
    solve_maxcut_verified,
)
from scripts.maxcut_comparison_helpers import (
    deterministic_seed,
    evaluate_instance,
    unique_cells,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def instance(n, edges, weights):
    return IsingInstance(
        n,
        np.asarray(edges, dtype=np.int32),
        np.asarray(weights, dtype=float),
        family="test",
        weight_kind="positive",
    )


def test_random_tree_is_deterministic_and_has_tree_metadata():
    first = make_instance(12, "tree", "unit", 1234, "comparison")
    second = make_instance(12, "tree", "unit", 1234, "comparison")

    assert first.fingerprint == second.fingerprint
    assert first.m == first.n - 1
    assert first.family == "tree"
    graph = nx.Graph()
    graph.add_nodes_from(range(first.n))
    graph.add_edges_from(first.edges)
    assert nx.is_tree(graph)


def test_random_tree_seeds_produce_distinct_fingerprints():
    fingerprints = {
        make_instance(12, "tree", "unit", seed, "comparison").fingerprint
        for seed in range(10)
    }

    assert len(fingerprints) == 10


def test_random_tree_falls_back_to_legacy_networkx_api(monkeypatch):
    calls = []

    def fake_random_tree(n, seed):
        calls.append((n, seed))
        return nx.path_graph(n)

    monkeypatch.delattr(instances_module.nx, "random_labeled_tree")
    monkeypatch.setattr(
        instances_module.nx, "random_tree", fake_random_tree, raising=False
    )

    inst = make_instance(12, "tree", "unit", 1234, "comparison")
    graph = nx.Graph()
    graph.add_nodes_from(range(inst.n))
    graph.add_edges_from(inst.edges)

    assert calls == [(12, 1234)]
    assert nx.is_tree(graph)


@pytest.mark.parametrize(
    ("inst", "optimum"),
    [
        (instance(2, [(0, 1)], [1.7]), 1.7),
        (instance(3, [(0, 1), (1, 2), (0, 2)], [1, 1, 1]), 2.0),
        (instance(4, [(0, 1), (1, 2), (2, 3)], [0.5, 1.2, 2.3]), 4.0),
    ],
)
def test_scip_and_gurobi_verify_known_maxcut(inst, optimum):
    pyscipopt = pytest.importorskip(
        "pyscipopt", reason="PySCIPOpt is an optional MaxCut backend"
    )
    gp = pytest.importorskip(
        "gurobipy", reason="gurobipy is an optional MaxCut backend"
    )
    try:
        result = solve_maxcut_verified(inst)
    except gp.GurobiError as exc:
        if exc.errno == gp.GRB.ERROR_NO_LICENSE:
            pytest.skip(f"Gurobi license unavailable: {exc}")
        raise
    assert result.scip_status == "optimal"
    assert result.gurobi_status == "optimal"
    assert result.optimum == pytest.approx(optimum, abs=1e-8)
    assert result.scip_objective == pytest.approx(
        result.gurobi_objective, abs=1e-8
    )
    scip = pyscipopt.Model()
    assert result.scip_version == ".".join(
        map(
            str,
            (
                scip.getMajorVersion(),
                scip.getMinorVersion(),
                scip.getTechVersion(),
            ),
        )
    )
    assert result.scip_version != pyscipopt.__version__
    assert result.gurobi_version == ".".join(map(str, gp.gurobi.version()))


@pytest.mark.parametrize("weight", [-1.0, 0.0])
def test_rejects_nonpositive_weights(weight):
    inst = instance(3, [(0, 1), (1, 2)], [weight, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        solve_maxcut_verified(inst)


def test_expected_cut_from_ising():
    inst = instance(3, [(0, 1), (1, 2)], [1.0, 2.0])
    assert expected_cut_from_ising(inst, ising_energy=-3.0) == pytest.approx(3.0)
    assert expected_cut_from_ising(inst, ising_energy=3.0) == pytest.approx(0.0)


@pytest.mark.parametrize("energy", [np.inf, -np.inf, np.nan])
def test_expected_cut_from_ising_rejects_nonfinite_energy(energy):
    inst = instance(2, [(0, 1)], [1.0])
    with pytest.raises(ValueError, match="finite"):
        expected_cut_from_ising(inst, ising_energy=energy)


@pytest.mark.parametrize(
    ("scip_status", "gurobi_status"),
    [("timelimit", "optimal"), ("optimal", "status-9")],
)
def test_rejects_nonoptimal_solver_status(
    monkeypatch, scip_status, gurobi_status
):
    monkeypatch.setattr(
        "cdtransfer.maxcut._solve_scip",
        lambda _: (3.0, scip_status, 0.01, "test"),
    )
    monkeypatch.setattr(
        "cdtransfer.maxcut._solve_gurobi",
        lambda _: (3.0, gurobi_status, 0.01, "test"),
    )

    with pytest.raises(RuntimeError, match="Exact solver failure"):
        solve_maxcut_verified(instance(2, [(0, 1)], [1.0]))


def test_rejects_solver_objective_disagreement(monkeypatch):
    monkeypatch.setattr(
        "cdtransfer.maxcut._solve_scip",
        lambda _: (1.0, "optimal", 0.01, "test"),
    )
    monkeypatch.setattr(
        "cdtransfer.maxcut._solve_gurobi",
        lambda _: (1.1, "optimal", 0.01, "test"),
    )

    with pytest.raises(RuntimeError, match="Exact solver disagreement"):
        solve_maxcut_verified(instance(2, [(0, 1)], [1.0]))


def test_rejects_nonfinite_solver_objectives(monkeypatch):
    monkeypatch.setattr(
        "cdtransfer.maxcut._solve_scip",
        lambda _: (np.inf, "optimal", 0.01, "10.0.2"),
    )
    monkeypatch.setattr(
        "cdtransfer.maxcut._solve_gurobi",
        lambda _: (np.inf, "optimal", 0.01, "13.0.3"),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        solve_maxcut_verified(instance(2, [(0, 1)], [1.0]))


def test_structure_and_weight_views_share_regular3_unit_cell():
    structure = [
        ("tree", "unit"),
        ("regular3", "unit"),
        ("er3", "unit"),
        ("er6", "unit"),
    ]
    weights = [
        ("regular3", "unit"),
        ("regular3", "uniform"),
        ("regular3", "bimodal"),
        ("regular3", "lognormal"),
    ]

    assert unique_cells(structure, weights) == [
        ("tree", "unit"),
        ("regular3", "unit"),
        ("er3", "unit"),
        ("er6", "unit"),
        ("regular3", "uniform"),
        ("regular3", "bimodal"),
        ("regular3", "lognormal"),
    ]


def test_deterministic_seed_is_stable():
    first = deterministic_seed(20261012, 12, "regular3", "unit", 3)
    second = deterministic_seed(20261012, 12, "regular3", "unit", 3)

    assert first == second
    assert 0 <= first < 2**32


def test_deterministic_seed_changes_when_any_input_changes():
    seeds = {
        deterministic_seed(20261012, 12, "regular3", "unit", 3),
        deterministic_seed(20261013, 12, "regular3", "unit", 3),
        deterministic_seed(20261012, 14, "regular3", "unit", 3),
        deterministic_seed(20261012, 12, "er3", "unit", 3),
        deterministic_seed(20261012, 12, "regular3", "uniform", 3),
        deterministic_seed(20261012, 12, "regular3", "unit", 4),
    }

    assert len(seeds) == 6


def test_current_formula_config_defines_350_unique_instances():
    config_path = PROJECT_ROOT / "configs" / "maxcut_approximation_current_formula.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cells = unique_cells(config["structure_cells"], config["weight_cells"])
    size_cells = len(cells) * len(config["n_values"])
    instances = size_cells * config["samples_per_cell"]

    assert len(cells) == 7
    assert size_cells == 35
    assert instances == 350


def test_evaluate_instance_records_both_frozen_methods(monkeypatch):
    inst = make_instance(8, "regular3", "unit", 99, split="comparison")
    verified = VerifiedMaxCut(
        12.0, 12.0, 12.0, "optimal", "optimal", 0.01, 0.01, "test", "test"
    )
    monkeypatch.setattr(
        "scripts.maxcut_comparison_helpers.solve_maxcut_verified",
        lambda _: verified,
    )
    model = {
        "csv_sign": -1,
        "alpha_degree_power": -0.5,
        "alpha_multiplier": 1.0,
        "templates": {
            "1": {
                "qaoa": {
                    "gamma_unit": [-0.472548621783316],
                    "beta": [0.3642612890208946],
                    "alpha_unit": [0.0],
                    "reference_degree": 3.0,
                },
                "cd": {
                    "gamma_unit": [-0.4956227054284411],
                    "beta": [0.46094955918648306],
                    "alpha_unit": [-0.17673179131513422],
                    "reference_degree": 3.0,
                }
            }
        },
    }

    row = evaluate_instance("case", inst, [1], model)

    assert row["id"] == "case"
    assert row["exact"]["optimum"] == 12.0
    assert set(row["evaluations"]["1"]) == {"qaoa_csv", "cd_proposed"}
    for result in row["evaluations"]["1"].values():
        assert set(result["angles"]) == {"gamma", "beta", "alpha"}
        assert np.isfinite(result["ising_energy"])
        assert np.isfinite(result["expected_cut"])
        assert 0.0 <= result["approximation_ratio"] <= 1.0


def _load_comparison_runner():
    path = PROJECT_ROOT / "scripts" / "13_compare_maxcut_approximation.py"
    spec = importlib.util.spec_from_file_location("maxcut_comparison_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_runner_writes_manifest_environment_and_resumes(
    tmp_path, monkeypatch
):
    runner = _load_comparison_runner()
    config = {
        "version": "runner-smoke-test",
        "seed": 7,
        "n_values": [8],
        "layer_n": 8,
        "depths": [1],
        "samples_per_cell": 1,
        "structure_cells": [["regular3", "unit"]],
        "weight_cells": [["regular3", "unit"]],
        "methods": ["qaoa_csv", "cd_proposed"],
        "frozen_model": "assets/CD_TRANSFER_V1_FROZEN_MODEL.json",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_path = tmp_path / "out"
    calls = []
    verified = VerifiedMaxCut(
        12.0, 12.0, 12.0, "optimal", "optimal", 0.01, 0.01, "test", "test"
    )

    def fake_solver(inst):
        calls.append(inst.fingerprint)
        return verified

    monkeypatch.setattr(
        "scripts.maxcut_comparison_helpers.solve_maxcut_verified", fake_solver
    )

    assert runner.main(config_path, out_path) == 0
    manifest = json.loads((out_path / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads(
        (out_path / "environment.json").read_text(encoding="utf-8")
    )
    records = json.loads((out_path / "records.json").read_text(encoding="utf-8"))
    assert manifest["config"] == config
    assert len(manifest["config_sha256"]) == 64
    assert {
        "python",
        "platform",
        "numpy",
        "networkx",
        "pyscipopt_package_version",
        "scip_engine_version",
        "gurobipy_package_version",
        "gurobi_engine_version",
    } <= set(environment)
    assert "pyscipopt" not in environment
    assert "gurobipy" not in environment
    assert environment["pyscipopt_package_version"] == importlib.metadata.version(
        "pyscipopt"
    )
    assert environment["gurobipy_package_version"] == importlib.metadata.version(
        "gurobipy"
    )
    from pyscipopt import Model
    import gurobipy as gp

    scip = Model()
    expected_scip = ".".join(
        map(
            str,
            (
                scip.getMajorVersion(),
                scip.getMinorVersion(),
                scip.getTechVersion(),
            ),
        )
    )
    assert environment["scip_engine_version"] == expected_scip
    assert environment["gurobi_engine_version"] == ".".join(
        map(str, gp.gurobi.version())
    )
    assert len(records) == 1
    assert records[0]["id"] == "regular3-unit-n8-sample00"
    assert len(calls) == 1

    assert runner.main(config_path, out_path) == 0
    resumed = json.loads((out_path / "records.json").read_text(encoding="utf-8"))
    assert resumed == records
    assert len(calls) == 1


def test_comparison_runner_resolves_cli_paths_from_scripts_cwd(
    tmp_path, monkeypatch
):
    runner = _load_comparison_runner()
    project = tmp_path / "project"
    scripts_dir = project / "scripts"
    scripts_dir.mkdir(parents=True)
    captured = []
    monkeypatch.setattr(
        runner,
        "main",
        lambda config_path, out_path: captured.append((config_path, out_path)) or 0,
    )
    monkeypatch.chdir(scripts_dir)

    assert runner.cli(
        ["--config", "../configs/config.json", "--out", "../results/out"]
    ) == 0
    assert captured == [
        (
            (project / "configs" / "config.json").resolve(),
            (project / "results" / "out").resolve(),
        )
    ]


def test_comparison_runner_main_resolves_project_relative_paths_from_any_cwd(
    tmp_path, monkeypatch
):
    runner = _load_comparison_runner()
    project = tmp_path / "project"
    config_dir = project / "configs"
    assets_dir = project / "assets"
    unrelated_cwd = tmp_path / "elsewhere"
    config_dir.mkdir(parents=True)
    assets_dir.mkdir()
    unrelated_cwd.mkdir()
    config = {
        "version": "project-relative-main-test",
        "seed": 7,
        "n_values": [8],
        "layer_n": 8,
        "depths": [1],
        "samples_per_cell": 1,
        "structure_cells": [["regular3", "unit"]],
        "weight_cells": [["regular3", "unit"]],
        "methods": ["qaoa_csv", "cd_proposed"],
        "frozen_model": "assets/CD_TRANSFER_V1_FROZEN_MODEL.json",
    }
    (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model_path = assets_dir / "CD_TRANSFER_V1_FROZEN_MODEL.json"
    model_path.write_bytes(
        (PROJECT_ROOT / "assets" / "CD_TRANSFER_V1_FROZEN_MODEL.json").read_bytes()
    )
    source_csv = assets_dir / "original_transfer_data.csv"
    source_csv.write_bytes(
        (PROJECT_ROOT / "assets" / "original_transfer_data.csv").read_bytes()
    )
    verified = VerifiedMaxCut(
        12.0, 12.0, 12.0, "optimal", "optimal", 0.01, 0.01, "test", "test"
    )
    monkeypatch.setattr(
        "scripts.maxcut_comparison_helpers.solve_maxcut_verified",
        lambda _: verified,
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    monkeypatch.setattr(runner, "SOURCE_CSV", source_csv)
    monkeypatch.chdir(unrelated_cwd)

    assert runner.main("configs/config.json", "results/out") == 0
    assert (project / "results" / "out" / "records.json").exists()


def test_comparison_runner_rejects_incomplete_saved_record(tmp_path, monkeypatch):
    runner = _load_comparison_runner()
    config = {
        "version": "corrupt-resume-test",
        "seed": 7,
        "n_values": [8],
        "layer_n": 8,
        "depths": [1],
        "samples_per_cell": 1,
        "structure_cells": [["regular3", "unit"]],
        "weight_cells": [["regular3", "unit"]],
        "methods": ["qaoa_csv", "cd_proposed"],
        "frozen_model": "assets/CD_TRANSFER_V1_FROZEN_MODEL.json",
        "bootstrap_resamples": 10,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_path = tmp_path / "out"
    verified = VerifiedMaxCut(
        12.0, 12.0, 12.0, "optimal", "optimal", 0.01, 0.01, "10.0.2", "13.0.3"
    )
    monkeypatch.setattr(
        "scripts.maxcut_comparison_helpers.solve_maxcut_verified",
        lambda _: verified,
    )
    runner.main(config_path, out_path)
    record = json.loads((out_path / "records.json").read_text(encoding="utf-8"))[0]
    (out_path / "records.json").write_text(
        json.dumps({"not": "a list"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="list"):
        runner.main(config_path, out_path)
    (out_path / "records.json").write_text(
        json.dumps([{"id": record["id"], "fingerprint": record["fingerprint"]}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing|required"):
        runner.main(config_path, out_path)


@pytest.mark.parametrize("tamper", ["model", "csv", "manifest"])
def test_comparison_runner_rejects_manifest_input_tampering(
    tmp_path, monkeypatch, tamper
):
    runner = _load_comparison_runner()
    model_path = tmp_path / "model.json"
    model_path.write_bytes(
        (PROJECT_ROOT / "assets" / "CD_TRANSFER_V1_FROZEN_MODEL.json").read_bytes()
    )
    csv_path = tmp_path / "source.csv"
    csv_path.write_bytes(
        (PROJECT_ROOT / "assets" / "original_transfer_data.csv").read_bytes()
    )
    monkeypatch.setattr(runner, "SOURCE_CSV", csv_path, raising=False)
    config = {
        "version": "manifest-integrity-test",
        "seed": 7,
        "n_values": [8],
        "layer_n": 8,
        "depths": [1],
        "samples_per_cell": 1,
        "structure_cells": [["regular3", "unit"]],
        "weight_cells": [["regular3", "unit"]],
        "methods": ["qaoa_csv", "cd_proposed"],
        "frozen_model": str(model_path),
        "bootstrap_resamples": 10,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_path = tmp_path / "out"
    verified = VerifiedMaxCut(
        12.0, 12.0, 12.0, "optimal", "optimal", 0.01, 0.01, "10.0.2", "13.0.3"
    )
    monkeypatch.setattr(
        "scripts.maxcut_comparison_helpers.solve_maxcut_verified",
        lambda _: verified,
    )
    runner.main(config_path, out_path)

    if tamper == "model":
        model_path.write_text(model_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif tamper == "csv":
        csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        manifest_path = out_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config_text"] += " "
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="[Mm]anifest|hash|changed"):
        runner.main(config_path, out_path)


@pytest.mark.parametrize("key", ["seed", "n_values", "layer_n", "depths", "samples_per_cell", "bootstrap_resamples"])
def test_comparison_runner_rejects_fractional_integer_config(key):
    runner = _load_comparison_runner()
    config = {
        "seed": 7,
        "n_values": [8],
        "layer_n": 8,
        "depths": [1],
        "samples_per_cell": 1,
        "structure_cells": [["regular3", "unit"]],
        "weight_cells": [["regular3", "unit"]],
        "methods": ["qaoa_csv", "cd_proposed"],
        "frozen_model": "assets/CD_TRANSFER_V1_FROZEN_MODEL.json",
        "bootstrap_resamples": 10,
    }
    config[key] = [1.5] if key in {"n_values", "depths"} else 1.5

    with pytest.raises(ValueError, match="integer"):
        runner._validate_config(config)


def _valid_saved_record(monkeypatch):
    runner = _load_comparison_runner()
    inst = make_instance(8, "regular3", "unit", 99, split="comparison")
    verified = VerifiedMaxCut(
        12.0,
        12.0,
        12.0,
        "optimal",
        "optimal",
        0.01,
        0.02,
        "10.0.2",
        "13.0.3",
    )
    monkeypatch.setattr(
        "scripts.maxcut_comparison_helpers.solve_maxcut_verified",
        lambda _: verified,
    )
    model = json.loads(
        (PROJECT_ROOT / "assets" / "CD_TRANSFER_V1_FROZEN_MODEL.json").read_text(
            encoding="utf-8"
        )
    )
    record = evaluate_instance("case", inst, [1], model)
    config = {
        "layer_n": 8,
        "depths": [1],
        "methods": ["qaoa_csv", "cd_proposed"],
    }
    return runner, inst, record, config


@pytest.mark.parametrize(
    "field", ["scip_seconds", "gurobi_seconds", "scip_version", "gurobi_version"]
)
def test_saved_record_requires_all_solver_metadata(monkeypatch, field):
    runner, inst, record, config = _valid_saved_record(monkeypatch)
    del record["exact"][field]

    with pytest.raises(ValueError, match="exact|required|incomplete"):
        runner._validate_saved_record(record, inst, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [("scip_objective", 11.0), ("gurobi_objective", 11.0), ("optimum", 11.0)],
)
def test_saved_record_rejects_inconsistent_exact_values(
    monkeypatch, field, value
):
    runner, inst, record, config = _valid_saved_record(monkeypatch)
    record["exact"][field] = value

    with pytest.raises(ValueError, match="objective|optimum|inconsistent|disagree"):
        runner._validate_saved_record(record, inst, config)


@pytest.mark.parametrize("field", ["expected_cut", "approximation_ratio"])
def test_saved_record_rejects_inconsistent_evaluation_values(monkeypatch, field):
    runner, inst, record, config = _valid_saved_record(monkeypatch)
    record["evaluations"]["1"]["qaoa_csv"][field] += 0.1

    with pytest.raises(ValueError, match="cut|ratio|inconsistent"):
        runner._validate_saved_record(record, inst, config)


@pytest.mark.parametrize("tamper", ["expected_records", "extra", "missing"])
def test_manifest_rejects_any_schema_tampering(tmp_path, tamper):
    runner = _load_comparison_runner()
    config_text = "{}"
    expected = {
        "config": {},
        "config_text": config_text,
        "config_sha256": runner._sha256_bytes(config_text.encode("utf-8")),
        "frozen_model_sha256": "model-hash",
        "qaoa_source_csv_sha256": "csv-hash",
        "expected_records": 1,
    }
    path = tmp_path / "manifest.json"
    runner._ensure_manifest(path, expected)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "expected_records":
        saved["expected_records"] = 2
    elif tamper == "extra":
        saved["unexpected"] = True
    else:
        del saved["expected_records"]
    path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(RuntimeError, match="[Mm]anifest|schema|match"):
        runner._ensure_manifest(path, expected)
