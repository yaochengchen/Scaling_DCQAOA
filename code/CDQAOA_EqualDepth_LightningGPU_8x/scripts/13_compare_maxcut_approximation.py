"""Run the resumable frozen-angle MaxCut approximation comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cdtransfer.instances import (  # noqa: E402
    IsingInstance,
    dump_json,
    features,
    make_instance,
)
from cdtransfer.transfer import SOURCE_CSV  # noqa: E402

try:  # Direct script execution places this directory on sys.path.
    from maxcut_comparison_helpers import (  # type: ignore[import-not-found]
        deterministic_seed,
        evaluate_instance,
        save_records_atomic,
        unique_cells,
    )
except ModuleNotFoundError:  # Importing the numbered entry point in tests.
    from scripts.maxcut_comparison_helpers import (  # noqa: E402
        deterministic_seed,
        evaluate_instance,
        save_records_atomic,
        unique_cells,
    )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _scip_engine_version() -> str:
    from pyscipopt import Model

    model = Model()
    return ".".join(
        map(
            str,
            (
                model.getMajorVersion(),
                model.getMinorVersion(),
                model.getTechVersion(),
            ),
        )
    )


def _gurobi_engine_version() -> str:
    import gurobipy as gp

    return ".".join(map(str, gp.gurobi.version()))


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "networkx": _package_version("networkx"),
        "pyscipopt_package_version": _package_version("pyscipopt"),
        "scip_engine_version": _scip_engine_version(),
        "gurobipy_package_version": _package_version("gurobipy"),
        "gurobi_engine_version": _gurobi_engine_version(),
    }


def _load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("records.json must contain a list")
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every saved record must be an object")
        case_id = record.get("id")
        fingerprint = record.get("fingerprint")
        if not isinstance(case_id, str) or not isinstance(fingerprint, str):
            raise ValueError("Every saved record needs string id and fingerprint")
        if case_id in ids:
            raise RuntimeError(f"Duplicate saved record id: {case_id}")
        if fingerprint in fingerprints:
            raise RuntimeError(f"Duplicate saved fingerprint: {fingerprint}")
        ids.add(case_id)
        fingerprints.add(fingerprint)
    return records


def _json_int(value, name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer in {bounds}")
    return value


def _integer_list(config: dict, name: str, *, minimum: int, maximum: int) -> list[int]:
    value = config.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of JSON integers")
    result = [
        _json_int(item, f"{name}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique integers")
    return result


def _validate_cells(config: dict, name: str) -> list[tuple[str, str]]:
    value = config.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of two-string cells")
    cells = []
    for index, cell in enumerate(value):
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(not isinstance(item, str) or not item for item in cell)
        ):
            raise ValueError(f"{name}[{index}] must be a two-string cell")
        cells.append((cell[0], cell[1]))
    if len(set(cells)) != len(cells):
        raise ValueError(f"{name} must not contain duplicate cells")
    return cells


def _validate_config(config: dict) -> None:
    _json_int(config.get("seed"), "seed")
    n_values = _integer_list(config, "n_values", minimum=2, maximum=23)
    layer_n = _json_int(config.get("layer_n"), "layer_n", minimum=2, maximum=23)
    depths = _integer_list(config, "depths", minimum=1, maximum=2)
    if 1 not in depths or any(p not in (1, 2) for p in depths):
        raise ValueError("depths must contain p=1 and may additionally contain p=2")
    if 2 in depths and layer_n not in n_values:
        raise ValueError("layer_n must occur in n_values when p=2 is configured")
    _json_int(config.get("samples_per_cell"), "samples_per_cell")
    if "bootstrap_resamples" in config:
        _json_int(config["bootstrap_resamples"], "bootstrap_resamples")
    _validate_cells(config, "structure_cells")
    _validate_cells(config, "weight_cells")
    methods = config.get("methods")
    if (
        not isinstance(methods, list)
        or len(methods) != 2
        or len(set(methods)) != 2
        or set(methods) != {"qaoa_csv", "cd_proposed"}
    ):
        raise ValueError("methods must uniquely contain qaoa_csv and cd_proposed")
    if not isinstance(config.get("frozen_model"), str) or not config["frozen_model"]:
        raise ValueError("frozen_model must be a nonempty string path")


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _cli_path(path: str | Path) -> Path:
    return Path(path).resolve()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _ensure_manifest(path: Path, expected: dict) -> None:
    if not path.exists():
        dump_json(path, expected)
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Existing manifest is unreadable") from exc
    if not isinstance(saved, dict):
        raise RuntimeError("Existing manifest must be an object")
    config_text = saved.get("config_text")
    if not isinstance(config_text, str):
        raise RuntimeError("Existing manifest is missing config_text")
    if _sha256_bytes(config_text.encode("utf-8")) != saved.get("config_sha256"):
        raise RuntimeError("Existing manifest config_text hash is invalid")
    try:
        saved_text_config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Existing manifest config_text is invalid JSON") from exc
    if saved.get("config") != saved_text_config:
        raise RuntimeError("Existing manifest config does not match config_text")
    for key in (
        "config_sha256",
        "frozen_model_sha256",
        "qaoa_source_csv_sha256",
    ):
        if saved.get(key) != expected[key]:
            raise RuntimeError(f"Manifest input hash changed: {key}")
    if saved != expected:
        raise RuntimeError("Existing manifest schema or values do not match expected")


def _ensure_environment(path: Path) -> None:
    current = _environment()
    if not path.exists():
        dump_json(path, current)
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Existing environment metadata is unreadable") from exc
    if saved != current:
        raise RuntimeError("Existing environment metadata does not match this runtime")


def _finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate_saved_record(record: dict, expected_inst: IsingInstance, config: dict) -> None:
    required = {
        "id",
        "fingerprint",
        "instance",
        "features",
        "exact",
        "evaluations",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Saved record is missing required fields: {missing}")
    if not isinstance(record["instance"], dict):
        raise ValueError(f"Saved record {record['id']} instance must be an object")
    try:
        rebuilt = IsingInstance.from_dict(record["instance"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Saved record {record['id']} has an invalid instance") from exc
    if rebuilt.fingerprint != record["fingerprint"]:
        raise RuntimeError(f"Saved record {record['id']} instance fingerprint is corrupt")
    if record["fingerprint"] != expected_inst.fingerprint:
        raise RuntimeError(f"Fingerprint mismatch for saved record: {record['id']}")
    if record["instance"] != expected_inst.to_dict():
        raise RuntimeError(f"Saved instance metadata mismatch for: {record['id']}")

    saved_features = record["features"]
    expected_features = features(rebuilt)
    if not isinstance(saved_features, dict) or set(saved_features) != set(expected_features):
        raise ValueError(f"Saved record {record['id']} features are incomplete")
    for key, expected_value in expected_features.items():
        value = saved_features[key]
        if not _finite_number(value) or not math.isclose(
            value, expected_value, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"Saved record {record['id']} feature {key} is invalid")

    exact = record["exact"]
    exact_required = {
        "optimum",
        "scip_objective",
        "gurobi_objective",
        "scip_status",
        "gurobi_status",
        "scip_seconds",
        "gurobi_seconds",
        "scip_version",
        "gurobi_version",
    }
    if not isinstance(exact, dict) or not exact_required.issubset(exact):
        raise ValueError(f"Saved record {record['id']} exact result is incomplete")
    exact_values = ("optimum", "scip_objective", "gurobi_objective")
    if any(not _finite_number(exact[key]) for key in exact_values):
        raise ValueError(f"Saved record {record['id']} exact objectives must be finite")
    if exact.get("scip_status") != "optimal" or exact.get("gurobi_status") != "optimal":
        raise ValueError(f"Saved record {record['id']} exact statuses must be optimal")
    for key in ("scip_seconds", "gurobi_seconds"):
        if not _finite_number(exact[key]) or exact[key] < 0:
            raise ValueError(f"Saved record {record['id']} exact {key} must be nonnegative")
    for key in ("scip_version", "gurobi_version"):
        if not isinstance(exact[key], str) or not exact[key].strip():
            raise ValueError(f"Saved record {record['id']} exact {key} must be nonempty")
    scip_objective = exact["scip_objective"]
    gurobi_objective = exact["gurobi_objective"]
    optimum = exact["optimum"]
    if not math.isclose(
        scip_objective, gurobi_objective, rel_tol=1e-8, abs_tol=1e-8
    ):
        raise ValueError(f"Saved record {record['id']} exact objectives disagree")
    mean_objective = 0.5 * (scip_objective + gurobi_objective)
    if optimum <= 0 or any(
        not math.isclose(optimum, value, rel_tol=1e-8, abs_tol=1e-8)
        for value in (scip_objective, gurobi_objective, mean_objective)
    ):
        raise ValueError(f"Saved record {record['id']} optimum is inconsistent")

    evaluations = record["evaluations"]
    if not isinstance(evaluations, dict):
        raise ValueError(f"Saved record {record['id']} evaluations must be an object")
    required_depths = [1]
    if rebuilt.n == config["layer_n"] and 2 in config["depths"]:
        required_depths.append(2)
    for p in required_depths:
        depth = evaluations.get(str(p))
        if not isinstance(depth, dict):
            raise ValueError(f"Saved record {record['id']} is missing required depth p={p}")
        for method in config["methods"]:
            result = depth.get(method)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Saved record {record['id']} is missing required method {method} at p={p}"
                )
            for key in ("ising_energy", "expected_cut", "approximation_ratio"):
                if key not in result or not _finite_number(result[key]):
                    raise ValueError(
                        f"Saved record {record['id']} {method} {key} must be finite"
                    )
            ratio = result["approximation_ratio"]
            if ratio < -1e-10 or ratio > 1.0 + 1e-10:
                raise ValueError(
                    f"Saved record {record['id']} {method} ratio is outside [0,1]"
                )
            recomputed_cut = 0.5 * (
                sum(float(weight) for weight in rebuilt.weights)
                - result["ising_energy"]
            )
            if not math.isclose(
                result["expected_cut"], recomputed_cut, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Saved record {record['id']} {method} expected cut is inconsistent"
                )
            recomputed_ratio = result["expected_cut"] / optimum
            if not math.isclose(
                ratio, recomputed_ratio, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Saved record {record['id']} {method} ratio is inconsistent"
                )
            angles = result.get("angles")
            if not isinstance(angles, dict):
                raise ValueError(f"Saved record {record['id']} {method} angles are missing")
            for name in ("gamma", "beta", "alpha"):
                vector = angles.get(name)
                if (
                    not isinstance(vector, list)
                    or len(vector) != p
                    or any(not _finite_number(value) for value in vector)
                ):
                    raise ValueError(
                        f"Saved record {record['id']} {method} angle {name} is invalid"
                    )


def main(config_path: str | Path, out_path: str | Path) -> int:
    config_path = _project_path(config_path)
    out_path = _project_path(out_path)
    raw_config = config_path.read_bytes()
    config_hash = _sha256_bytes(raw_config)
    config = json.loads(raw_config.decode("utf-8"))
    _validate_config(config)

    cells = unique_cells(config["structure_cells"], config["weight_cells"])
    expected_records = (
        len(cells) * len(config["n_values"]) * config["samples_per_cell"]
    )
    model_path = _project_path(config["frozen_model"])
    manifest = {
        "config": config,
        "config_text": raw_config.decode("utf-8"),
        "config_sha256": config_hash,
        "frozen_model_sha256": _sha256_file(model_path),
        "qaoa_source_csv_sha256": _sha256_file(Path(SOURCE_CSV)),
        "expected_records": expected_records,
    }
    _ensure_manifest(out_path / "manifest.json", manifest)
    _ensure_environment(out_path / "environment.json")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    required_depths = {str(p) for p in config["depths"]}
    missing = sorted(required_depths.difference(model.get("templates", {})))
    if missing:
        raise ValueError(f"Frozen model lacks templates for depths: {missing}")

    cases = []
    fingerprints: dict[str, str] = {}
    ids: set[str] = set()
    for family, weight in cells:
        for n_value in config["n_values"]:
            n = n_value
            for sample in range(config["samples_per_cell"]):
                case_id = f"{family}-{weight}-n{n}-sample{sample:02d}"
                if case_id in ids:
                    raise RuntimeError(f"Duplicate generated case id: {case_id}")
                ids.add(case_id)
                seed = deterministic_seed(
                    config["seed"], n, family, weight, sample
                )
                inst = make_instance(n, family, weight, seed, split="comparison")
                previous_id = fingerprints.get(inst.fingerprint)
                if previous_id is not None:
                    raise RuntimeError(
                        "Duplicate generated fingerprint for different ids: "
                        f"{previous_id} and {case_id}"
                    )
                fingerprints[inst.fingerprint] = case_id
                cases.append((case_id, inst))

    records_path = out_path / "records.json"
    records = _load_existing_records(records_path)
    completed = {record["id"]: record for record in records}
    generated_fingerprints = {case_id: inst.fingerprint for case_id, inst in cases}
    for case_id, record in completed.items():
        expected_fingerprint = generated_fingerprints.get(case_id)
        if expected_fingerprint is None:
            raise RuntimeError(f"Saved record id is not present in config: {case_id}")
        if record["fingerprint"] != expected_fingerprint:
            raise RuntimeError(f"Fingerprint mismatch for saved record: {case_id}")
        expected_inst = next(inst for generated_id, inst in cases if generated_id == case_id)
        _validate_saved_record(record, expected_inst, config)

    layer_n = config["layer_n"]
    configured_depths = config["depths"]
    for case_id, inst in cases:
        if case_id in completed:
            continue
        depths = [p for p in configured_depths if p == 1 or inst.n == layer_n]
        record = evaluate_instance(case_id, inst, depths, model)
        records.append(record)
        save_records_atomic(records_path, records)
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def cli(argv=None) -> int:
    args = _parse_args(argv)
    return main(_cli_path(args.config), _cli_path(args.out))


if __name__ == "__main__":
    raise SystemExit(cli())
