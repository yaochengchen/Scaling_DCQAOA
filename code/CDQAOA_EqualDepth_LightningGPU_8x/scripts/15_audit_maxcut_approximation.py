"""Independently audit the completed MaxCut approximation production artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("qaoa_csv", "cd_proposed")
TOLERANCE = 1e-8
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ENVIRONMENT_FIELDS = {
    "python",
    "platform",
    "numpy",
    "networkx",
    "pyscipopt_package_version",
    "scip_engine_version",
    "gurobipy_package_version",
    "gurobi_engine_version",
}


def _load(path: Path, kind):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, kind):
        raise ValueError(f"{path.name} has wrong top-level type")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _cells(config: dict) -> list[tuple[str, str]]:
    raw = config["structure_cells"] + config["weight_cells"]
    cells = list(dict.fromkeys((cell[0], cell[1]) for cell in raw))
    if len(cells) != 7:
        raise ValueError(f"expected 7 unique cells, got {len(cells)}")
    return cells


def _fingerprint(instance: dict) -> str:
    payload = {
        "n": instance["n"],
        "edges": instance["edges"],
        "weights": instance["weights"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _expected_seed(base: int, n: int, family: str, weight: str, sample: int) -> int:
    token = f"{base}:{n}:{family}:{weight}:{sample}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def _summary_seed(base: int, *parts) -> int:
    token = ":".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _bootstrap_mean_ci(values: list[float], *, n_boot: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(n_boot, len(array)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _validate_plotter_independence(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {"solver", "simulator", "runner", "make_instance"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(name.name.split(".")[-1] for name in node.names if name.name.split(".")[-1] in forbidden)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[-1]
            if module in forbidden:
                found.add(module)
            found.update(name.name for name in node.names if name.name in forbidden)
    if found:
        raise ValueError(f"plotter imports forbidden production dependencies: {sorted(found)}")


def _validate_summary(summary: dict, config: dict, records: list[dict]) -> int:
    if summary.get("complete") is not True or summary.get("watermark") is not None:
        raise ValueError("summary is not marked complete production output")
    if summary.get("record_count") != 350 or summary.get("solver_agreement_count") != 350:
        raise ValueError("summary record/solver counts are incomplete")
    figures = summary.get("figures")
    if not isinstance(figures, dict) or figures.get("layout") != [4, 2] or figures.get("watermark") is not None:
        raise ValueError("summary figure metadata is incomplete")
    if figures.get("warning") != "Same-layer comparison; not matched two-qubit-gate budget":
        raise ValueError("summary figure warning is missing")
    expected_keys = {
        (family, weight, n, p, method)
        for family, weight in _cells(config)
        for n in config["n_values"]
        for p in ([1, 2] if n == config["layer_n"] else [1])
        for method in METHODS
    }
    cells = summary.get("cells")
    if not isinstance(cells, list):
        raise ValueError("summary cells must be a list")
    observed = {}
    for item in cells:
        try:
            key = (item["family"], item["weight"], item["n_qubits"], item["p"], item["method"])
        except (KeyError, TypeError) as exc:
            raise ValueError("summary cell is malformed") from exc
        if key in observed or item.get("n") != 10:
            raise ValueError("summary cells are duplicated or incomplete")
        observed[key] = item
    if set(observed) != expected_keys:
        raise ValueError(f"summary cells are incomplete: found {len(observed)} of {len(expected_keys)}")
    grouped: dict[tuple, list[float]] = {key: [] for key in expected_keys}
    for row in records:
        inst = row["instance"]
        for p_text, depth in row["evaluations"].items():
            for method, result in depth.items():
                grouped[(inst["family"], inst["weight_kind"], inst["n"], int(p_text), method)].append(
                    result["approximation_ratio"]
                )
    for key, item in observed.items():
        values = grouped[key]
        if len(values) != 10:
            raise ValueError(f"summary count disagrees with records: {key}")
        family, weight, n, p, method = key
        array = np.asarray(values, dtype=float)
        expected = {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std(ddof=1)),
            "ci95": _bootstrap_mean_ci(
                values,
                n_boot=config["bootstrap_resamples"],
                seed=_summary_seed(config["seed"], family, weight, n, p, method),
            ),
        }
        for field in ("mean", "median", "std"):
            if not math.isclose(
                _finite(item.get(field), f"summary {field}"),
                expected[field],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"summary {field} disagrees with records: {key}")
        ci95 = item.get("ci95")
        if not isinstance(ci95, list) or len(ci95) != 2:
            raise ValueError(f"summary CI is malformed: {key}")
        for observed_bound, expected_bound in zip(ci95, expected["ci95"]):
            if not math.isclose(
                _finite(observed_bound, "summary CI bound"),
                expected_bound,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"summary CI disagrees with deterministic bootstrap: {key}")
    return len(observed)


def _audit(config_path: Path, out_path: Path) -> dict:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    manifest = _load(out_path / "manifest.json", dict)
    environment = _load(out_path / "environment.json", dict)
    records = _load(out_path / "records.json", list)
    summary = _load(out_path / "summary.json", dict)

    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("config hash does not match manifest")
    config_text = manifest.get("config_text")
    if not isinstance(config_text, str) or hashlib.sha256(config_text.encode()).hexdigest() != config_hash:
        raise ValueError("manifest config_text hash is invalid")
    if json.loads(config_text) != config or manifest.get("config") != config:
        raise ValueError("manifest config copies differ from requested config")
    model_path = (ROOT / config["frozen_model"]).resolve()
    source_csv = ROOT / "assets" / "original_transfer_data.csv"
    model_hash, csv_hash = _sha256(model_path), _sha256(source_csv)
    if manifest.get("frozen_model_sha256") != model_hash:
        raise ValueError("frozen model hash does not match manifest/current asset")
    if manifest.get("qaoa_source_csv_sha256") != csv_hash:
        raise ValueError("source CSV hash does not match manifest/current asset")
    model = _load(model_path, dict)
    if model.get("no_target_instance_objective_calls") is not True:
        raise ValueError("frozen model does not certify no target optimization")
    if set(environment) != ENVIRONMENT_FIELDS or any(not isinstance(environment[key], str) or not environment[key] for key in ENVIRONMENT_FIELDS):
        raise ValueError("environment versions are incomplete")

    cells = _cells(config)
    expected_slots = {
        (family, weight, n, sample)
        for family, weight in cells
        for n in config["n_values"]
        for sample in range(config["samples_per_cell"])
    }
    if len(expected_slots) != 350 or manifest.get("expected_records") != 350 or len(records) != 350:
        raise ValueError("record population is not 7 cells * 5 n * 10 samples = 350")

    ids: set[str] = set()
    fingerprints: set[str] = set()
    slots: set[tuple] = set()
    evaluation_count = p1_evaluations = p2_n12_evaluations = 0
    ratio_min = math.inf
    ratio_max = -math.inf
    minimum_weight = math.inf
    max_solver_objective_diff = 0.0
    max_energy_cut_residual = 0.0
    max_ratio_identity_residual = 0.0
    for row in records:
        case_id, fingerprint, instance = row.get("id"), row.get("fingerprint"), row.get("instance")
        if not isinstance(case_id, str) or case_id in ids:
            raise ValueError("record ids are not 350 unique strings")
        if not isinstance(fingerprint, str) or not HEX64.fullmatch(fingerprint) or fingerprint in fingerprints:
            raise ValueError("record fingerprints are not 350 unique SHA-256 strings")
        match = re.fullmatch(r"(.+)-(.+)-n(\d+)-sample(\d{2})", case_id)
        if not match or not isinstance(instance, dict):
            raise ValueError(f"record identity is malformed: {case_id}")
        family, weight, n, sample = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
        slot = (family, weight, n, sample)
        if slot not in expected_slots or slot in slots:
            raise ValueError(f"record slot is unexpected or duplicated: {case_id}")
        expected_metadata = (n, family, weight, _expected_seed(config["seed"], n, family, weight, sample), "comparison")
        actual_metadata = (instance.get("n"), instance.get("family"), instance.get("weight_kind"), instance.get("seed"), instance.get("split"))
        if actual_metadata != expected_metadata:
            raise ValueError(f"record metadata disagrees with deterministic case: {case_id}")
        edges, raw_weights = instance.get("edges"), instance.get("weights")
        if not isinstance(edges, list) or not isinstance(raw_weights, list) or not edges or len(edges) != len(raw_weights):
            raise ValueError(f"edge/weight arrays are malformed: {case_id}")
        if any(not isinstance(edge, list) or len(edge) != 2 or any(type(v) is not int or v < 0 or v >= n for v in edge) or edge[0] >= edge[1] for edge in edges):
            raise ValueError(f"edges are invalid: {case_id}")
        if len({tuple(edge) for edge in edges}) != len(edges) or edges != sorted(edges):
            raise ValueError(f"edges are duplicated or unsorted: {case_id}")
        weights = [_finite(value, "edge weight") for value in raw_weights]
        if any(value <= 0 for value in weights):
            raise ValueError(f"weights are not strictly positive: {case_id}")
        minimum_weight = min(minimum_weight, min(weights))
        if _fingerprint(instance) != fingerprint:
            raise ValueError(f"fingerprint disagrees with saved graph/weights: {case_id}")

        exact = row.get("exact")
        if not isinstance(exact, dict) or exact.get("scip_status") != "optimal" or exact.get("gurobi_status") != "optimal":
            raise ValueError(f"solver status is not optimal: {case_id}")
        scip = _finite(exact.get("scip_objective"), "SCIP objective")
        gurobi = _finite(exact.get("gurobi_objective"), "Gurobi objective")
        optimum = _finite(exact.get("optimum"), "optimum")
        max_solver_objective_diff = max(max_solver_objective_diff, abs(scip - gurobi))
        if optimum <= 0 or max(abs(scip - gurobi), abs(optimum - scip), abs(optimum - gurobi)) > TOLERANCE:
            raise ValueError(f"solver objectives/optimum disagree: {case_id}")
        for seconds in ("scip_seconds", "gurobi_seconds"):
            if _finite(exact.get(seconds), seconds) < 0:
                raise ValueError(f"solver time is negative: {case_id}")

        expected_depths = {"1", "2"} if n == config["layer_n"] else {"1"}
        evaluations = row.get("evaluations")
        if not isinstance(evaluations, dict) or set(evaluations) != expected_depths:
            raise ValueError(f"depth coverage is not exact: {case_id}")
        for p_text, depth_results in evaluations.items():
            if not isinstance(depth_results, dict) or set(depth_results) != set(METHODS):
                raise ValueError(f"only qaoa_csv/cd_proposed may be evaluated; no target-optimized method: {case_id}")
            p = int(p_text)
            for method in METHODS:
                result = depth_results[method]
                if not isinstance(result, dict):
                    raise ValueError(f"evaluation is malformed: {case_id}/{p}/{method}")
                energy = _finite(result.get("ising_energy"), "Ising energy")
                cut = _finite(result.get("expected_cut"), "expected cut")
                ratio = _finite(result.get("approximation_ratio"), "approximation ratio")
                if not 0 <= ratio <= 1:
                    raise ValueError(f"approximation ratio is outside [0,1]: {case_id}/{p}/{method}")
                energy_cut_residual = abs(cut - 0.5 * (sum(weights) - energy))
                ratio_identity_residual = abs(ratio - cut / optimum)
                ratio_min = min(ratio_min, ratio)
                ratio_max = max(ratio_max, ratio)
                max_energy_cut_residual = max(max_energy_cut_residual, energy_cut_residual)
                max_ratio_identity_residual = max(max_ratio_identity_residual, ratio_identity_residual)
                if energy_cut_residual > TOLERANCE:
                    raise ValueError(f"cut/energy identity fails: {case_id}/{p}/{method}")
                if ratio_identity_residual > TOLERANCE:
                    raise ValueError(f"ratio/cut/optimum identity fails: {case_id}/{p}/{method}")
                angles = result.get("angles")
                if not isinstance(angles, dict) or set(angles) != {"gamma", "beta", "alpha"}:
                    raise ValueError(f"angles are malformed: {case_id}/{p}/{method}")
                for vector in angles.values():
                    if not isinstance(vector, list) or len(vector) != p:
                        raise ValueError(f"angle depth is malformed: {case_id}/{p}/{method}")
                    for value in vector:
                        _finite(value, "angle")
                evaluation_count += 1
                if p == 1:
                    p1_evaluations += 1
                else:
                    p2_n12_evaluations += 1
        ids.add(case_id)
        fingerprints.add(fingerprint)
        slots.add(slot)

    if slots != expected_slots or len(ids) != 350 or len(fingerprints) != 350:
        raise ValueError("record ids/fingerprints/slots are incomplete")
    if (evaluation_count, p1_evaluations, p2_n12_evaluations) != (840, 700, 140):
        raise ValueError("method-depth evaluation coverage is not 840 = 700 p1 + 140 p2@n12")
    summary_cells = _validate_summary(summary, config, records)
    if summary.get("config_sha256") != config_hash:
        raise ValueError("summary config hash disagrees with manifest/current config")

    artifacts = [
        out_path / "figures" / "fig01_graph_structures_same_layer.png",
        out_path / "figures" / "fig02_weight_distributions_same_layer.png",
        out_path / "EXPERIMENT_REPORT.md",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in artifacts):
        raise ValueError("two PNGs and experiment report must exist and be nonempty")
    report_text = artifacts[-1].read_text(encoding="utf-8")
    if "INCOMPLETE SMOKE" in report_text or "same-layer" not in report_text.lower():
        raise ValueError("experiment report is incomplete or lacks same-layer warning")
    _validate_plotter_independence(ROOT / "scripts" / "14_plot_maxcut_approximation.py")
    return {
        "status": "PASS",
        "record_count": 350,
        "unique_id_count": len(ids),
        "unique_fingerprint_count": len(fingerprints),
        "cell_n_sample_slots": len(slots),
        "evaluation_count": evaluation_count,
        "p1_evaluations": p1_evaluations,
        "p2_n12_evaluations": p2_n12_evaluations,
        "summary_cell_count": summary_cells,
        "config_sha256": config_hash,
        "frozen_model_sha256": model_hash,
        "qaoa_source_csv_sha256": csv_hash,
        "environment": environment,
        "extrema": {
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "minimum_weight": minimum_weight,
            "max_solver_objective_diff": max_solver_objective_diff,
            "max_energy_cut_residual": max_energy_cut_residual,
            "max_ratio_identity_residual": max_ratio_identity_residual,
        },
        "artifact_sha256": {path.name: _sha256(path) for path in artifacts},
    }


def _render(result: dict) -> str:
    env = result["environment"]
    extrema = result["extrema"]
    artifacts = "\n".join(f"- `{name}`: `{digest}`" for name, digest in result["artifact_sha256"].items())
    versions = "\n".join(f"- {name}: `{env[name]}`" for name in sorted(env))
    return f"""# MaxCut Approximation Production Audit

## Independent checks

- 350/350 unique record IDs and 350/350 unique graph fingerprints.
- 7 cells × 5 qubit counts × 10 samples = 350 exact slots.
- 840 method-depth evaluations: 700 at p=1 and 140 at p=2 only for n=12.
- Methods are exactly `qaoa_csv` and `cd_proposed`; the frozen model certifies no target-instance objective calls.
- All solver statuses are optimal; paired objectives and the saved optimum agree within 1e-8.
- All weights and numeric results are finite, all weights are positive, and all ratios are in [0,1].
- Every expected-cut/Ising-energy and ratio/cut/optimum identity holds within 1e-8.
- Config, frozen-model, and source-CSV SHA-256 values match the manifest and current assets.
- Summary is complete with {result['summary_cell_count']} exact cells; two PNGs and the experiment report exist.
- Plotter AST has no imports of solver, simulator, runner, or `make_instance`.

## Numeric extrema and residuals

- Ratio minimum / maximum: `{extrema['ratio_min']:.17g}` / `{extrema['ratio_max']:.17g}`
- Minimum positive edge weight: `{extrema['minimum_weight']:.17g}`
- Maximum |SCIP − Gurobi objective|: `{extrema['max_solver_objective_diff']:.17g}`
- Maximum |expected cut − energy-derived cut|: `{extrema['max_energy_cut_residual']:.17g}`
- Maximum |ratio − expected cut / optimum|: `{extrema['max_ratio_identity_residual']:.17g}`
- Tolerance for all identity/agreement checks: `1e-8` (absolute)

## Input hashes

- Config: `{result['config_sha256']}`
- Frozen model: `{result['frozen_model_sha256']}`
- Source CSV: `{result['qaoa_source_csv_sha256']}`

## Environment

{versions}

## Artifact hashes

{artifacts}

## Final status

FINAL: PASS
"""


def main(config_path: str | Path, out_path: str | Path, report_path: str | Path) -> dict:
    config_path, out_path, report_path = Path(config_path), Path(out_path), Path(report_path)
    report_path.unlink(missing_ok=True)
    result = _audit(config_path.resolve(), out_path.resolve())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(_render(result), encoding="utf-8")
    temporary.replace(report_path)
    return result


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = main(args.config, args.out, args.report)
    except Exception as exc:
        print(f"AUDIT FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
