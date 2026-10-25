"""Validate, summarize, and plot saved MaxCut approximation records only.

This module intentionally has no dependency on the experiment runner, simulator,
MaxCut solver, NetworkX, or instance-generation code.  It reconstructs only the
saved immutable instance representation; it never regenerates graph data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cdtransfer.instances import IsingInstance, features as instance_features  # noqa: E402


METHODS = ("qaoa_csv", "cd_proposed")
METHOD_LABELS = {"qaoa_csv": "QAOA (same p)", "cd_proposed": "CD-QAOA (same p)"}
COLORS = {"qaoa_csv": "tab:blue", "cd_proposed": "tab:orange"}
OFFSETS = {"qaoa_csv": -0.10, "cd_proposed": 0.10}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
APPROVED_PRODUCTION = {
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
APPROVED_SMOKE = {
    "version": "maxcut-approx-smoke-v1",
    "seed": 20261012,
    "n_values": [8],
    "layer_n": 8,
    "depths": [1],
    "samples_per_cell": 1,
    "structure_cells": [["regular3", "unit"]],
    "weight_cells": [["regular3", "unit"]],
    "methods": ["qaoa_csv", "cd_proposed"],
    "frozen_model": "assets/CD_TRANSFER_V1_FROZEN_MODEL.json",
    "bootstrap_resamples": 200,
}


def bootstrap_mean_ci(values, *, n_boot: int, seed: int) -> list[float]:
    """Return a deterministic percentile bootstrap CI for the sample mean."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("Bootstrap values must be a nonempty one-dimensional sequence")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if np.any(~np.isfinite(x)):
        raise ValueError("Bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_cell(
    rows,
    *,
    method: str,
    p: int,
    n_boot: int,
    seed: int,
    expected_count: int = 10,
) -> dict:
    """Summarize one already-selected family/weight/n/depth cell."""
    if len(rows) != expected_count:
        raise ValueError(f"A complete plot cell requires exactly {expected_count} records")
    try:
        values = np.asarray(
            [row["evaluations"][str(p)][method]["approximation_ratio"] for row in rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid p={p}/{method} approximation ratio") from exc
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("A complete plot cell requires finite approximation ratios in [0,1]")
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "ci95": bootstrap_mean_ci(values, n_boot=n_boot, seed=seed),
    }


def _read_json(path: Path, expected_type):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{path.name} has the wrong top-level type")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cells(config: dict) -> list[tuple[str, str]]:
    raw = config.get("structure_cells", []) + config.get("weight_cells", [])
    try:
        cells = list(dict.fromkeys((x[0], x[1]) for x in raw))
    except (TypeError, IndexError) as exc:
        raise ValueError("Config cells must be family/weight pairs") from exc
    if not cells or any(not all(isinstance(y, str) and y for y in x) for x in cells):
        raise ValueError("Config cells must be nonempty family/weight string pairs")
    return cells


def same_typed(a, b) -> bool:
    """Recursively compare JSON-like values without bool/int/float coercion."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(same_typed(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_typed(x, y) for x, y in zip(a, b))
    return a == b


def _validate_config(config: dict) -> None:
    required = {
        "version", "seed", "n_values", "layer_n", "depths", "samples_per_cell",
        "structure_cells", "weight_cells", "methods", "frozen_model", "bootstrap_resamples",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Config is missing fields: {missing}")
    for name in ("seed", "layer_n", "samples_per_cell", "bootstrap_resamples"):
        if type(config[name]) is not int:
            raise ValueError(f"Config {name} must be a JSON integer")
    for name in ("n_values", "depths"):
        if not isinstance(config[name], list) or not config[name] or any(type(x) is not int for x in config[name]):
            raise ValueError(f"Config {name} elements must be JSON integers")
    if config["samples_per_cell"] < 1 or config["bootstrap_resamples"] < 1:
        raise ValueError("Sample and bootstrap counts must be positive")
    if not config["n_values"] or not config["depths"] or 1 not in config["depths"]:
        raise ValueError("Config requires nonempty n_values and depths containing p=1")
    _cells(config)
    is_production = same_typed(config, APPROVED_PRODUCTION)
    is_smoke = same_typed(config, APPROVED_SMOKE)
    if not is_production and not is_smoke:
        raise ValueError("Config does not match the approved production config or approved smoke config")


def _validate_manifest(config_text: str, config: dict, manifest: dict) -> None:
    if manifest.get("config_sha256") != _sha256(config_text.encode("utf-8")):
        raise ValueError("Config hash does not match manifest")
    saved_text = manifest.get("config_text")
    if not isinstance(saved_text, str) or _sha256(saved_text.encode("utf-8")) != manifest["config_sha256"]:
        raise ValueError("Manifest config_text hash is invalid")
    try:
        saved_text_config = json.loads(saved_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Manifest config_text is invalid JSON") from exc
    if not same_typed(manifest.get("config"), config) or not same_typed(saved_text_config, config):
        raise ValueError("Manifest and requested config differ")
    for name in ("frozen_model_sha256", "qaoa_source_csv_sha256"):
        if not isinstance(manifest.get(name), str) or not HEX64.fullmatch(manifest[name]):
            raise ValueError(f"Manifest {name} is not a SHA-256 digest")


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _deterministic_seed(base: int, n: int, family: str, weight: str, sample: int) -> int:
    token = f"{base}:{n}:{family}:{weight}:{sample}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def _expected_cases(config: dict) -> dict[str, tuple[str, str, int, int]]:
    return {
        f"{family}-{weight}-n{n}-sample{sample:02d}": (family, weight, n, sample)
        for family, weight in _cells(config)
        for n in config["n_values"]
        for sample in range(config["samples_per_cell"])
    }


def _validate_records(records: list, config: dict) -> None:
    expected_cases = _expected_cases(config)
    ids: set[str] = set()
    fingerprints: set[str] = set()
    slots: set[tuple[str, str, int, int]] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Every record must be an object")
        for field in ("id", "fingerprint", "instance", "features", "exact", "evaluations"):
            if field not in row:
                raise ValueError(f"Record is missing {field}")
        if not isinstance(row["id"], str) or row["id"] in ids:
            raise ValueError("Record IDs must be unique strings")
        if not isinstance(row["fingerprint"], str) or not HEX64.fullmatch(row["fingerprint"]):
            raise ValueError(f"Record {row['id']} has an invalid fingerprint")
        if row["fingerprint"] in fingerprints:
            raise ValueError("Record fingerprints must be unique")
        case = expected_cases.get(row["id"])
        if case is None:
            raise ValueError(f"Record {row['id']} lies outside configured deterministic cases")
        expected_family, expected_weight, expected_n, expected_sample = case
        expected_seed = _deterministic_seed(
            config["seed"], expected_n, expected_family, expected_weight, expected_sample
        )
        inst = row["instance"]
        if not isinstance(inst, dict):
            raise ValueError(f"Record {row['id']} instance must be an object")
        for field in ("n", "family", "weight_kind", "seed", "split", "edges", "weights"):
            if field not in inst:
                raise ValueError(f"Record {row['id']} instance is missing {field}")
        expected_metadata = {
            "n": expected_n,
            "family": expected_family,
            "weight_kind": expected_weight,
            "seed": expected_seed,
            "split": "comparison",
        }
        if any(not same_typed(inst.get(field), value) for field, value in expected_metadata.items()):
            raise ValueError(f"Record {row['id']} does not match its deterministic instance metadata")
        if not isinstance(inst["edges"], list) or not isinstance(inst["weights"], list) or len(inst["edges"]) != len(inst["weights"]) or not inst["edges"]:
            raise ValueError(f"Record {row['id']} has malformed edge/weight arrays")
        if any(not isinstance(edge, list) or len(edge) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in edge) for edge in inst["edges"]):
            raise ValueError(f"Record {row['id']} has malformed edges")
        weights = [_number(value, "edge weight") for value in inst["weights"]]
        if any(value <= 0 for value in weights):
            raise ValueError(f"Record {row['id']} requires strictly positive weights")
        try:
            saved_inst = IsingInstance.from_dict(inst)
            computed_fingerprint = saved_inst.fingerprint
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Record {row['id']} cannot reconstruct a valid IsingInstance") from exc
        if computed_fingerprint != row["fingerprint"]:
            raise ValueError(f"Record {row['id']} fingerprint does not match its n/edges/weights")
        cell = (expected_family, expected_weight)
        n = expected_n
        sample = expected_sample
        slot = (*cell, n, sample)
        if slot in slots:
            raise ValueError(f"Duplicate configured record slot: {slot}")
        exact = row["exact"]
        if not isinstance(exact, dict) or exact.get("scip_status") != "optimal" or exact.get("gurobi_status") != "optimal":
            raise ValueError(f"Record {row['id']} lacks two proven optimal statuses")
        scip = _number(exact.get("scip_objective"), "SCIP objective")
        gurobi = _number(exact.get("gurobi_objective"), "Gurobi objective")
        optimum = _number(exact.get("optimum"), "MaxCut optimum")
        if optimum <= 0 or not math.isclose(scip, gurobi, rel_tol=1e-8, abs_tol=1e-8) or not math.isclose(optimum, (scip + gurobi) / 2, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Record {row['id']} has solver disagreement")
        for field in ("scip_version", "gurobi_version"):
            if not isinstance(exact.get(field), str) or not exact[field]:
                raise ValueError(f"Record {row['id']} lacks {field}")
        for field in ("scip_seconds", "gurobi_seconds"):
            if _number(exact.get(field), field) < 0:
                raise ValueError(f"Record {row['id']} has negative solver time")
        saved_features = row["features"]
        expected_features = instance_features(saved_inst)
        if not isinstance(saved_features, dict) or set(saved_features) != set(expected_features):
            raise ValueError(f"Record {row['id']} features schema disagrees with recomputation")
        for field, expected_value in expected_features.items():
            saved_value = _number(saved_features[field], f"feature {field}")
            if not math.isclose(saved_value, float(expected_value), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Record {row['id']} features disagree at {field}")
        required_depths = [1] + ([2] if 2 in config["depths"] and n == config["layer_n"] else [])
        evaluations = row["evaluations"]
        if not isinstance(evaluations, dict):
            raise ValueError(f"Record {row['id']} evaluations must be an object")
        if set(evaluations) != {str(p) for p in required_depths}:
            raise ValueError(f"Record {row['id']} evaluation depths are not exact")
        for p in required_depths:
            depth_results = evaluations[str(p)]
            if not isinstance(depth_results, dict) or set(depth_results) != set(config["methods"]):
                raise ValueError(f"Record {row['id']} evaluation methods are not exact")
            for method in METHODS:
                try:
                    result = depth_results[method]
                    ratio = _number(result["approximation_ratio"], "approximation ratio")
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"Record {row['id']} lacks p={p}/{method}") from exc
                if ratio < 0 or ratio > 1:
                    raise ValueError(f"Record {row['id']} ratio is outside [0,1]")
                cut = _number(result.get("expected_cut"), "expected cut")
                energy = _number(result.get("ising_energy"), "Ising energy")
                expected_cut = 0.5 * (sum(weights) - energy)
                if not math.isclose(cut, expected_cut, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"Record {row['id']} expected cut disagrees with Ising energy")
                if not math.isclose(ratio, cut / optimum, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"Record {row['id']} ratio disagrees with expected_cut/optimum")
                angles = result.get("angles")
                if not isinstance(angles, dict) or set(angles) != {"gamma", "beta", "alpha"}:
                    raise ValueError(f"Record {row['id']} has malformed angles")
                for angle_name, angle_values in angles.items():
                    if not isinstance(angle_values, list) or len(angle_values) != p:
                        raise ValueError(f"Record {row['id']} has malformed {angle_name} angles")
                    for angle in angle_values:
                        _number(angle, f"{angle_name} angle")
        ids.add(row["id"])
        fingerprints.add(row["fingerprint"])
        slots.add(slot)


def _expected_slots(config: dict) -> set[tuple[str, str, int, int]]:
    return {
        (family, weight, n, sample)
        for family, weight in _cells(config)
        for n in config["n_values"]
        for sample in range(config["samples_per_cell"])
    }


def _seed(base: int, *parts) -> int:
    token = ":".join(map(str, (base, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _collect_summaries(records: list[dict], config: dict, expected_count: int) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in records:
        inst = row["instance"]
        grouped[(inst["family"], inst["weight_kind"], inst["n"])].append(row)
    output = []
    n_boot = config["bootstrap_resamples"]
    for family, weight in _cells(config):
        for n in config["n_values"]:
            rows = grouped.get((family, weight, n), [])
            if len(rows) != expected_count:
                continue
            for method in METHODS:
                item = summarize_cell(rows, method=method, p=1, n_boot=n_boot, seed=_seed(config["seed"], family, weight, n, 1, method), expected_count=expected_count)
                output.append({"family": family, "weight": weight, "n_qubits": n, "p": 1, "method": method, **item})
            if n == config["layer_n"]:
                for p in config["depths"]:
                    if p == 1:
                        continue
                    for method in METHODS:
                        item = summarize_cell(rows, method=method, p=p, n_boot=n_boot, seed=_seed(config["seed"], family, weight, n, p, method), expected_count=expected_count)
                        output.append({"family": family, "weight": weight, "n_qubits": n, "p": p, "method": method, **item})
    return output


def _global_ylim(records: list[dict]) -> tuple[float, float]:
    values = [
        float(result["approximation_ratio"])
        for row in records
        for depth in row["evaluations"].values()
        for result in depth.values()
    ]
    lower = max(0.0, math.floor((min(values) - 0.03) / 0.05) * 0.05)
    return lower, 1.01


def _plot_atlas_unclosed(
    path: Path,
    *,
    rows_spec: list[tuple[str, str, str]],
    records: list[dict],
    summaries: list[dict],
    config: dict,
    y_lim: tuple[float, float],
    title: str,
    incomplete: bool,
    figure_sink: list,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13), sharey=True, constrained_layout=True)
    figure_sink.append(fig)
    summary_map = {(x["family"], x["weight"], x["n_qubits"], x["p"], x["method"]): x for x in summaries}
    for row_index, (family, weight, label) in enumerate(rows_spec):
        left, right = axes[row_index]
        for method in METHODS:
            for n in config["n_values"]:
                ratios = [
                    r["evaluations"]["1"][method]["approximation_ratio"]
                    for r in records
                    if r["instance"]["family"] == family and r["instance"]["weight_kind"] == weight and r["instance"]["n"] == n
                ]
                if ratios:
                    left.scatter(np.full(len(ratios), n + OFFSETS[method]), ratios, s=18, alpha=0.70, color=COLORS[method], label=METHOD_LABELS[method] if n == config["n_values"][0] else None)
            ps, means, low, high = [], [], [], []
            for p in config["depths"]:
                item = summary_map.get((family, weight, config["layer_n"], p, method))
                if item:
                    ps.append(p); means.append(item["mean"])
                    low.append(item["mean"] - item["ci95"][0]); high.append(item["ci95"][1] - item["mean"])
            if ps:
                right.errorbar(ps, means, yerr=[low, high], color=COLORS[method], marker="o", linewidth=1.6, capsize=3, label=METHOD_LABELS[method])
        left.set_title(f"{label}: p=1 instance ratios")
        right.set_title(f"{label}: n={config['layer_n']} bootstrap means")
        left.set_xticks(config["n_values"])
        right.set_xticks(config["depths"])
        left.set_xlabel("Qubits n")
        right.set_xlabel("Layers p")
        left.set_ylabel("Approximation ratio")
        for axis in (left, right):
            axis.set_ylim(*y_lim); axis.grid(alpha=0.22)
        if row_index == 0:
            if left.get_legend_handles_labels()[0]:
                left.legend(loc="lower right", fontsize=8)
            if right.get_legend_handles_labels()[0]:
                right.legend(loc="lower right", fontsize=8)
    fig.suptitle(title + "\nSame-layer comparison; not matched two-qubit-gate budget", fontsize=15)
    if incomplete:
        fig.text(0.5, 0.5, "INCOMPLETE SMOKE", ha="center", va="center", fontsize=42, color="crimson", alpha=0.18, rotation=28, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)


def _plot_atlas(path: Path, **kwargs) -> None:
    figures = []
    try:
        _plot_atlas_unclosed(path, figure_sink=figures, **kwargs)
    finally:
        for figure in figures:
            plt.close(figure)


def _write_report(path: Path, config: dict, manifest: dict, summaries: list[dict], records: list[dict], incomplete: bool) -> None:
    marker = "\n> **INCOMPLETE SMOKE** — diagnostic output only; not a production result.\n" if incomplete else ""
    lines = [
        "# MaxCut Approximation Experiment Report", marker,
        "## Configuration", "",
        f"- Version: `{config['version']}`",
        f"- Samples per cell: {config['samples_per_cell']}",
        f"- Qubits: {config['n_values']}",
        f"- Layers: {config['depths']}",
        f"- Config SHA-256: `{manifest['config_sha256']}`", "",
        "## Solver agreement", "",
        f"All {len(records)} saved records report optimal SCIP and Gurobi statuses with objectives agreeing to 1e-8.", "",
        "## Results", "",
        "| Family | Weight | n | p | Method | Count | Mean | 95% bootstrap CI |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(f"| {item['family']} | {item['weight']} | {item['n_qubits']} | {item['p']} | {item['method']} | {item['n']} | {item['mean']:.6f} | [{item['ci95'][0]:.6f}, {item['ci95'][1]:.6f}] |")
    lines += [
        "", "## Figures", "",
        "| Figure | Saved artifact |",
        "|---|---|",
        "| Graph structures | `figures/fig01_graph_structures_same_layer.png` |",
        "| Weight distributions | `figures/fig02_weight_distributions_same_layer.png` |",
        "", "## Interpretation limitation", "",
        "This is a **same-layer** comparison. It is not a matched or equal two-qubit-gate-budget comparison; CD-QAOA and QAOA at the same p can have different compiled costs.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(config_path: str | Path, out_path: str | Path, *, allow_incomplete_smoke: bool = False) -> dict:
    config_path, out_path = Path(config_path), Path(out_path)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    _validate_config(config)
    production_config = same_typed(config, APPROVED_PRODUCTION)
    smoke_config = not production_config
    if allow_incomplete_smoke and production_config:
        raise ValueError("--allow-incomplete-smoke is forbidden for a production config")
    if smoke_config and not allow_incomplete_smoke:
        raise ValueError("Smoke/incomplete config requires --allow-incomplete-smoke")
    manifest = _read_json(out_path / "manifest.json", dict)
    records = _read_json(out_path / "records.json", list)
    _validate_manifest(config_text, config, manifest)
    _validate_records(records, config)
    observed = {(r["instance"]["family"], r["instance"]["weight_kind"], r["instance"]["n"], int(re.search(r"-sample(\d+)$", r["id"]).group(1))) for r in records}
    expected = _expected_slots(config)
    if observed != expected:
        raise ValueError(f"Saved records are incomplete: found {len(observed)} of {len(expected)} configured records")
    if manifest.get("expected_records") != len(expected) or len(records) != len(expected):
        raise ValueError("Manifest expected_records does not match the validated record set")
    expected_count = config["samples_per_cell"] if allow_incomplete_smoke else 10
    summaries = _collect_summaries(records, config, expected_count)
    complete = not allow_incomplete_smoke
    output = {
        "version": config["version"], "complete": complete,
        "watermark": "INCOMPLETE SMOKE" if allow_incomplete_smoke else None,
        "config_sha256": manifest["config_sha256"], "record_count": len(records),
        "solver_agreement_count": len(records), "cells": summaries,
    }
    y_lim = _global_ylim(records)
    structure_rows = [("tree", "unit", "Random tree"), ("regular3", "unit", "3-regular"), ("er3", "unit", "ER3"), ("er6", "unit", "ER6")]
    weight_rows = [("regular3", "unit", "Unit"), ("regular3", "uniform", "Uniform"), ("regular3", "bimodal", "Bimodal"), ("regular3", "lognormal", "Lognormal")]
    output["figures"] = {
        "layout": [4, 2],
        "structure_rows": [f"{family}/{weight}" for family, weight, _ in structure_rows],
        "weight_rows": [f"{family}/{weight}" for family, weight, _ in weight_rows],
        "colors": dict(COLORS),
        "offsets": dict(OFFSETS),
        "n_values": list(config["n_values"]),
        "depths": list(config["depths"]),
        "shared_ylim": list(y_lim),
        "dpi": 180,
        "warning": "Same-layer comparison; not matched two-qubit-gate budget",
        "watermark": "INCOMPLETE SMOKE" if allow_incomplete_smoke else None,
    }
    figures = out_path / "figures"
    _plot_atlas(figures / "fig01_graph_structures_same_layer.png", rows_spec=structure_rows, records=records, summaries=summaries, config=config, y_lim=y_lim, title="Graph structures", incomplete=allow_incomplete_smoke)
    _plot_atlas(figures / "fig02_weight_distributions_same_layer.png", rows_spec=weight_rows, records=records, summaries=summaries, config=config, y_lim=y_lim, title="Positive-weight distributions on 3-regular graphs", incomplete=allow_incomplete_smoke)
    (out_path / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_path / "EXPERIMENT_REPORT.md", config, manifest, summaries, records, allow_incomplete_smoke)
    return output


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-incomplete-smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.out, allow_incomplete_smoke=args.allow_incomplete_smoke)


if __name__ == "__main__":
    _cli()
