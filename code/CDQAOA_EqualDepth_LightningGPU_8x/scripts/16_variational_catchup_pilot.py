"""Run a bounded variational QAOA/CD-QAOA catch-up pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cdtransfer.instances import IsingInstance
from cdtransfer.maxcut import expected_cut_from_ising
from cdtransfer.simulator import Angles, StatevectorSimulator
from scripts.variational_catchup_helpers import (
    cumulative_best,
    first_crossing,
    mean_trajectory,
    pad_best,
)

METHODS = ("qaoa", "cd_qaoa")
PARAMETER_BOUNDS = ((-3 * np.pi, 3 * np.pi), (-np.pi, np.pi), (-2 * np.pi, 2 * np.pi))
OUTPUT_ARTIFACTS = ("trajectories.json", "summary.json", "variational_catchup.png", "REPORT.md")
PLOT_STYLES = {
    "fixed_cd_qaoa": {"color": "#B5179E", "marker": "D"},
    "qaoa": {"label": "QAOA", "color": "#D95F02"},
    "cd_qaoa": {"label": "CD-QAOA", "color": "#00A6A6"},
}


def deterministic_seed(base_seed: int, instance_id: str, method: str, start: int) -> int:
    token = f"{base_seed}:{instance_id}:{method}:{start}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def run_trajectory(
    *,
    instance_id: str,
    fingerprint: str,
    instance: IsingInstance,
    optimum: float,
    method: str,
    start: int,
    seed: int,
    max_evaluations: int,
    evaluator=None,
) -> dict:
    """Optimize one initial point; record exactly every statevector evaluation."""
    if method not in METHODS:
        raise ValueError(f"Unknown variational method: {method}")
    if not math.isfinite(float(optimum)) or optimum <= 0:
        raise ValueError("optimum must be finite and positive")
    if not 1 <= int(max_evaluations):
        raise ValueError("max_evaluations must be positive")
    p = 1
    dimensions = 2 if method == "qaoa" else 3
    bounds = PARAMETER_BOUNDS[:dimensions]
    rng = np.random.default_rng(int(seed))
    initial_point = [float(rng.uniform(low, high)) for low, high in bounds]
    sim_evaluate = evaluator or StatevectorSimulator(instance).evaluate
    evaluations = []
    best_ratio = -math.inf

    def objective(theta):
        nonlocal best_ratio
        if len(evaluations) >= max_evaluations:
            raise RuntimeError("optimizer exceeded the objective evaluation budget")
        theta = np.asarray(theta, dtype=np.float64)
        angles = Angles.unpack(theta, p=p, include_cd=(method == "cd_qaoa"))
        energy = float(sim_evaluate(angles))
        cut = expected_cut_from_ising(instance, energy)
        ratio = cut / float(optimum)
        if not math.isfinite(energy) or not math.isfinite(cut) or not math.isfinite(ratio):
            raise ValueError("objective produced a non-finite value")
        if ratio < -1e-9 or ratio > 1 + 1e-9:
            raise ValueError(f"approximation ratio outside [0, 1]: {ratio}")
        ratio = min(1.0, max(0.0, ratio))
        best_ratio = max(best_ratio, ratio)
        evaluations.append(
            {
                "evaluation": len(evaluations) + 1,
                "angles": theta.tolist(),
                "ising_energy": energy,
                "expected_cut": cut,
                "approximation_ratio": ratio,
                "best_so_far": best_ratio,
            }
        )
        return energy

    result = minimize(
        objective,
        np.asarray(initial_point, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxfev": int(max_evaluations)},
    )
    return {
        "instance_id": instance_id,
        "instance_fingerprint": fingerprint,
        "method": method,
        "p": p,
        "start": int(start),
        "seed": int(seed),
        "initial_point": initial_point,
        "max_evaluations": int(max_evaluations),
        "actual_evaluations": len(evaluations),
        "evaluations": evaluations,
        "optimizer": "Nelder-Mead",
        "optimizer_status": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
        },
    }


def load_source_records(path: Path, config: dict) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("source records must be a JSON list")
    selected = []
    for row in records:
        instance_data = row.get("instance", {})
        if (
            instance_data.get("family") == config["family"]
            and instance_data.get("weight_kind") == config["weight_kind"]
            and instance_data.get("n") == config["n_qubits"]
        ):
            instance = IsingInstance.from_dict(instance_data)
            if instance.fingerprint != row.get("fingerprint"):
                raise ValueError(f"source fingerprint mismatch for {row.get('id')}")
            optimum = float(row["exact"]["optimum"])
            fixed_ratio = float(row["evaluations"]["1"]["cd_proposed"]["approximation_ratio"])
            if not math.isfinite(optimum) or optimum <= 0:
                raise ValueError(f"invalid exact optimum for {row.get('id')}")
            if not math.isfinite(fixed_ratio) or not 0 <= fixed_ratio <= 1:
                raise ValueError(f"invalid fixed CD-QAOA ratio for {row.get('id')}")
            selected.append(
                {
                    "id": row["id"],
                    "fingerprint": row["fingerprint"],
                    "instance": instance,
                    "optimum": optimum,
                    "fixed_ratio": fixed_ratio,
                }
            )
    selected = selected[: int(config["instances"])]
    if len(selected) != int(config["instances"]):
        raise ValueError(f"only found {len(selected)} matching saved instances")
    if len({item["fingerprint"] for item in selected}) != len(selected):
        raise ValueError("selected source records contain duplicate fingerprints")
    return selected, hashlib.sha256(raw).hexdigest()


def validate_trajectories(trajectories: list[dict], instances: list[dict], starts: int, cap: int) -> None:
    expected = len(instances) * starts * len(METHODS)
    if len(trajectories) != expected:
        raise ValueError(f"expected {expected} trajectories, found {len(trajectories)}")
    known_fingerprints = {item["fingerprint"] for item in instances}
    keys = set()
    counts = {method: 0 for method in METHODS}
    for trace in trajectories:
        key = (trace["instance_id"], trace["method"], trace["start"])
        if key in keys:
            raise ValueError(f"duplicate trajectory key: {key}")
        keys.add(key)
        method = trace["method"]
        if method not in METHODS or trace["p"] != 1:
            raise ValueError("trajectory method/depth is invalid")
        counts[method] += 1
        if trace["instance_fingerprint"] not in known_fingerprints:
            raise ValueError("trajectory fingerprint is not from selected source records")
        evaluations = trace["evaluations"]
        if not 1 <= len(evaluations) <= cap or trace["actual_evaluations"] != len(evaluations):
            raise ValueError("trajectory evaluation count is outside its configured cap")
        dimensions = 2 if method == "qaoa" else 3
        best = -math.inf
        for expected_index, item in enumerate(evaluations, start=1):
            if item["evaluation"] != expected_index or len(item["angles"]) != dimensions:
                raise ValueError("trajectory evaluation numbering or parameter dimension is invalid")
            numeric = [*item["angles"], item["ising_energy"], item["expected_cut"],
                       item["approximation_ratio"], item["best_so_far"]]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise ValueError("trajectory contains non-finite values")
            ratio = float(item["approximation_ratio"])
            if not 0 <= ratio <= 1:
                raise ValueError("trajectory ratio is outside [0, 1]")
            best = max(best, ratio)
            if not math.isclose(float(item["best_so_far"]), best, abs_tol=1e-12):
                raise ValueError("best-so-far trajectory is not the cumulative maximum")
    if any(counts[method] != len(instances) * starts for method in METHODS):
        raise ValueError("each method must have exactly instances × starts trajectories")


def summarize(trajectories: list[dict], instances: list[dict], cap: int) -> dict:
    fixed_target = float(np.mean([item["fixed_ratio"] for item in instances]))
    methods = {}
    for method in METHODS:
        traces = [
            pad_best([row["best_so_far"] for row in trace["evaluations"]], cap)
            for trace in trajectories if trace["method"] == method
        ]
        mean = mean_trajectory(traces)
        crossing = first_crossing(mean, fixed_target)
        methods[method] = {
            "trajectory_count": len(traces),
            "mean_best_so_far": mean,
            "first_crossing_evaluation": crossing,
            "crossing_status": "reached" if crossing is not None else "not reached",
            "final_mean_at_max_evaluations": mean[-1],
        }
    return {
        "fixed_cd_qaoa_target": fixed_target,
        "fixed_target_instance_ratios": [
            {"instance_id": item["id"], "approximation_ratio": item["fixed_ratio"]}
            for item in instances
        ],
        "evaluation_counts": list(range(1, cap + 1)),
        "methods": methods,
    }


def crossing_annotation(method: str, evaluation: int | None) -> str:
    label = {"qaoa": "QAOA", "cd_qaoa": "CD-QAOA"}.get(method)
    if label is None:
        raise ValueError(f"Unknown variational method: {method}")
    return f"{label}: {evaluation}" if evaluation is not None else f"{label}: not reached"


def crossing_annotation_offset(method: str, evaluation: int | None) -> tuple[int, int]:
    if method not in METHODS:
        raise ValueError(f"Unknown variational method: {method}")
    if evaluation is None:
        return (-4, -18)
    return (7, 9) if method == "qaoa" else (7, -18)


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def plot_y_limits(summary: dict) -> tuple[float, float]:
    values = [float(summary["fixed_cd_qaoa_target"])]
    for method in METHODS:
        values.extend(float(value) for value in summary["methods"][method]["mean_best_so_far"])
    return (math.floor((min(values) - 0.03) * 100) / 100,
            math.ceil((max(values) + 0.02) * 100) / 100)


def write_plot(summary: dict, path: Path, instance_count: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = summary["fixed_cd_qaoa_target"]
    counts = summary["evaluation_counts"]
    figure, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    fixed = PLOT_STYLES["fixed_cd_qaoa"]
    axis.scatter([0], [target], color=fixed["color"], marker=fixed["marker"], s=55, zorder=5,
                 label="Fixed CD-QAOA (p=1)")
    axis.axhline(target, color=fixed["color"], linewidth=1.8, alpha=0.8)
    for method in METHODS:
        style = PLOT_STYLES[method]
        label, color = style["label"], style["color"]
        detail = summary["methods"][method]
        mean = detail["mean_best_so_far"]
        axis.plot(counts, mean, color=color, linewidth=2, label=f"Variational {label}")
        crossing = detail["first_crossing_evaluation"]
        offset = crossing_annotation_offset(method, crossing)
        if crossing is not None:
            axis.scatter([crossing], [mean[crossing - 1]], color=color, edgecolor="white", s=55, zorder=6)
            axis.annotate(
                crossing_annotation(method, crossing), (crossing, mean[crossing - 1]),
                xytext=offset, textcoords="offset points",
                color=color, fontsize=9,
            )
        else:
            axis.annotate(crossing_annotation(method, crossing), (counts[-1], mean[-1]),
                          xytext=offset, textcoords="offset points", ha="right", color=color, fontsize=9)
    axis.set_title(f"Variational catch-up pilot — {instance_count} saved instances (exploratory)", fontsize=12)
    axis.set_xlabel("Target-instance objective evaluations")
    axis.set_ylabel("Approximation ratio (best so far)")
    axis.set_xlim(-1, counts[-1])
    axis.set_ylim(*plot_y_limits(summary))
    axis.grid(True, alpha=0.22)
    axis.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(summary: dict, config: dict, elapsed_seconds: float, path: Path) -> None:
    target = summary["fixed_cd_qaoa_target"]
    evaluation_cap = len(summary["evaluation_counts"])
    lines = [
        "# Variational catch-up pilot",
        "",
        "This exploratory pilot reuses saved `regular3/unit`, n=12 instances and does not rerun the prior production experiment.",
        "It is not the planned 70-instance paper-scale comparison.",
        "",
        f"- Configuration: p={config['p']}, {config['instances']} instances, {config['starts']} starts per method, at most {config['max_evaluations']} objective evaluations per trajectory.",
        f"- Optimizer: {config['optimizer']}; deterministic base seed: {config['base_seed']}.",
        f"- Fixed p=1 CD-QAOA target (selected instances counted once each): {target:.6f}.",
        f"- Simulation and validation time: {elapsed_seconds:.2f} seconds.",
        "",
        f"| Method | Mean catch-up evaluation | Mean best-so-far at evaluation {evaluation_cap} | Trajectories |",
        "|---|---:|---:|---:|",
    ]
    for method, label in (("qaoa", "Variational QAOA"), ("cd_qaoa", "Variational CD-QAOA")):
        detail = summary["methods"][method]
        crossing = detail["first_crossing_evaluation"]
        crossing_text = str(crossing) if crossing is not None else "not reached"
        lines.append(f"| {label} | {crossing_text} | {detail['final_mean_at_max_evaluations']:.6f} | {detail['trajectory_count']} |")
    lines.extend(["", f"Mean curves equally weight all trajectories; runs ending early carry their final best ratio forward to evaluation {evaluation_cap}.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_is_available(output_dir: Path) -> None:
    existing = [name for name in OUTPUT_ARTIFACTS if (output_dir / name).exists()]
    if existing:
        names = ", ".join(existing)
        raise FileExistsError(f"Refusing to overwrite existing pilot output artifact(s): {names}")


def run_pilot(config_path: Path, output_dir: Path, *, instances_count=None, starts=None, max_evaluations=None) -> dict:
    _ensure_output_is_available(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = dict(config)
    if instances_count is not None:
        config["instances"] = int(instances_count)
    if starts is not None:
        config["starts"] = int(starts)
    if max_evaluations is not None:
        config["max_evaluations"] = int(max_evaluations)
    source_path = (PROJECT_ROOT / config["source_records"]).resolve()
    selected, source_hash = load_source_records(source_path, config)
    started = time.perf_counter()
    trajectories = []
    for item in selected:
        for method in METHODS:
            for start in range(config["starts"]):
                seed = deterministic_seed(config["base_seed"], item["id"], method, start)
                trajectories.append(
                    run_trajectory(
                        instance_id=item["id"], fingerprint=item["fingerprint"],
                        instance=item["instance"], optimum=item["optimum"], method=method,
                        start=start, seed=seed, max_evaluations=config["max_evaluations"],
                    )
                )
    validate_trajectories(trajectories, selected, config["starts"], config["max_evaluations"])
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": config,
        "source_records_sha256": source_hash,
        "trajectories": trajectories,
    }
    _atomic_json(output_dir / "trajectories.json", payload)
    summary = summarize(trajectories, selected, config["max_evaluations"])
    summary["configuration"] = config
    summary["source_records_sha256"] = source_hash
    summary["simulation_and_validation_seconds"] = elapsed
    _atomic_json(output_dir / "summary.json", summary)
    write_plot(summary, output_dir / "variational_catchup.png", len(selected))
    write_report(summary, config, elapsed, output_dir / "REPORT.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "variational_catchup_pilot.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instances", type=int)
    parser.add_argument("--starts", type=int)
    parser.add_argument("--max-evaluations", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or PROJECT_ROOT / config["output_dir"]
    summary = run_pilot(
        args.config, output_dir, instances_count=args.instances,
        starts=args.starts, max_evaluations=args.max_evaluations,
    )
    print(json.dumps({
        "fixed_cd_qaoa_target": summary["fixed_cd_qaoa_target"],
        "methods": {
            key: {
                "first_crossing_evaluation": value["first_crossing_evaluation"],
                "final_mean_at_max_evaluations": value["final_mean_at_max_evaluations"],
            }
            for key, value in summary["methods"].items()
        },
        "simulation_and_validation_seconds": summary["simulation_and_validation_seconds"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
