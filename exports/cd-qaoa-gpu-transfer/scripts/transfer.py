"""Offline GPU-optima aggregation and simulation-free CD-QAOA angle prediction.

Building needs NumPy and SciPy. Predicting from the frozen JSON uses only the
Python standard library. Neither operation evaluates a quantum objective.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


DEPTHS = (1, 2, 3, 4, 6)
PACKAGE = Path(__file__).resolve().parents[1]


def degree_factor(degree: float) -> float:
    return 1.0 if degree <= 1.0 else math.atan(1.0 / math.sqrt(degree - 1.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_cd(path: Path, depths: tuple[int, ...]) -> dict[int, dict[str, dict]]:
    result = {p: {} for p in depths}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["cd"].strip().lower() != "true":
                continue
            p = int(row["p"])
            if p not in result:
                continue
            instance = row["id"]
            if not instance.startswith("train-"):
                raise ValueError(f"non-training row in {path}: {instance}")
            if instance in result[p]:
                raise ValueError(f"duplicate CD optimum in {path}: {instance}, p={p}")
            degree = float(row["degree"])
            rms = float(row["rms"])
            if not (math.isfinite(degree) and degree > 0 and math.isfinite(rms) and rms > 0):
                raise ValueError(f"invalid features for {instance}")
            gamma = [float(x) for x in json.loads(row["gamma"])]
            beta = [float(x) for x in json.loads(row["beta"])]
            alpha = [float(x) for x in json.loads(row["alpha"])]
            if not all(len(v) == p and all(math.isfinite(x) for x in v)
                       for v in (gamma, beta, alpha)):
                raise ValueError(f"invalid angle lengths or values for {instance}, p={p}")
            if sum(gamma) / p > 0:
                gamma = [-x for x in gamma]
                beta = [-x for x in beta]
            vector = (
                [x * rms / degree_factor(degree) for x in gamma]
                + beta
                + [x * rms for x in alpha]
            )
            result[p][instance] = {"degree": degree, "vector": vector}
    return result


def _medoid(rows: dict[str, dict]) -> tuple[str, list[float]]:
    import numpy as np
    from scipy.spatial.distance import cdist

    ids = sorted(rows)
    vectors = np.asarray([rows[name]["vector"] for name in ids], dtype=np.float64)
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite normalized angle")
    totals = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), 256):
        end = min(start + 256, len(ids))
        totals[start:end] = cdist(vectors[start:end], vectors, metric="euclidean").sum(axis=1)
    best = int(np.argmin(totals))
    return ids[best], vectors[best].tolist()


def build_model(
    small: str | Path,
    large: str | Path,
    extra: str | Path,
    depths: tuple[int, ...] = DEPTHS,
) -> dict:
    """Use small+large for p<=4, small+extra for p=6; never use test data."""
    import numpy as np

    depths = tuple(sorted(set(int(p) for p in depths)))
    if any(p not in DEPTHS for p in depths):
        raise ValueError("unsupported depth requested")
    sources = {key: Path(value) for key, value in
               (("small", small), ("large", large), ("extra", extra))}
    records = {key: _read_cd(path, depths) for key, path in sources.items()}
    if 1 not in records["small"] or not records["small"][1] or not records["large"][1]:
        raise ValueError("small and large sources need p=1 training records for D0")
    reference_rows = dict(records["small"][1])
    overlap = set(reference_rows) & set(records["large"][1])
    if overlap:
        raise ValueError(f"small/large training IDs overlap: {sorted(overlap)[:3]}")
    reference_rows.update(records["large"][1])
    d0 = float(np.median([row["degree"] for row in reference_rows.values()]))
    if not math.isfinite(d0) or d0 <= 0:
        raise ValueError("invalid shared reference degree")

    templates = {}
    counts = {}
    for p in depths:
        second = "extra" if p == 6 else "large"
        rows = dict(records["small"][p])
        overlap = set(rows) & set(records[second][p])
        if overlap:
            raise ValueError(f"selected training IDs overlap at p={p}: {sorted(overlap)[:3]}")
        rows.update(records[second][p])
        if not rows:
            raise ValueError(f"missing training angles for p={p}")
        medoid_id, vector = _medoid(rows)
        templates[str(p)] = {
            "gamma_unit": vector[:p],
            "beta": vector[p:2 * p],
            "alpha_unit": vector[2 * p:],
            "medoid_training_id": medoid_id,
        }
        counts[str(p)] = len(rows)
    return {
        "name": "cd-qaoa-unified-gpu-offline",
        "version": 1,
        "status": "offline_pooled_training_template_not_tested_as_one_model",
        "depths": list(depths),
        "reference_degree": d0,
        "alpha_degree_power": -0.5,
        "alpha_multiplier": 1.0,
        "templates": templates,
        "training_counts": counts,
        "unique_reference_training_graphs": len(reference_rows),
        "provenance": {
            key: {"file": f"data/{key}_best_optima.csv.gz", "sha256": _sha256(path)}
            for key, path in sources.items()
        },
        "normalization": "gamma_unit=gamma*rms/f_D(D); beta=beta; alpha_unit=alpha*rms; if mean(gamma)>0 flip gamma,beta",
        "template_selection": "pooled Euclidean medoid; p<=4 small+large, p=6 small+extra",
        "no_target_instance_objective_calls": True,
    }


def predict_angles(model: dict, n: int, edges: list[list[float]], p: int) -> dict:
    """Predict fixed angles from a positive-weight graph; no simulation."""
    p = int(p)
    if p not in model.get("depths", []) or str(p) not in model.get("templates", {}):
        raise ValueError(f"unsupported p={p}; available: {model.get('depths', [])}")
    if not isinstance(n, int) or n < 2 or not edges:
        raise ValueError("n must be >=2 and edges must be nonempty")
    seen = set()
    weights = []
    for edge in edges:
        if len(edge) != 3:
            raise ValueError("each edge must be [i,j,positive_weight]")
        i, j, weight = edge
        if not isinstance(i, int) or not isinstance(j, int) or not (0 <= i < j < n):
            raise ValueError("edges must use 0-based i<j<n")
        if (i, j) in seen:
            raise ValueError("duplicate edge")
        seen.add((i, j))
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("this model requires finite positive weights")
        weights.append(weight)
    degree = 2.0 * len(edges) / n
    rms = math.sqrt(sum(w * w for w in weights) / len(weights))
    d0 = float(model["reference_degree"])
    b = float(model["alpha_degree_power"])
    c = float(model["alpha_multiplier"])
    entry = model["templates"][str(p)]
    for key in ("gamma_unit", "beta", "alpha_unit"):
        if len(entry[key]) != p:
            raise ValueError(f"corrupt model: {key} length != p")
    return {
        "gamma": [float(x) * degree_factor(degree) / rms for x in entry["gamma_unit"]],
        "beta": [float(x) for x in entry["beta"]],
        "alpha": [float(x) / rms * (degree / d0) ** b * c
                  for x in entry["alpha_unit"]],
        "features": {"degree": degree, "rms": rms, "reference_degree": d0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="offline rebuild from bundled training CSV.gz")
    for name in ("small", "large", "extra"):
        build.add_argument(f"--{name}", type=Path,
                           default=PACKAGE / "data" / f"{name}_best_optima.csv.gz")
    build.add_argument("--out", type=Path, default=PACKAGE / "assets" / "unified_gpu_model.json")
    predict = commands.add_parser("predict", help="compute angles from graph JSON, no simulation")
    predict.add_argument("--model", type=Path,
                         default=PACKAGE / "assets" / "unified_gpu_model.json")
    predict.add_argument("--graph", type=Path, required=True)
    predict.add_argument("--p", type=int, required=True)
    args = parser.parse_args()
    if args.command == "build":
        model = build_model(args.small, args.large, args.extra)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(model, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        args.out.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        args.out.with_suffix(args.out.suffix + ".sha256").write_text(
            f"{digest}  {args.out.name}\n", encoding="ascii"
        )
        print(json.dumps({"model": str(args.out), "sha256": digest,
                          "counts": model["training_counts"], "D0": model["reference_degree"]}))
    else:
        model = json.loads(args.model.read_text(encoding="utf-8"))
        graph = json.loads(args.graph.read_text(encoding="utf-8"))
        print(json.dumps(predict_angles(model, graph["n"], graph["edges"], args.p), indent=2))


if __name__ == "__main__":
    main()
