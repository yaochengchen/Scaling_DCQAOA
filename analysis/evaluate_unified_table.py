"""Evaluate the locked unified CD-QAOA table on archived, held-out graphs.

This program never optimizes angles or computes MaxCut extrema. Historical run
directories are read-only; per-graph results are written to a separate folder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "CDQAOA_EqualDepth_LightningGPU_8x"))

import numpy as np
from cdtransfer.instances import IsingInstance
from cdtransfer.simulator import Angles
from cdtransfer.transfer import predict


EXPECTED_MODEL_SHA256 = "f2be3446fd992a826dcfec8f6048b27a8ccf21fff146d06faa84fd507d7f1eb4"
COHORTS = (
    ("small", "eqdepth_small_N20", 4000, (1, 2, 3, 4, 6)),
    ("large", "eqdepth_large_N20", 3000, (1, 2, 3, 4)),
)


def _read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _instance(row: dict) -> IsingInstance:
    return IsingInstance.from_dict({key: row[key] for key in
                                    ("n", "edges", "weights", "family", "weight_kind", "seed", "split")})


def load_archived_inputs(root: Path = ROOT):
    model_path = root / "exports" / "cd-qaoa-gpu-transfer" / "assets" / "unified_gpu_model.json"
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(f"Unified model changed: {actual_hash}")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model["status"] != "offline_pooled_training_template_not_tested_as_one_model":
        raise ValueError("Unexpected unified model status")
    cohorts = []
    for name, directory, expected_count, expected_depths in COHORTS:
        source = root / "results" / "gpu_eqdepth" / directory
        manifest = _read_gzip_json(source / "manifest.json.gz")
        rows = manifest["splits"]["test"]
        depths = tuple(manifest["config"]["depths"])
        if len(rows) != expected_count or depths != expected_depths:
            raise ValueError(f"Unexpected {name} test split or depths")
        old = _read_gzip_json(source / "heldout_results.json.gz")
        old_by_key = {(record["id"], record["p"]): record for record in old}
        if len(old_by_key) != expected_count * len(depths):
            raise ValueError(f"Missing or duplicate historical results in {name}")
        for row in rows:
            for p in depths:
                historical = old_by_key.get((row["id"], p))
                if historical is None or any(historical[key] != row[key] for key in
                                             ("n", "family", "weight_kind", "seed")):
                    raise ValueError(f"Historical result mismatch: {row['id']}, p={p}")
                if historical["m"] != len(row["edges"]):
                    raise ValueError(f"Historical edge-count mismatch: {row['id']}, p={p}")
                if historical["energy_qaoa_best"] is None or any(
                    historical[key] is None for key in ("ground_energy", "max_energy")
                ):
                    raise ValueError(f"Missing baseline or extrema: {row['id']}, p={p}")
        cohorts.append({"name": name, "rows": rows, "depths": depths,
                        "old_results": old_by_key, "source": source})
    return model, cohorts


def predict_unified(inst: IsingInstance, p: int, model: dict) -> Angles:
    if p not in model.get("depths", (int(key) for key in model["templates"])):
        raise ValueError(f"Depth {p} absent from unified model")
    entry = dict(model["templates"][str(p)], reference_degree=model["reference_degree"])
    compatible = {"templates": {str(p): entry},
                  "alpha_degree_power": model["alpha_degree_power"],
                  "alpha_multiplier": model["alpha_multiplier"]}
    return predict(inst, p, "cd_proposed", compatible)


def energy_qulacs(inst: IsingInstance, angles: Angles) -> float:
    """Exact statevector expectation with the project's cost→YZ→ZY→mixer order."""
    from qulacs import Observable, QuantumCircuit, QuantumState
    from qulacs.gate import H, PauliRotation

    circuit = QuantumCircuit(inst.n)
    for i in range(inst.n):
        circuit.add_gate(H(i))
    for gamma, beta, alpha in zip(angles.gamma, angles.beta, angles.alpha):
        for (i, j), weight in zip(inst.edges, inst.weights):
            circuit.add_gate(PauliRotation([int(i), int(j)], [3, 3],
                                           float(-2 * gamma * weight)))
        for (i, j), weight in zip(inst.edges, inst.weights):
            circuit.add_gate(PauliRotation([int(i), int(j)], [2, 3],
                                           float(-2 * alpha * weight)))
            circuit.add_gate(PauliRotation([int(i), int(j)], [3, 2],
                                           float(-2 * alpha * weight)))
        for i in range(inst.n):
            circuit.add_gate(PauliRotation([i], [1], float(-2 * beta)))
    state = QuantumState(inst.n)
    circuit.update_quantum_state(state)
    observable = Observable(inst.n)
    for (i, j), weight in zip(inst.edges, inst.weights):
        observable.add_operator(float(weight), f"Z {int(i)} Z {int(j)}")
    return float(observable.get_expectation_value(state))


def evaluate_row(row: dict, depths: tuple[int, ...], model: dict, cohort: str):
    inst = _instance(row)
    out = []
    for p in depths:
        angles = predict_unified(inst, p, model)
        energy = energy_qulacs(inst, angles)
        if not np.isfinite(energy):
            raise ValueError(f"Nonfinite energy: {row['id']}, p={p}")
        out.append({"cohort": cohort, "id": row["id"], "fingerprint": row["fingerprint"],
                    "p": p, "n": inst.n, "family": inst.family,
                    "weight_kind": inst.weight_kind, "energy_cd_unified": energy,
                    "model_sha256": EXPECTED_MODEL_SHA256})
    return out


def _checkpoint_valid(path: Path, row: dict, depths: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    records = json.loads(path.read_text(encoding="utf-8"))
    return (len(records) == len(depths) and
            [record["p"] for record in records] == list(depths) and
            all(record["fingerprint"] == row["fingerprint"] and
                record["model_sha256"] == EXPECTED_MODEL_SHA256 and
                np.isfinite(record["energy_cd_unified"]) for record in records))


def _save_checkpoint(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(cohort_names: tuple[str, ...], workers: int, limit: int | None = None,
        root: Path = ROOT) -> None:
    model, cohorts = load_archived_inputs(root)
    output = root / "results" / "unified_table_heldout"
    for cohort in cohorts:
        if cohort["name"] not in cohort_names:
            continue
        rows = cohort["rows"][:limit] if limit else cohort["rows"]
        destinations = [(row, output / cohort["name"] / f"{row['id']}.json") for row in rows]
        pending = [(row, path) for row, path in destinations if not _checkpoint_valid(
            path, row, cohort["depths"])]
        print(f"[{cohort['name']}] {len(rows)} graphs, {len(pending)} pending; "
              f"depths={cohort['depths']}; workers={workers}", flush=True)
        if workers == 1:
            for index, (row, path) in enumerate(pending, 1):
                _save_checkpoint(path, evaluate_row(row, cohort["depths"], model, cohort["name"]))
                if index % 25 == 0 or index == len(pending):
                    print(f"[{cohort['name']}] {index}/{len(pending)} completed", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(evaluate_row, row, cohort["depths"], model,
                                       cohort["name"]): path for row, path in pending}
                for index, future in enumerate(as_completed(futures), 1):
                    _save_checkpoint(futures[future], future.result())
                    if index % 25 == 0 or index == len(pending):
                        print(f"[{cohort['name']}] {index}/{len(pending)} completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("small", "large", "both"), default="both")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="first N graphs in each cohort, for a speed check")
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("workers and limit must be positive")
    names = ("small", "large") if args.cohort == "both" else (args.cohort,)
    run(names, args.workers, args.limit)


if __name__ == "__main__":
    main()
