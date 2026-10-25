from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from cdtransfer.instances import IsingInstance, dump_json, features
from cdtransfer.maxcut import expected_cut_from_ising, solve_maxcut_verified
from cdtransfer.simulator import StatevectorSimulator
from cdtransfer.transfer import predict


def unique_cells(structure_cells, weight_cells):
    return list(dict.fromkeys(tuple(x) for x in [*structure_cells, *weight_cells]))


def deterministic_seed(
    base: int, n: int, family: str, weight: str, sample: int
) -> int:
    token = f"{base}:{n}:{family}:{weight}:{sample}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def evaluate_instance(
    case_id: str, inst: IsingInstance, depths: list[int], model: dict
) -> dict:
    exact = solve_maxcut_verified(inst)
    simulator = StatevectorSimulator(inst)
    evaluations = {}
    for p in depths:
        depth_results = {}
        for method in ("qaoa_csv", "cd_proposed"):
            angles = predict(inst, p, method, model)
            energy = simulator.evaluate(angles)
            cut = expected_cut_from_ising(inst, energy)
            ratio = cut / exact.optimum
            if not np.isfinite(ratio) or ratio < -1e-10 or ratio > 1.0 + 1e-10:
                raise RuntimeError(
                    f"Invalid approximation ratio {ratio} for {case_id}"
                )
            depth_results[method] = {
                "angles": angles.to_dict(),
                "ising_energy": energy,
                "expected_cut": cut,
                "approximation_ratio": min(1.0, max(0.0, float(ratio))),
            }
        evaluations[str(p)] = depth_results
    return {
        "id": case_id,
        "instance": inst.to_dict(),
        "fingerprint": inst.fingerprint,
        "features": features(inst),
        "exact": exact.to_dict(),
        "evaluations": evaluations,
    }


def save_records_atomic(path: Path, records: list[dict]) -> None:
    dump_json(path, records)
