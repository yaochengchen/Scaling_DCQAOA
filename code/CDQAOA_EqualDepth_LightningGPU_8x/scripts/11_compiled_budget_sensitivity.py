"""Exploratory diagnostic: re-match QAOA depth after Qiskit level-3 CX cancellation.

The primary locked test remains CD p versus QAOA 3p at optimization level 0/1.
This post-confirmatory analysis uses QAOA 2p because representative level-3
transpilation yields 4*m*p CX for CD and 2*m*p' CX for ordinary QAOA.
Candidate QAOA scaling is selected on validation only, then frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cdtransfer.analysis import paired_method_summary
from cdtransfer.instances import IsingInstance, dump_json, load_json
from cdtransfer.qiskit_optional import transpile_cost
from cdtransfer.simulator import StatevectorSimulator
from cdtransfer.transfer import predict


def source(row: dict) -> IsingInstance:
    keys = ("n", "edges", "weights", "family", "weight_kind", "seed", "split")
    return IsingInstance.from_dict({key: row[key] for key in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    manifest = load_json(out / "manifest.json")
    model = load_json(out / "FROZEN_MODEL.json")
    primary = load_json(out / "heldout_results.json")
    diagnostic = out / "analysis" / "compiled_level3_sensitivity"
    diagnostic.mkdir(parents=True, exist_ok=True)

    selections = {}
    validation_scores = {}
    for p_cd in manifest["config"]["depths"]:
        p_qaoa = 2 * p_cd
        methods = ["qaoa_csv", "qaoa_csv_rms"]
        if str(p_qaoa) in model["templates"]:
            methods.append("qaoa_trained")
        scores = {}
        for method in methods:
            values = []
            for row in manifest["splits"]["val"]:
                inst = source(row)
                energy = StatevectorSimulator(inst).evaluate(predict(inst, p_qaoa, method, model))
                values.append(energy / np.sum(np.abs(inst.weights)))
            scores[method] = float(np.mean(values))
        selections[str(p_cd)] = min(scores, key=scores.get)
        validation_scores[str(p_cd)] = scores
    frozen = {
        "status": "exploratory post-confirmatory resource sensitivity",
        "selection_data": "validation only",
        "qiskit_basis": ["cx", "rz", "sx", "x"],
        "optimization_level": 3,
        "qaoa_depth_rule": "p_qaoa=2*p_cd",
        "qaoa_method_by_cd_depth": selections,
        "validation_normalized_energy": validation_scores,
        "parent_manifest_sha256": manifest["config_sha256"],
    }
    dump_json(diagnostic / "FROZEN_DIAGNOSTIC_MODEL.json", frozen)

    primary_by_key = {(row["id"], row["p_cd"]): row for row in primary}
    rows = []
    for record in manifest["splits"]["test"]:
        inst = source(record)
        sim = StatevectorSimulator(inst)
        for p_cd in manifest["config"]["depths"]:
            parent = primary_by_key[(record["id"], p_cd)]
            p_qaoa = 2 * p_cd
            method = selections[str(p_cd)]
            q_angles = predict(inst, p_qaoa, method, model)
            q_energy = sim.evaluate(q_angles)
            rows.append({
                "id": record["id"], "p_cd": p_cd, "p_qaoa": p_qaoa,
                "n": inst.n, "family": inst.family, "weight_kind": inst.weight_kind,
                "norm_weights": float(np.sum(np.abs(inst.weights))),
                "energy_qaoa_level3_match": q_energy,
                "energy_cd_proposed": parent["energy_cd_proposed"],
                "qaoa_method": method,
            })
    dump_json(diagnostic / "heldout_sensitivity.json", rows)
    summary = {"all": paired_method_summary(rows, "energy_qaoa_level3_match", "energy_cd_proposed")}
    for p_cd in manifest["config"]["depths"]:
        group = [row for row in rows if row["p_cd"] == p_cd]
        summary[f"depth-{p_cd}"] = paired_method_summary(
            group, "energy_qaoa_level3_match", "energy_cd_proposed")

    exemplar = source(manifest["splits"]["test"][0])
    counts = []
    for p_cd in manifest["config"]["depths"]:
        cd = predict(exemplar, p_cd, "cd_proposed", model)
        q = predict(exemplar, 2*p_cd, selections[str(p_cd)], model)
        counts.append({"p_cd": p_cd,
                       "cd": transpile_cost(exemplar, cd, optimization_level=3),
                       "qaoa": transpile_cost(exemplar, q, optimization_level=3)})
    summary["representative_native_counts"] = counts
    summary["interpretation"] = (
        "Exploratory sensitivity selected after the primary 3p comparison; "
        "method choice itself uses validation only."
    )
    dump_json(diagnostic / "summary.json", summary)
    print(summary["all"])


if __name__ == "__main__":
    main()
