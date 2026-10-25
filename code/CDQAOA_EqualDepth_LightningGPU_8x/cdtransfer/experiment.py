"""Train/fit/evaluate/report pipeline; NO test data used in fit functions."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .instances import IsingInstance, dump_json, load_json, make_splits, features
from .simulator import StatevectorSimulator
from .optimization import optimize, template_from_records
from .transfer import predict, qaoa_csv, bch_angles, learned_angles
from .resources import gate_cost, comparable_qaoa_depth


def create_manifest(config: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if path.exists():
        manifest = load_json(path)
        if manifest["config_sha256"] != key:
            raise ValueError("Configuration changed; use a NEW output directory, do not overwrite split")
        return manifest
    manifest = {"config": config, "config_sha256": key, "splits": make_splits(config),
                "protocol": "Target-instance features only at test inference; no test optimization"}
    dump_json(path, manifest)
    return manifest


def _source(inst_record):
    return IsingInstance.from_dict({k: inst_record[k] for k in
                                    ("n","edges","weights","family","weight_kind","seed","split")})


def train_offline(manifest: dict, directory: Path) -> list[dict]:
    """Always train on train split only. Can resume at record-level checkpoint."""
    path = directory / "offline_optima.json"
    existing = load_json(path) if path.exists() else []
    seen = {(row["id"], row["p"], row["cd"], row["restart"]) for row in existing}
    depths = sorted(set(manifest["config"]["depths"] +
                        [comparable_qaoa_depth(p) for p in manifest["config"]["depths"]]))
    restarts = int(manifest["config"]["optimization"]["restarts"])
    iterations = int(manifest["config"]["optimization"]["maxiter"])
    start = time.perf_counter()
    for index, row in enumerate(manifest["splits"]["train"]):
        inst = _source(row)
        for p in depths:
            for cd in ([False, True] if p in manifest["config"]["depths"] else [False]):
                missing = [r for r in range(restarts) if (row["id"],p,cd,r) not in seen]
                if not missing:
                    continue
                for restart in missing:
                    # run one restart per checkpoint; each initialization uses unique seed
                    token = f'{inst.seed}:{p}:{int(cd)}:{restart}'.encode()
                    optseed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
                    res = optimize(inst,p,cd=cd,seed=optseed,restarts=1,maxiter=iterations)[0]
                    record = {"id": row["id"], "p":p, "cd":cd, "restart":restart,
                              "energy":res.energy, "angles":res.angles.to_dict(),
                              "features":features(inst),"weight_kind":inst.weight_kind,
                              "family":inst.family,"n":inst.n,"seed":inst.seed,
                              "eval_calls":res.evaluations,"optimizer_success":res.success,
                              "optimizer_message":res.message,
                              "instance_fingerprint":inst.fingerprint}
                    existing.append(record)
                    seen.add((row["id"],p,cd,restart))
                    dump_json(path,existing)
        if (index+1)%2==0:
            print(f"[train] {index+1}/{len(manifest['splits']['train'])} instances; "
                  f"{len(existing)} optimization records; {time.perf_counter()-start:.1f}s", flush=True)
    return existing


def _fixed_energy(inst, p, method, model, sim=None):
    sim = sim or StatevectorSimulator(inst)
    return sim.evaluate(predict(inst,p,method,model))


def _best_qaoa_method_on_validation(rows: list[dict], model: dict, p: int):
    scores = {}
    for method in ("qaoa_csv", "qaoa_csv_rms", "qaoa_trained"):
        vals = []
        for row in rows:
            inst = _source(row)
            vals.append(_fixed_energy(inst,p,method,model)/max(np.sum(np.abs(inst.weights)),1e-12))
        scores[method] = float(np.mean(vals))
    return min(scores, key=scores.get), scores


def export_fixed_parameter_table(directory: Path, model: dict) -> None:
    """Human-readable, portable CSV/JSON, labels indicate unit-RMS coefficients."""
    import csv
    rows=[]
    for p in model["depths"]:
        temp=model["templates"][str(p)]["cd"]
        for l in range(p):
            rows.append({"p":p,"layer":l+1,
                         "gamma_unit":temp["gamma_unit"][l],
                         "beta":temp["beta"][l],
                         "alpha_unit":temp["alpha_unit"][l],
                         "reference_degree":temp["reference_degree"],
                         "alpha_degree_power":model["alpha_degree_power"],
                         "alpha_multiplier":model["alpha_multiplier"],
                         "source_csv_gamma_sign":model["csv_sign"],
                         "angle_convention":"U_B U_CD U_C; H_C=sum JZZ; little_endian"})
    fname=directory/"cd_transfer_table_v1.csv"
    with fname.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    dump_json(directory/"cd_transfer_table_v1.json",
              {"formula":"gamma=gamma_unit*degree_factor(D)/rms; beta=beta_table; alpha=alpha_unit/rms*(D/D0)**b*c",
               "rows":rows, "zero_target_instance_training":True,
               "caution":"Pilot calibration only until large independent confirmation"})


def fit_on_validation(manifest: dict, records: list[dict], directory: Path):
    """Hyperparameter fit on validation split, THEN freeze immutable JSON."""
    path = directory / "FROZEN_MODEL.json"
    if path.exists():
        model = load_json(path)
        if model.get("manifest_sha256") != manifest["config_sha256"]:
            raise ValueError("Cannot reuse frozen model with modified manifest")
        export_fixed_parameter_table(directory,model)
        return model
    depths = manifest["config"]["depths"]
    expected_depths = sorted(set(depths + [3*p for p in depths]))
    restarts = int(manifest["config"]["optimization"]["restarts"])
    expected_keys = {
        (entry["id"], p, cd, r)
        for entry in manifest["splits"]["train"]
        for p in expected_depths
        for cd in ([False,True] if p in depths else [False])
        for r in range(restarts)
    }
    actual_keys = {(r["id"],r["p"],r["cd"],r["restart"]) for r in records}
    if not expected_keys.issubset(actual_keys):
        raise RuntimeError(f"Offline training incomplete: {len(expected_keys-actual_keys)} records missing. "
                           "Do not freeze/evaluate test before training finishes.")
    qaoa_depths = sorted(set(depths + [3*p for p in depths]))
    templates = {}
    for p in qaoa_depths:
        d = {"qaoa": template_from_records(records,p=p,cd=False)}
        if p in depths:
            d["cd"] = template_from_records(records,p=p,cd=True)
        templates[str(p)] = d
    val = manifest["splits"]["val"]
    # Tune a single CSV Hamiltonian-convention sign on validation, not test.
    losses_by_sign = {}
    for sign in (-1,1):
        model = {"csv_sign": sign}
        losses = []
        for p in qaoa_depths:
            for row in val:
                inst = _source(row)
                losses.append(_fixed_energy(inst,p,"qaoa_csv",model)/sum(abs(inst.weights)))
        losses_by_sign[str(sign)] = float(np.mean(losses))
    csv_sign = int(min(losses_by_sign,key=losses_by_sign.get))
    model = {"version": "0.1.0", "templates": templates, "csv_sign":csv_sign,
             "csv_sign_selection": losses_by_sign, "depths":depths,
             "manifest_sha256":manifest["config_sha256"],
             "no_target_instance_objective_calls": True,
             "protocol": "frozen after validation, before evaluating held-out test"}
    selected = {}
    qaoa_scores = {}
    for p in qaoa_depths:
        selected[str(p)],qaoa_scores[str(p)] = _best_qaoa_method_on_validation(val,model,p)
    model["qaoa_fixed_best_by_depth"] = selected
    model["qaoa_validation_scores_by_depth"] = qaoa_scores
    # Choose low-dimensional CD scaling with VALIDATION (not test).
    grid = []
    for exponent in (-0.5,0.0,0.5,1.0):
        for multiplier in (0.5,1.0,1.5):
            vals=[]
            for p in depths:
                for row in val:
                    inst=_source(row)
                    pred=learned_angles(inst,p,templates,cd=True,
                                        alpha_degree_power=exponent,alpha_multiplier=multiplier)
                    vals.append(StatevectorSimulator(inst).evaluate(pred) / sum(abs(inst.weights)))
            grid.append({"alpha_degree_power":exponent,"alpha_multiplier":multiplier,
                         "validation_normalized_energy":float(np.mean(vals))})
    grid.sort(key=lambda r:r["validation_normalized_energy"])
    model["alpha_degree_power"] = grid[0]["alpha_degree_power"]
    model["alpha_multiplier"] = grid[0]["alpha_multiplier"]
    model["alpha_grid_val"] = grid
    bch_grid=[]
    for c in (-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0):
        vals=[]
        for p in depths:
            for row in val:
                inst=_source(row)
                vals.append(StatevectorSimulator(inst).evaluate(bch_angles(inst,p,c,csv_sign))/sum(abs(inst.weights)))
        bch_grid.append({"c":c,"validation_normalized_energy":float(np.mean(vals))})
    bch_grid.sort(key=lambda r:r["validation_normalized_energy"])
    model["bch_c"] = bch_grid[0]["c"]
    model["bch_val"] = bch_grid
    dump_json(path,model)
    export_fixed_parameter_table(directory,model)
    return model


def evaluate_heldout(manifest: dict, model: dict, directory: Path, *, shots: int | None = None):
    """Read frozen model, use immutable test manifest. No refits or test optimization."""
    path = directory / "heldout_results.json"
    # Resume after interruption; never redo already reported instance-depth pairs.
    output = load_json(path) if path.exists() else []
    tested_pairs = {(r["id"], r["p_cd"]) for r in output}
    if len(tested_pairs) != len(output):
        raise ValueError("Duplicate held-out pair in checkpoint")
    if model.get("manifest_sha256") != manifest["config_sha256"]:
        raise ValueError("Frozen model does not match dataset")
    shots = manifest["config"].get("shots",0) if shots is None else shots
    for record in manifest["splits"]["test"]:
        inst=_source(record)
        sim=StatevectorSimulator(inst)
        z=float(sum(abs(inst.weights)))
        for p in manifest["config"]["depths"]:
            if (record["id"], p) in tested_pairs:
                continue
            p_budget=comparable_qaoa_depth(p)
            qaoa_best_p=model["qaoa_fixed_best_by_depth"][str(p)]
            qaoa_best_budget=model["qaoa_fixed_best_by_depth"][str(p_budget)]
            methods = {
                "qaoa_csv_same": (p,"qaoa_csv"),
                "qaoa_best_same": (p,qaoa_best_p),
                "qaoa_csv_budget": (p_budget,"qaoa_csv"),
                "qaoa_best_budget": (p_budget,qaoa_best_budget),
                "cd_zero": (p,"cd_zero"),
                "cd_bch": (p,"cd_bch"),
                "cd_reference": (p,"cd_reference"),
                "cd_proposed": (p,"cd_proposed"),
            }
            result={"id":record["id"],"p_cd":p,"p_qaoa_budget":p_budget,
                    "n":inst.n,"m":inst.m,"family":inst.family,"weight_kind":inst.weight_kind,
                    "seed":inst.seed,"norm_weights":z,
                    "rms":features(inst)["rms"],"degree":features(inst)["degree"],
                    "ground_energy":sim.brute_force_min() if inst.n<=18 else None}
            for name,(depth,method) in methods.items():
                angle=predict(inst,depth,method,model)
                result[f"energy_{name}"]=sim.evaluate(angle)
                result[f"cx_{name}"]=gate_cost(inst,angle)["decomposed_cx"]
                if shots and name in ("qaoa_best_budget","cd_proposed"):
                    sample_seed=int.from_bytes(hashlib.sha256(f"{record['seed']}:{name}:{p}".encode()).digest()[:4],"big")
                    result[f"best_{shots}_shots_{name}"] = float(np.min(sim.sample_energies(angle,shots,sample_seed)))
            result["gain_equal_budget_norm"] = (result["energy_qaoa_best_budget"] -
                                                 result["energy_cd_proposed"])/z
            result["gain_same_depth_norm"] = (result["energy_qaoa_best_same"] -
                                               result["energy_cd_proposed"])/z
            result["gain_over_csv_budget_norm"] = (result["energy_qaoa_csv_budget"] -
                                                   result["energy_cd_proposed"])/z
            result["costs_equal_budget"] = bool(result["cx_qaoa_best_budget"] == result["cx_cd_proposed"])
            output.append(result)
            tested_pairs.add((record["id"], p))
            dump_json(path, output)
        print(f"[heldout] tested {record['id']}",flush=True)
    return output


def bootstrap_ci(samples: list[float], *, n_boot: int = 2000, seed: int = 1441):
    x=np.asarray(samples,dtype=float)
    if x.size==0:return [None,None]
    if x.size==1:return [float(x[0]),float(x[0])]
    rng=np.random.default_rng(seed)
    boot=np.mean(x[rng.integers(x.size,size=(n_boot,x.size))],axis=1)
    return [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]


def summarize_results(rows: list[dict]):
    by_group=defaultdict(list)
    for r in rows:
        for key in ("all",f"depth-{r['p_cd']}",f"family-{r['family']}",f"weights-{r['weight_kind']}"):
            by_group[key].append(r)
    res={}
    for key,group in by_group.items():
        gain=[r["gain_equal_budget_norm"] for r in group]
        res[key]={"n":len(group),"mean_gain":float(np.mean(gain)),
                  "median_gain":float(np.median(gain)),
                  "bootstrap_95_ci":bootstrap_ci(gain),
                  "win_rate":float(np.mean(np.asarray(gain)>1e-9)),
                  "tie_rate":float(np.mean(abs(np.asarray(gain))<=1e-9)),
                  "all_equal_budget_decomposition":bool(all(r["costs_equal_budget"] for r in group))}
    return res


def report(directory: Path, manifest: dict, records: list[dict], model: dict, test: list[dict]):
    from .plots import make_figures
    summary=summarize_results(test)
    dump_json(directory/"summary.json",summary)
    figure_names=make_figures(directory,records,test)
    overall=summary["all"]
    expected_test_pairs=len(manifest["splits"]["test"])*len(manifest["config"]["depths"])
    tests_complete=(len(test)==expected_test_pairs)
    ntrain=len(manifest["splits"]["train"])
    nval=len(manifest["splits"]["val"])
    ntest=len(manifest["splits"]["test"])
    stats=[
        "# Initial experimental report — AUTOMATED PILOT, not a paper result",
        "",
        f"- Dataset: train={ntrain}, validation={nval}, test={ntest}; independent random seeds (not necessarily non-isomorphic graphs).",
        f"- Held-out evaluation complete: {tests_complete} ({len(test)}/{expected_test_pairs} pairs).",
        f"- Offline optimization records: {len(records)} (including restarts and QAOA/CD budgets).",
        f"- CD depths: {manifest['config']['depths']}; equal-budget ordinary QAOA depth = 3 × CD depth.",
        "- Two-qubit budget uses **fixed unoptimized 2-CX-per-Pauli-rotation decomposition**, NOT hardware-transpiled counts.",
        f"- Validation-selected CSV sign: {model['csv_sign']}; alpha degree exponent: {model['alpha_degree_power']}; alpha multiplier: {model['alpha_multiplier']}; BCH c: {model['bch_c']}.",
        "- Fixed model is frozen before test: FROZEN_MODEL.json; no test energy used to predict angles.",
        "",
        "## Held-out results (normalized signed Ising energy, lower is better)",
        f"- Paired gain (QAOA budget fixed − CD proposed): mean **{overall['mean_gain']:.6f}**, median **{overall['median_gain']:.6f}**.",
        f"- Bootstrap 95% confidence interval: [{overall['bootstrap_95_ci'][0]:.6f}, {overall['bootstrap_95_ci'][1]:.6f}].",
        f"- CD win rate: **{overall['win_rate']:.1%}** over {overall['n']} paired tests.",
        "",
        "| Group | Pairs | Mean normalized gain | Win rate | 95% bootstrap CI |",
        "|---|---:|---:|---:|---|",
    ]
    for key,s in summary.items():
        stats.append(f"| {key} | {s['n']} | {s['mean_gain']:+.6f} | {s['win_rate']:.1%} | "
                     f"[{s['bootstrap_95_ci'][0]:+.4f}, {s['bootstrap_95_ci'][1]:+.4f}] |")
    stats.extend(["", "## Interpretation and limitations",
                  "- Positive gain indicates proposed fixed CD beats the validation-selected fixed QAOA baseline under a matched unoptimized gate decomposition.",
                  "- A negative gain does **not** support the main performance claim; report failures rather than changing test data.",
                  "- All tests may be too small for statistical reliability; confirm on larger independent seeds and actual native-basis transpilation.",
                  "- The offline CD optimization may be trapped in local minima; multi-start and optimization-budget convergence studies remain necessary.",
                  "- This pilot provides a complete pipeline, not validated new physics or publishable evidence by itself.",
                  "- GPU scale-up intentionally deferred to collaborator.",
                  "", "## Figures", "",*[f"- `{name}`" for name in figure_names], "",
                  "## Next concrete actions", "",
                  "1. Validate simulator Pauli-rotation matrix conventions and compare against Qulacs/Aer when available.",
                  "2. Expand independent n=8–16 discovery training with multi-start and collect normalized angle distributions.",
                  "3. Lock validation-fit model and test across independent n=12–20 / OOD distributions with uncertainty intervals.",
                  "4. Verify actual CX/CZ compiled budgets; add matching-depth QAOA and noise robustness.",
                  "5. Draft manuscript based on observed positive **or** negative evidence; larger GPU jobs are optional.",
                  ])
    (directory/"EXPERIMENT_REPORT.md").write_text("\n".join(stats)+"\n",encoding="utf-8")
    return summary
