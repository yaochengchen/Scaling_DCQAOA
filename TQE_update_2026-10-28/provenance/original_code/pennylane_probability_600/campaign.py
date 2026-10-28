"""Frozen PennyLane low-degree and 600-graph p=6 campaign with resumable method records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "pennylane_probability_workers"))
import core  # noqa: E402; sets all numeric libraries to one thread before import

OUT = ROOT / "results" / "pennylane_probability_low_degree_p6_600"
SOURCE = ROOT / "results" / "pennylane_probability_1050"
P6_SOURCE = ROOT / "results" / "u1_p6_small_4000_v2"
LOW_SOURCE = ROOT / "results" / "u1_low_degree_55"
DEPTHS = (1, 2, 3, 4, 6)
KINDS = ("u1", "old", "qaoa")
ONE_WORKER_BUNDLE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def taskkey(gid, p):
    return f"{gid}-p{p}"


def path_for(gid, p, kind):
    return OUT / "records" / f"{taskkey(gid, p)}-{kind}.json"


def status(stage, **kwargs):
    core.atomic_json(OUT / "status.json", {"stage": stage, "updated_unix": time.time(),
                                           "pid": os.getpid(), **kwargs})


def source_paths():
    return {
        "main_manifest": SOURCE / "manifest.json",
        "small_graphs": ROOT / "inputs" / "eqdepth_small_N20_manifest.json.gz",
        "low_manifest": LOW_SOURCE / "frozen_manifest.json",
        "p6_manifest": P6_SOURCE / "frozen_manifest.json",
        "energy_reference": P6_SOURCE / "paired_records_32000.csv",
        "powerlaw_model": ROOT / "models" / "U1_power_law.json",
        "core": ROOT / "code" / "pennylane_probability_workers" / "core.py",
        "code": Path(__file__),
    }


def load_bundle():
    b = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    copy = dict(b)
    digest = copy.pop("campaign")
    if core.object_digest(copy) != digest:
        raise ValueError("Campaign manifest checksum mismatch")
    for meta in b["source_hashes"].values():
        if sha(ROOT / meta["path"]) != meta["sha256"]:
            raise ValueError(f"Frozen source changed: {meta['path']}")
    if b["backend"] != core.BACKEND or b["runtime"]["pennylane"] != core.qml.__version__:
        raise ValueError("PennyLane runtime changed")
    return b


def prepare():
    if (OUT / "manifest.json").exists():
        return load_bundle()
    paths = source_paths()
    main = json.loads(paths["main_manifest"].read_text(encoding="utf-8"))
    selected = [g for g in main["graphs"] if g["n"] in (10, 12, 14, 16)]
    selected_ids = {g["id"] for g in selected}
    if len(selected) != len(selected_ids) or len(selected) != 600:
        raise ValueError("Expected exactly 600 distinct frozen main-sample graphs")
    for n in (10, 12, 14, 16):
        group = [g for g in selected if g["n"] == n]
        if len(group) != 150 or len({(g["family"], g["weight_kind"]) for g in group}) != 50:
            raise ValueError("Unexpected balanced size stratum")
    low_archive = json.loads(paths["low_manifest"].read_text(encoding="utf-8"))
    low_ids = {t["id"] for t in low_archive["tasks"]}
    if len(low_ids) != 11 or len(low_archive["tasks"]) != 55:
        raise ValueError("Expected 11 low-degree graphs and 55 tasks")
    with gzip.open(paths["small_graphs"], "rt", encoding="utf-8") as handle:
        all_graphs = {g["id"]: g for g in json.load(handle)["splits"]["test"]}
    if not selected_ids | low_ids <= all_graphs.keys():
        raise ValueError("Some frozen graph identities are not in the test manifest")
    graphs = {gid: all_graphs[gid] for gid in sorted(selected_ids | low_ids)}
    for gid, g in graphs.items():
        if core.fingerprint(g) != g["fingerprint"]:
            raise ValueError(f"Graph fingerprint changed: {gid}")
    for g in selected:
        if graphs[g["id"]] != g:
            raise ValueError(f"Main-sample graph changed: {g['id']}")
    if len(selected_ids & low_ids) != 1:
        raise ValueError("Expected exactly one low-degree graph in the 600-graph cohort")
    p6_archive = json.loads(paths["p6_manifest"].read_text(encoding="utf-8"))
    p6_tasks = {t["id"]: t for t in p6_archive["tasks"]}
    if len(p6_tasks) != 4000:
        raise ValueError("The p=6 reference archive is incomplete")
    for gid, g in graphs.items():
        if p6_tasks[gid]["fingerprint"] != g["fingerprint"]:
            raise ValueError(f"p=6 graph fingerprint mismatch: {gid}")

    requirements = {}
    for gid in selected_ids:
        requirements[taskkey(gid, 6)] = ["u1", "qaoa"]
    for gid in low_ids:
        for p in DEPTHS:
            requirements[taskkey(gid, p)] = list(KINDS)
    if len(requirements) != 654 or sum(len(v) for v in requirements.values()) != 1363:
        raise ValueError("Unexpected scientific task counts")
    references = {}
    with paths["energy_reference"].open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = taskkey(row["id"], int(row["p"]))
            if key in requirements:
                if key in references:
                    raise ValueError(f"Duplicate energy reference {key}")
                references[key] = {kind: float(row[f"energy_{kind}"]) for kind in KINDS}
    if len(references) != len(requirements):
        raise ValueError("Missing energy reference")
    models = main["models"]
    if main["source_hashes"]["u1"]["sha256"] != sha(paths["powerlaw_model"]):
        raise ValueError("Main-sample model is not the current power law")
    runtime = {"pennylane": core.qml.__version__, "numpy": core.np.__version__,
               "dtype": "complex128", "shots": None, "backend": core.BACKEND,
               "omp_threads_per_worker": 1}
    b = {"selected_ids": sorted(selected_ids), "low_ids": sorted(low_ids),
         "graphs": graphs, "p6_extrema": {gid: {"min": p6_tasks[gid]["ground_energy"],
                                              "max": p6_tasks[gid]["max_energy"]} for gid in graphs},
         "requirements": requirements, "references": references, "models": models,
         "backend": core.BACKEND, "runtime": runtime, "protocol": core.PROTOCOL,
         "source_hashes": {kind: {"path": os.path.relpath(path, ROOT), "sha256": sha(path)}
                           for kind, path in paths.items()},
         "imported_campaign": main["campaign"]}
    b["campaign"] = core.object_digest(b)
    OUT.mkdir(parents=True, exist_ok=True)
    core.atomic_json(OUT / "manifest.json", b)
    status("prepared", campaign=b["campaign"], methods_total=1363)
    return b


def expected(b, gid, p, kind):
    g = b["graphs"][gid]
    angles = core.predict(g, p, kind, b["models"])
    angle_list = [v.tolist() for v in angles]
    return {"campaign": b["campaign"], "id": gid, "p": p, "method": kind,
            "fingerprint": g["fingerprint"], "backend": core.BACKEND,
            "dtype": "complex128", "shots": None,
            "angles_sha256": core.object_digest(angle_list)}, angles


def check_method(b, gid, p, kind):
    target = path_for(gid, p, kind)
    record = json.loads(target.read_text(encoding="utf-8"))
    ident, _ = expected(b, gid, p, kind)
    for key, value in ident.items():
        if record.get(key) != value:
            raise ValueError(f"Method record identity mismatch: {target} {key}")
    payload = dict(record)
    if payload.pop("content_sha256", None) != core.object_digest(payload):
        raise ValueError(f"Method record checksum mismatch: {target}")
    m = record["result"]
    if any(not core.math.isfinite(float(m.get(key, float("nan"))))
           for key in ("probability", "energy", "ratio", "norm_error")):
        raise ValueError(f"Nonfinite method result: {target}")
    if not 0 <= m["probability"] <= 1 or m["norm_error"] > 1e-9:
        raise ValueError(f"Invalid probability or normalization: {target}")
    ref = b["references"][taskkey(gid, p)][kind]
    if abs(m["energy"] - ref) > 1e-7 * max(1, sum(b["graphs"][gid]["weights"])):
        raise ValueError(f"Method energy differs from frozen reference: {target}")
    return record


def imported_record(b, gid, p, kind):
    source = SOURCE / "records" / f"{taskkey(gid, p)}.json"
    r = json.loads(source.read_text(encoding="utf-8"))
    payload = dict(r)
    if payload.pop("content_sha256", None) != core.object_digest(payload):
        raise ValueError(f"Source record checksum mismatch: {source}")
    g = b["graphs"][gid]
    if r["campaign"] != b["imported_campaign"] or r["fingerprint"] != g["fingerprint"] or r["backend"] != core.BACKEND:
        raise ValueError(f"Imported source identity mismatch: {source}")
    ident, angles = expected(b, gid, p, kind)
    old_angles = r["angles"][kind]
    if max(abs(float(x) - float(y)) for left, right in zip(angles, old_angles)
           for x, y in zip(left, right)) > 1e-12:
        raise ValueError(f"Imported angles differ from frozen rule: {source}")
    record = {**ident, "optimal_states": r["optimal_states"], "result": r["methods"][kind],
              "provenance": {"type": "imported_pennylane_record", "path": str(source.relative_to(ROOT)),
                             "sha256": sha(source), "source_campaign": r["campaign"]}}
    record["content_sha256"] = core.object_digest(record)
    return record


def import_reusable(b):
    imported = 0
    for gid in b["low_ids"]:
        if gid not in b["selected_ids"]:
            continue
        for p in (1, 2, 3, 4):
            for kind in KINDS:
                target = path_for(gid, p, kind)
                if target.exists():
                    check_method(b, gid, p, kind)
                    continue
                core.atomic_json(target, imported_record(b, gid, p, kind))
                check_method(b, gid, p, kind)
                imported += 1
    return imported


def legacy_probabilities(b, gid, p):
    g = b["graphs"][gid]
    source = ROOT / "results" / "optimal_probability_u1_old_qaoa" / "records" / str(g["n"]) / f"{taskkey(gid, p)}.json"
    if not source.exists():
        return None
    r = json.loads(source.read_text(encoding="utf-8"))
    if r["fingerprint"] != g["fingerprint"] or r["model_sha256"] != b["source_hashes"]["powerlaw_model"]["sha256"]:
        raise ValueError(f"Legacy probability identity mismatch: {source}")
    return r["prob"]


def run_one(gid, p):
    b = ONE_WORKER_BUNDLE
    methods = b["requirements"][taskkey(gid, p)]
    missing = []
    for kind in methods:
        if path_for(gid, p, kind).exists():
            check_method(b, gid, p, kind)
        else:
            missing.append(kind)
    if not missing:
        return {"id": gid, "p": p, "new": 0}
    g = b["graphs"][gid]
    spectrum = core.spectrum(g)
    mask = core.optimal_mask(spectrum, sum(g["weights"]))
    extrema = b["p6_extrema"][gid]
    if abs(spectrum.min() - extrema["min"]) > 1e-8 or abs(spectrum.max() - extrema["max"]) > 1e-8:
        raise ValueError(f"Spectrum disagrees with archived exact extrema: {gid}")
    old = legacy_probabilities(b, gid, p)
    for kind in missing:
        ident, angles = expected(b, gid, p, kind)
        t0 = time.perf_counter()
        psi = core.make_state_function(g, p, kind != "qaoa")(*angles)
        result = core.measure(psi, spectrum, mask)
        result["seconds"] = time.perf_counter() - t0
        ref = b["references"][taskkey(gid, p)][kind]
        error = abs(result["energy"] - ref)
        if error > 1e-7 * max(1, sum(g["weights"])):
            raise ValueError(f"Energy reference mismatch: {gid} p={p} {kind}: {error}")
        result["energy_reference_error"] = error
        if old is not None:
            drift = abs(result["probability"] - old[kind])
            if drift > 1e-8:
                raise ValueError(f"Legacy probability crosscheck mismatch: {gid} p={p} {kind}: {drift}")
            result["legacy_probability_error"] = drift
        record = {**ident, "optimal_states": int(mask.sum()), "result": result,
                  "provenance": {"type": "computed_pennylane", "pid": os.getpid(),
                                 "omp_threads": 1, "source": "new_campaign"}}
        record["content_sha256"] = core.object_digest(record)
        core.atomic_json(path_for(gid, p, kind), record)
        check_method(b, gid, p, kind)
    return {"id": gid, "p": p, "new": len(missing)}


def worker_init():
    global ONE_WORKER_BUNDLE
    ONE_WORKER_BUNDLE = load_bundle()


def inventory(b):
    imported = computed = missing = 0
    records = []
    for key, methods in b["requirements"].items():
        gid, p_text = key.rsplit("-p", 1)
        p = int(p_text)
        for kind in methods:
            path = path_for(gid, p, kind)
            if path.exists():
                r = check_method(b, gid, p, kind)
                records.append(r)
                if r["provenance"]["type"] == "imported_pennylane_record":
                    imported += 1
                else:
                    computed += 1
            else:
                missing += 1
    expected_names = {path_for(k.rsplit("-p", 1)[0], int(k.rsplit("-p", 1)[1]), method).name
                      for k, methods in b["requirements"].items() for method in methods}
    extras = [p.name for p in (OUT / "records").glob("*.json") if p.name not in expected_names]
    if extras:
        raise ValueError(f"Unexpected method files: {extras[:3]}")
    return {"imported": imported, "computed": computed, "missing": missing, "records": records}


def run(b, workers):
    global ONE_WORKER_BUNDLE
    OUT.joinpath("records").mkdir(parents=True, exist_ok=True)
    imported = import_reusable(b)
    ONE_WORKER_BUNDLE = b
    low_tasks = [(gid, p) for gid in b["low_ids"] for p in DEPTHS]
    status("running_low", campaign=b["campaign"], imported_this_run=imported, total_low=len(low_tasks))
    low_new = 0
    for i, (gid, p) in enumerate(low_tasks, 1):
        if OUT.joinpath("STOP").exists():
            status("paused", phase="low", completed=i - 1)
            return
        low_new += run_one(gid, p)["new"]
        if i % 5 == 0:
            status("running_low", completed=i, total_low=55, methods_new=low_new, campaign=b["campaign"])
    selected = set(b["selected_ids"])
    p6_tasks = sorted(((gid, 6) for gid in selected),
                      key=lambda task: (-b["graphs"][task[0]]["n"], task[0]))
    p6_tasks = [(gid, p) for gid, p in p6_tasks if any(not path_for(gid, p, kind).exists()
                                                        for kind in ("u1", "qaoa"))]
    status("running_p6", total_graphs=600, tasks_remaining=len(p6_tasks), methods_new=low_new,
           workers=workers, campaign=b["campaign"])
    done = 0
    new = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = {pool.submit(run_one, gid, p): (gid, p) for gid, p in p6_tasks}
        for future in as_completed(futures):
            answer = future.result()
            done += 1
            new += answer["new"]
            if done % 20 == 0 or done == len(p6_tasks):
                status("running_p6", completed=done, total=len(p6_tasks), methods_new=low_new + new,
                       workers=workers, last=answer["id"], campaign=b["campaign"])
                print(json.dumps({"completed": done, "total": len(p6_tasks),
                                  "methods_new": low_new + new, "last": answer["id"]}), flush=True)
    status("simulation_complete", methods_new=low_new + new, campaign=b["campaign"])


def verify(b):
    inv = inventory(b)
    if inv["missing"]:
        raise ValueError(f"Incomplete methods: {inv['missing']}/1363 missing")
    low = [r for r in inv["records"] if r["id"] in b["low_ids"]]
    p6 = [r for r in inv["records"] if r["id"] in b["selected_ids"] and r["p"] == 6 and r["method"] in ("u1", "qaoa")]
    if len(low) != 165 or len(p6) != 1200:
        raise ValueError(f"Wrong report scopes: low {len(low)}, p6 {len(p6)}")
    report = {"campaign": b["campaign"], "backend": core.BACKEND,
              "total_method_records": len(inv["records"]), "imported": inv["imported"],
              "computed": inv["computed"], "missing": inv["missing"],
              "low_graphs": 11, "low_graph_depth_records": 55, "low_method_records": len(low),
              "p6_graphs": 600, "p6_method_records": len(p6),
              "max_energy_reference_error": max(r["result"].get("energy_reference_error", 0) for r in inv["records"]),
              "max_legacy_probability_error": max(r["result"].get("legacy_probability_error", 0) for r in inv["records"]),
              "max_norm_error": max(r["result"]["norm_error"] for r in inv["records"])}
    core.atomic_json(OUT / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pipeline", "run", "verify", "plot", "status"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be 1 through 12")
    if args.command == "status":
        print((OUT / "status.json").read_text(encoding="utf-8"))
        return
    b = prepare() if args.command in ("prepare", "pipeline") else load_bundle()
    if args.command == "prepare":
        print(json.dumps({"campaign": b["campaign"], "graphs_p6": len(b["selected_ids"]),
                          "low_graphs": len(b["low_ids"]), "methods_total": 1363}))
        return
    if args.command in ("pipeline", "run"):
        run(b, args.workers)
        if OUT.joinpath("STOP").exists():
            return
    if args.command in ("pipeline", "verify", "plot"):
        report = verify(b)
        print(json.dumps(report), flush=True)
    if args.command in ("pipeline", "plot"):
        import plots
        plots.render(b)
        status("complete", campaign=b["campaign"], methods_total=1363)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    main()
