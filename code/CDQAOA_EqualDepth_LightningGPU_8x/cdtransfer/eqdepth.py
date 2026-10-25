"""Equal-depth CD-QAOA study on multi-GPU PennyLane Lightning (config "comparison": "same_depth").

Research question: do optimized (gamma, beta, alpha) concentrate / follow scaling
laws at larger n, deeper p, more graph families and positive-weight shapes, and
can the fixed (training-free) formula be improved?  Comparisons are CD p vs QAOA p
at the SAME number of layers; no 3p QAOA budget runs are produced in this mode.

Stages (each resumable, each writes into one output directory):
  train    per training instance, ascending depth chain: QAOA restarts then CD restarts
  fit      templates from train optima; hyper-parameters chosen on validation; FROZEN_MODEL.json
  evaluate fixed formulas on the held-out test split (no test optimization) + exact E_min/E_max
  report   summary tables, bootstrap CIs, figures

Data isolation: train -> templates, val -> hyper-parameters, test -> evaluated once after freezing.
"""
from __future__ import annotations
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
from .instances import (IsingInstance, dump_json, load_json, features, make_splits_stable,
                        exact_extrema)

MODE = "same_depth"
V1_MODEL = Path(__file__).resolve().parents[1] / "assets" / "CD_TRANSFER_V1_FROZEN_MODEL.json"


def is_eqdepth(config: dict) -> bool:
    return config.get("comparison") == MODE


def _row_instance(row) -> IsingInstance:
    return IsingInstance.from_dict({k: row[k] for k in
                                    ("n", "edges", "weights", "family", "weight_kind", "seed", "split")})


def _opt(config):
    o = config["optimization"]
    return {"restarts": int(o["restarts"]), "maxiter": int(o["maxiter"]),
            "ftol": float(o.get("ftol", 1e-10)), "gtol": float(o.get("gtol", 1e-5)),
            "optimizer": "L-BFGS-B", "gradient": "adjoint"}


def _reference_settings(settings: dict) -> dict:
    """Same backend in complex128: used to re-check optimized energies and for all fixed-angle
    evaluations (fit/test), so reported energies and ratios never carry complex64 rounding."""
    return dict(settings, dtype="complex128")


def _exact_max_n(config) -> int:
    return int(config.get("exact_max_n", 24))


# ------------------------------------------------------------------ manifest
RUNTIME_KEYS = ("gpus", "slots_per_gpu", "task_weight_by_n", "workers_per_gpu")


def _science(config: dict) -> dict:
    """Config without pure scheduling options; those may change between reruns."""
    c = json.loads(json.dumps(config))
    c.pop("_note", None)
    for k in RUNTIME_KEYS:
        c.get("execution", {}).pop(k, None)
    return c


def create_manifest(config: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if path.exists():
        manifest = load_json(path)
        if _science(manifest["config"]) != _science(config):
            raise ValueError("Scientific configuration changed (instances, depths, optimizer or backend); "
                             "use a NEW output directory (--reuse-from OLD_DIR imports finished work)")
        return manifest      # only scheduling options (GPU ids, slots) differ: keep the stored identity
    splits = make_splits_stable(config)
    split_hash = {s: hashlib.sha256("\n".join(r["fingerprint"] for r in rows).encode()).hexdigest()
                  for s, rows in splits.items()}
    manifest = {"config": config, "config_sha256": key, "splits": splits,
                "split_fingerprint_sha256": split_hash, "id_scheme": "stable_v1",
                "comparison": MODE,
                "protocol": "train->templates, val->hyper-parameters, freeze, then test once; "
                            "target-instance static features only at inference"}
    dump_json(path, manifest)
    for split, rows in splits.items():
        with (directory / f"split_{split}.csv").open("w", encoding="utf-8") as f:
            f.write("id,n,m,family,weight_kind,seed,fingerprint\n")
            for r in rows:
                f.write(f"{r['id']},{r['n']},{len(r['edges'])},{r['family']},{r['weight_kind']},"
                        f"{r['seed']},{r['fingerprint']}\n")
    return manifest


# ------------------------------------------------------------- training plan
def instance_plan(config: dict) -> list[tuple[int, bool, int, str]]:
    from .optimization import start_plan
    depths = sorted(config["depths"])
    restarts = _opt(config)["restarts"]
    plan = []
    for k, p in enumerate(depths):
        for cd in (False, True):
            for r, style in enumerate(start_plan(p, cd, restarts, has_prev=k > 0)):
                plan.append((p, cd, r, style))
    return plan


def _key(p, cd, r):
    return f"p{p}_{'cd' if cd else 'qaoa'}_r{r}"


def _ckpt(directory, row_id, p, cd, r):
    return Path(directory) / "checkpoints" / row_id / f"{_key(p, cd, r)}.json"


def _trace(directory, row_id, p, cd, r):
    return Path(directory) / "traces" / row_id / f"{_key(p, cd, r)}.jsonl.gz"


def _instance_meta(directory, row, exact_max_n):
    """Exact extrema (CPU exhaustive enumeration), cached per instance."""
    path = Path(directory) / "checkpoints" / row["id"] / "instance.json"
    if path.exists():
        meta = load_json(path)
        if meta["fingerprint"] == row["fingerprint"]:
            return meta
    inst = _row_instance(row)
    meta = {"id": row["id"], "fingerprint": row["fingerprint"], "exact": False,
            "ground_energy": None, "max_energy": None}
    if inst.n <= exact_max_n:
        t = time.perf_counter()
        lo, hi = exact_extrema(inst)
        meta.update(exact=True, ground_energy=lo, max_energy=hi,
                    exact_method="exhaustive enumeration of 2^(n-1) spin states (global flip symmetry)",
                    exact_seconds=time.perf_counter() - t)
    dump_json(path, meta)
    return meta


def _ratio(energy, meta):
    if not meta.get("exact") or meta["max_energy"] - meta["ground_energy"] <= 1e-12:
        return None
    return (meta["max_energy"] - energy) / (meta["max_energy"] - meta["ground_energy"])


def _ratio_kind(inst):
    return ("maxcut_approximation_ratio" if np.all(inst.weights > 0)
            else "ising_relative_energy (E_max-E)/(E_max-E_min)")


# ------------------------------------------------------- train task (one instance)
def train_instance(row, ctx, gpu):
    """Handler run inside a gpu_pool worker: the full depth chain of one training instance.

    Already-finished optimizations (checkpoints) are loaded, not recomputed, so an
    interrupted instance resumes exactly where it stopped.
    """
    from .lightning import make_simulator
    from .optimization import ramp_start, random_start, interp_start, run_lbfgs
    directory, config, settings = Path(ctx["directory"]), ctx["config"], ctx["settings"]
    opt, plan, depths = _opt(config), instance_plan(config), sorted(config["depths"])
    started = time.perf_counter()
    try:
        inst = _row_instance(row)
        meta = _instance_meta(directory, row, _exact_max_n(config))
        best: dict[tuple[int, bool], tuple[float, np.ndarray]] = {}
        new, sim, ref = 0, None, None
        dtype = settings.get("dtype", "complex128")
        for p, cd, r, style in plan:
            path = _ckpt(directory, row["id"], p, cd, r)
            if path.exists():
                rec = load_json(path)
                x = np.asarray(rec["angles"]["gamma"] + rec["angles"]["beta"] +
                               (rec["angles"]["alpha"] if cd else []))
            else:
                if sim is None:
                    sim = make_simulator(inst, settings)
                token = f"{inst.seed}:{p}:{int(cd)}:{r}".encode()
                optseed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
                rng = np.random.default_rng(optseed)
                k = depths.index(p)
                used = style
                if style == "from_qaoa" and (p, False) in best:
                    x0 = np.concatenate([best[(p, False)][1], np.zeros(p)])
                elif style == "interp" and k > 0 and (depths[k-1], cd) in best:
                    x0 = interp_start(best[(depths[k-1], cd)][1], depths[k-1], p, cd)
                elif style == "ramp":
                    x0 = ramp_start(inst, p, cd)
                else:
                    used = "random" if style == "random" else f"random(fallback:{style})"
                    x0 = random_start(inst, p, cd, rng)
                tpath = _trace(directory, row["id"], p, cd, r)
                tpath.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                tmp = tpath.with_name(tpath.name + ".partial")
                with gzip.open(tmp, "wt", compresslevel=3) as trace:
                    def log(entry):
                        entry["elapsed_seconds"] = time.perf_counter() - t0
                        trace.write(json.dumps(entry, allow_nan=False) + "\n")
                    res = run_lbfgs(inst, p, cd, x0, sim, maxiter=opt["maxiter"], trace_callback=log,
                                    ftol=opt["ftol"], gtol=opt["gtol"])
                tmp.replace(tpath)
                energy_opt = res.energy
                if dtype != "complex128":
                    if ref is None:
                        ref = make_simulator(inst, _reference_settings(settings))
                    res.energy = ref.evaluate(res.angles)
                rec = {"id": row["id"], "p": p, "cd": cd, "restart": r, "init_style": used,
                       "energy": res.energy, "energy_dtype": "complex128",
                       "energy_optimizer_dtype": energy_opt, "angles": res.angles.to_dict(),
                       "initial_angles": res.initial_angles,
                       "normalized_energy": res.energy / float(np.sum(np.abs(inst.weights))),
                       "ground_energy": meta["ground_energy"], "max_energy": meta["max_energy"],
                       "ratio": _ratio(res.energy, meta), "ratio_kind": _ratio_kind(inst),
                       "features": features(inst), "graph_features": row.get("graph_features"),
                       "family": inst.family, "weight_kind": inst.weight_kind, "n": inst.n,
                       "m": inst.m, "seed": inst.seed, "optimization_seed": optseed,
                       "instance_fingerprint": inst.fingerprint,
                       "eval_calls": res.evaluations, "gradient_calls": res.gradient_evaluations,
                       "optimizer": "L-BFGS-B", "maxiter": opt["maxiter"],
                       "ftol": opt["ftol"], "gtol": opt["gtol"],
                       "optimizer_success": res.success, "optimizer_message": res.message,
                       "backend": settings["backend"], "dtype": dtype,
                       "engine": settings.get("engine", "tape"), "gpu": str(gpu), "gpu_slots_used": ctx.get("weights", {}).get(str(inst.n)),
                       "gpu_slots_total": ctx.get("slots"),
                       "wall_seconds": time.perf_counter() - t0, "finished_unix": time.time(),
                       "trace_path": str(tpath.relative_to(directory)),
                       "config_sha256": config.get("_sha256"),
                       "cost_accounting": "eval_calls = paired energy + adjoint-gradient simulator "
                                          "executions (not hardware circuit evaluations)"}
                dump_json(path, rec)
                new += 1
                x = np.concatenate([res.angles.gamma, res.angles.beta] +
                                   ([res.angles.alpha] if cd else []))
            if (p, cd) not in best or rec["energy"] < best[(p, cd)][0]:
                best[(p, cd)] = (rec["energy"], x)
        return {"id": row["id"], "new": new, "seconds": time.perf_counter() - started, "gpu": str(gpu)}
    except Exception:
        dump_json(directory / "errors" / f"{row['id']}.json",
                  {"id": row["id"], "gpu": str(gpu), "traceback": traceback.format_exc()})
        raise


# Relative GPU cost of one gate pass per slot, from scripts/20_quick_bench.py on 8x RTX PRO 6000
# with MPS (n=18/20 at 4 slots per GPU, n=22 one per GPU).  Only used for the ETA print.
_N_COST = {18: 0.52, 20: 1.0, 22: 6.7, 24: 27.0}


def _work(row, depths) -> float:
    """Rough cost of one instance's full plan: gates per (QAOA + CD) layer pair x sum(p) x n-cost."""
    n, m = int(row["n"]), len(row["edges"])
    return sum(depths) * (8 * m + 2 * n) * _N_COST.get(n, max(0.2, 2.0 ** (n - 20)))


def _missing(directory, row, plan):
    folder = Path(directory) / "checkpoints" / row["id"]
    have = {p.name for p in folder.glob("p*_r*.json")} if folder.exists() else set()
    return [k for k in plan if f"{_key(k[0], k[1], k[2])}.json" not in have]


def import_previous(directory: Path, previous: Path, manifest: dict) -> int:
    """Copy finished optimizations from an older run of the same design (e.g. 20 -> 50 per cell).

    Accepted only when instance fingerprint, init style, maxiter, ftol, gtol, backend and dtype agree.
    """
    previous = Path(previous)
    if not (previous / "checkpoints").exists():
        raise FileNotFoundError(previous / "checkpoints")
    cfg = manifest["config"]
    plan = instance_plan(cfg)
    settings = cfg["execution"]
    copied = 0
    for row in manifest["splits"]["train"]:
        for p, cd, r, style in plan:
            dst = _ckpt(directory, row["id"], p, cd, r)
            src = _ckpt(previous, row["id"], p, cd, r)
            if dst.exists() or not src.exists():
                continue
            rec = load_json(src)
            same_style = (rec["init_style"] == style or
                          (style == "random" and rec["init_style"].startswith("random")))
            ok = (same_style and rec["instance_fingerprint"] == row["fingerprint"] and
                  rec["maxiter"] == _opt(cfg)["maxiter"] and rec["backend"] == settings["backend"] and
                  rec["dtype"] == settings.get("dtype", "complex128") and
                  float(rec.get("ftol", 1e-10)) == _opt(cfg)["ftol"] and
                  float(rec.get("gtol", 1e-5)) == _opt(cfg)["gtol"])
            if not ok:
                continue
            tsrc = previous / rec["trace_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if tsrc.exists():
                tdst = directory / rec["trace_path"]
                tdst.parent.mkdir(parents=True, exist_ok=True)
                if not tdst.exists():
                    try:
                        os.link(tsrc, tdst)          # hard link: no extra disk space per round
                    except OSError:
                        shutil.copy2(tsrc, tdst)
            rec["imported_from"] = str(previous)
            dump_json(dst, rec)
            copied += 1
    return copied


def train(manifest: dict, directory: Path, reuse_from: str | None = None,
          runtime: dict | None = None) -> list[dict]:
    import fcntl
    from .gpu_pool import resolve_execution, run_tasks, task_weight
    directory = Path(directory)
    for folder in ("checkpoints", "traces", "errors"):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".training.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        owner = lock.read().strip() or "unknown"
        lock.close()
        raise RuntimeError(
            f"{directory} is still being trained by another process (launcher pid {owner}). "
            "The lock is released automatically when that process exits, so a leftover run from a "
            f"dropped SSH session is still alive. Check with:  ps -e | grep cdqaoa ; nvidia-smi   "
            "and stop it (launcher and its GPU workers) with:  pkill cdqaoa   "
            "then rerun the same command - finished work is kept.")
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    monitor, monitor_log = None, None
    try:
        config = dict(manifest["config"], _sha256=manifest["config_sha256"])
        settings = config["execution"]
        ex = resolve_execution(config, runtime)
        for source in [s.strip() for s in str(reuse_from or "").split(",") if s.strip()]:
            if Path(source).resolve() == directory.resolve():
                continue
            print(f"[train] imported {import_previous(directory, Path(source), manifest)} "
                  f"finished optimizations from {source}", flush=True)
        for stale in (directory / "errors").glob("*.json"):
            stale.rename(stale.with_suffix(f".resolved-{time.time_ns()}"))
        plan = instance_plan(config)
        rows = [r for r in manifest["splits"]["train"] if _missing(directory, r, plan)]
        total_opt = len(manifest["splits"]["train"]) * len(plan)
        pending_opt = sum(len(_missing(directory, r, plan)) for r in rows)
        sizes = sorted({r["n"] for r in manifest["splits"]["train"]})
        print(f"[train] {len(manifest['splits']['train'])} instances x {len(plan)} optimizations = "
              f"{total_opt}; pending {pending_opt} in {len(rows)} instances", flush=True)
        print(f"[train] GPUs {','.join(ex['gpus'])}, {ex['slots_per_gpu']} slots/GPU; slots per task: " +
              ", ".join(f"n={n}:{task_weight(n, ex)}" for n in sizes), flush=True)
        if rows:
            dp = sum(config["depths"])
            rows.sort(key=lambda r: (r["n"], len(r["edges"]) * dp), reverse=True)   # heavy first
            _write_environment(directory, manifest, ex, pending_opt)
            if settings["backend"] == "lightning.gpu":
                try:
                    monitor_log = (directory / f"gpu_utilization_{time.time_ns()}.csv").open("w")
                    monitor = subprocess.Popen(
                        ["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,"
                         "utilization.memory,memory.used,memory.total,power.draw",
                         "--format=csv", "-l", "5"], stdout=monitor_log, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    pass
            ctx = {"directory": str(directory), "config": config, "settings": settings,
                   "weights": {str(n): task_weight(n, ex) for n in sizes}, "slots": ex["slots_per_gpu"]}
            done, errors, t_start, opt_done = 0, [], time.perf_counter(), 0
            # Heavy instances run first, so a count-based ETA overestimates badly; weight by cost.
            work = [_work(r, config["depths"]) * len(_missing(directory, r, plan)) / len(plan)
                    for r in rows]
            work_total, work_done = sum(work), 0.0
            for status, i, payload in run_tasks(rows, [r["n"] for r in rows], ex, "train_instance", ctx):
                done += 1
                if status == "error":
                    errors.append(rows[i]["id"])
                    print(f"[train ERROR] {rows[i]['id']}; see errors/{rows[i]['id']}.json", flush=True)
                    continue
                opt_done += payload["new"]
                work_done += work[i]
                el = time.perf_counter() - t_start
                eta = el / max(work_done, 1e-12) * (work_total - work_done)
                print(f"[train] {done}/{len(rows)} instances, {opt_done}/{pending_opt} optimizations, "
                      f"{payload['id']} {payload['seconds']:.0f}s GPU{payload['gpu']}, "
                      f"elapsed {el/3600:.2f} h, work {work_done/work_total:.1%}, "
                      f"cost-weighted ETA {eta/3600:.1f} h", flush=True)
            if errors:
                raise RuntimeError(f"{len(errors)} instances failed; inspect errors/ and rerun to resume")
        return collect_records(manifest, directory)
    finally:
        if monitor is not None:
            monitor.terminate()
        if monitor_log:
            monitor_log.close()
        lock.close()


def _write_environment(directory, manifest, ex, pending):
    import importlib.metadata as md
    import platform
    versions = {}
    for package in ("pennylane", "pennylane-lightning", "pennylane-lightning-gpu", "custatevec-cu12",
                    "numpy", "scipy", "networkx"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            pass
    gpu_info = None
    try:
        gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                   "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=20).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    root = Path(__file__).resolve().parents[1]
    dump_json(directory / f"environment_{time.time_ns()}.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"), "versions": versions, "nvidia_smi": gpu_info,
        "nproc": os.cpu_count(), "execution_resolved": ex,
        "config_sha256": manifest["config_sha256"], "pending_optimizations": pending,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ("cdtransfer", "scripts") for p in (root / folder).glob("*.py")}})


def collect_records(manifest: dict, directory: Path, *, require_complete: bool = False) -> list[dict]:
    directory = Path(directory)
    plan = instance_plan(manifest["config"])
    records, missing = [], 0
    for row in manifest["splits"]["train"]:
        for p, cd, r, _ in plan:
            path = _ckpt(directory, row["id"], p, cd, r)
            if path.exists():
                records.append(load_json(path))
            else:
                missing += 1
    if require_complete and missing:
        raise RuntimeError(f"Training incomplete: {missing} optimizations missing; "
                           "do not freeze/evaluate before training finishes")
    with (directory / "offline_optima.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
    _write_optimizer_audit(directory, records)
    return records


def load_records(directory: Path) -> list[dict]:
    path = Path(directory) / "offline_optima.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_optimizer_audit(directory, records):
    """Non-converged runs and CD-below-QAOA cases (should be ~0 thanks to the warm start)."""
    best = {}
    for r in records:
        k = (r["id"], r["p"], r["cd"])
        if k not in best or r["energy"] < best[k]["energy"]:
            best[k] = r
    worse = []
    for (rid, p, cd), r in best.items():
        if cd and (rid, p, False) in best:
            gap = (r["energy"] - best[(rid, p, False)]["energy"]) / r["features"]["rms"] / r["m"]
            if gap > 1e-6:
                worse.append({"id": rid, "p": p, "cd_minus_qaoa_per_edge_rms": gap})
    dump_json(Path(directory) / "optimizer_audit.json", {
        "records": len(records),
        "not_converged": [{"id": r["id"], "p": r["p"], "cd": r["cd"], "restart": r["restart"],
                           "message": r["optimizer_message"]} for r in records
                          if not r["optimizer_success"]],
        "cd_best_worse_than_qaoa_best": worse,
        "total_eval_calls": int(sum(r["eval_calls"] for r in records)),
        "total_wall_seconds": float(sum(r["wall_seconds"] for r in records))})


# ---------------------------------------------------------- parallel evaluation
def eval_instance(job, ctx, gpu):
    """Handler run inside a gpu_pool worker: fixed-angle energies (+ exact extrema) for one graph."""
    from .lightning import make_simulator
    from .simulator import Angles
    row, specs, exact_max_n = job
    sim = make_simulator(_row_instance(row), _reference_settings(ctx["settings"]))
    out = {}
    for key, angles in specs:
        e = sim.evaluate(Angles(**angles))
        if not np.isfinite(e):
            raise FloatingPointError(f"Non-finite energy for {row['id']} {key}")
        out[key] = {"energy": e}
    if row["n"] <= exact_max_n:
        lo, hi = exact_extrema(_row_instance(row))
        out["__exact__"] = {"ground_energy": lo, "max_energy": hi}
    return {"id": row["id"], "values": out}


def _map(jobs, config, runtime):
    """Yield (id, values) as graphs finish, using the same GPU slot scheduler as training."""
    from .gpu_pool import resolve_execution, run_tasks
    if not jobs:
        return
    ex = resolve_execution(config, runtime)
    jobs = sorted(jobs, key=lambda j: (j[0]["n"], len(j[0]["edges"])), reverse=True)
    for status, i, payload in run_tasks(jobs, [j[0]["n"] for j in jobs], ex, "eval_instance",
                                        {"settings": config["execution"]}):
        if status == "error":
            raise RuntimeError(f"Evaluation failed for {jobs[i][0]['id']}:\n{payload}")
        yield payload["id"], payload["values"]


def _medoid(vectors: np.ndarray) -> int:
    """Index minimizing the summed Euclidean distance; chunked (O(N) memory)."""
    totals = np.zeros(len(vectors))
    for start in range(0, len(vectors), 512):
        block = vectors[start:start + 512]
        d = np.sqrt(((block[:, None, :] - vectors[None, :, :]) ** 2).sum(-1))
        totals[start:start + 512] = d.sum(1)
    return int(np.argmin(totals))


def template(records, p, cd):
    from .optimization import select_best_per_instance
    from .transfer import degree_factor
    rows = select_best_per_instance(records, p=p, cd=cd)
    if not rows:
        raise ValueError(f"No training optima for p={p}, cd={cd}")
    vec = []
    for row in rows:
        f, a = row["features"], row["angles"]
        g, b, al = np.asarray(a["gamma"]), np.asarray(a["beta"]), np.asarray(a["alpha"])
        if np.mean(g) > 0:          # energy-preserving branch alignment (gamma,beta)->(-gamma,-beta)
            g, b = -g, -b
        vec.append(np.concatenate([g * f["rms"] / degree_factor(f["degree"]), b, al * f["rms"]]))
    x = np.asarray(vec)
    i = _medoid(x)
    return {"gamma_unit": x[i, :p].tolist(), "beta": x[i, p:2*p].tolist(),
            "alpha_unit": x[i, 2*p:].tolist(),
            "reference_degree": float(np.median([r["features"]["degree"] for r in rows])),
            "instances": len(rows), "medoid_record_id": rows[i]["id"],
            "median_gamma_unit": np.median(x[:, :p], 0).tolist(),
            "median_beta": np.median(x[:, p:2*p], 0).tolist(),
            "median_alpha_unit": np.median(x[:, 2*p:], 0).tolist(),
            "normalization": "gamma_unit=gamma*rms/degree_factor, alpha_unit=alpha*rms; "
                             "branch aligned so mean(gamma)<0"}


def fit(manifest: dict, records: list[dict], directory: Path, runtime: dict | None = None) -> dict:
    from .transfer import predict, learned_angles, bch_angles
    from .experiment import export_fixed_parameter_table
    directory = Path(directory)
    path = directory / "FROZEN_MODEL.json"
    if path.exists():
        model = load_json(path)
        if model.get("manifest_sha256") != manifest["config_sha256"]:
            raise ValueError("Frozen model belongs to another manifest")
        return model
    collect_records(manifest, directory, require_complete=True)
    depths = sorted(manifest["config"]["depths"])
    settings = manifest["config"]["execution"]
    templates = {str(p): {"qaoa": template(records, p, False), "cd": template(records, p, True)}
                 for p in depths}
    model = {"version": "eqdepth-0.3.0", "comparison": MODE, "templates": templates,
             "depths": depths, "manifest_sha256": manifest["config_sha256"],
             "no_target_instance_objective_calls": True,
             "protocol": "templates from train optima; hyper-parameters from validation; "
                         "frozen before test"}
    val = manifest["splits"]["val"]
    candidates = [(b, c) for b in (-0.5, 0.0, 0.5, 1.0) for c in (0.5, 1.0, 1.5)]
    bch_values = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    jobs = []
    for row in val:
        inst = _row_instance(row)
        specs = []
        for p in depths:
            for sign in (-1, 1):
                specs.append((f"csv:{p}:{sign}", predict(inst, p, "qaoa_csv", {"csv_sign": sign}).to_dict()))
                for i, c in enumerate(bch_values):
                    specs.append((f"bch:{p}:{sign}:{i}", bch_angles(inst, p, c, sign).to_dict()))
                specs.append((f"csvrms:{p}:{sign}", predict(inst, p, "qaoa_csv_rms", {"csv_sign": sign}).to_dict()))
            specs.append((f"qtrained:{p}", learned_angles(inst, p, templates, cd=False).to_dict()))
            for i, (b, c) in enumerate(candidates):
                specs.append((f"cd:{p}:{i}", learned_angles(inst, p, templates, cd=True,
                              alpha_degree_power=b, alpha_multiplier=c).to_dict()))
        jobs.append((row, specs, -1))
    values = {}
    for k, (rid, out) in enumerate(_map(jobs, manifest["config"], runtime), 1):
        values[rid] = out
        if k % 50 == 0 or k == len(jobs):
            print(f"[fit/val] {k}/{len(jobs)} validation instances", flush=True)
    z = {row["id"]: float(np.sum(np.abs(row["weights"]))) for row in val}

    def avg(keys_for):
        return float(np.mean([values[r["id"]][k]["energy"] / z[r["id"]]
                              for r in val for k in keys_for]))
    sign_loss = {str(s): avg([f"csv:{p}:{s}" for p in depths]) for s in (-1, 1)}
    sign = int(min(sign_loss, key=sign_loss.get))
    model.update(csv_sign=sign, csv_sign_selection=sign_loss)
    scores = {str(p): {"qaoa_csv": avg([f"csv:{p}:{sign}"]), "qaoa_csv_rms": avg([f"csvrms:{p}:{sign}"]),
                       "qaoa_trained": avg([f"qtrained:{p}"])} for p in depths}
    model["qaoa_validation_scores_by_depth"] = scores
    model["qaoa_fixed_best_by_depth"] = {p: min(s, key=s.get) for p, s in scores.items()}
    grid = sorted([{"alpha_degree_power": b, "alpha_multiplier": c,
                    "validation_normalized_energy": avg([f"cd:{p}:{i}" for p in depths])}
                   for i, (b, c) in enumerate(candidates)], key=lambda g: g["validation_normalized_energy"])
    model.update(alpha_grid_val=grid, alpha_degree_power=grid[0]["alpha_degree_power"],
                 alpha_multiplier=grid[0]["alpha_multiplier"])
    bch = sorted([{"c": c, "validation_normalized_energy": avg([f"bch:{p}:{sign}:{i}" for p in depths])}
                  for i, c in enumerate(bch_values)], key=lambda g: g["validation_normalized_energy"])
    model.update(bch_val=bch, bch_c=bch[0]["c"])
    dump_json(path, model)
    model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "FROZEN_MODEL.sha256").write_text(f"{model_hash}  FROZEN_MODEL.json\n")
    export_fixed_parameter_table(directory, model)
    print(f"[fit] FROZEN_MODEL.json sha256={model_hash}", flush=True)
    return model


# ------------------------------------------------------------------ test stage
def _test_methods(p, model, v1):
    from .transfer import predict
    methods = {"qaoa_csv": (model, "qaoa_csv"),
               "qaoa_best": (model, model["qaoa_fixed_best_by_depth"][str(p)]),
               "qaoa_trained": (model, "qaoa_trained"),
               "cd_bch": (model, "cd_bch"), "cd_reference": (model, "cd_reference"),
               "cd_proposed": (model, "cd_proposed")}
    if v1 is not None and p in v1["depths"]:
        methods["cd_v1"] = (v1, "cd_proposed")
        methods["qaoa_v1"] = (v1, v1["qaoa_fixed_best_by_depth"][str(p)])
    return methods


def evaluate(manifest: dict, model: dict, directory: Path, runtime: dict | None = None) -> list[dict]:
    from .transfer import predict
    from .resources import gate_cost
    directory = Path(directory)
    if model.get("manifest_sha256") != manifest["config_sha256"]:
        raise ValueError("Frozen model does not match this manifest")
    frozen = directory / "FROZEN_MODEL.sha256"
    if frozen.exists():
        want = frozen.read_text().split()[0]
        have = hashlib.sha256((directory / "FROZEN_MODEL.json").read_bytes()).hexdigest()
        if want != have:
            raise ValueError("FROZEN_MODEL.json changed after freezing")
    v1 = load_json(V1_MODEL) if V1_MODEL.exists() else None
    if v1 is not None:
        v1 = dict(v1, depths=sorted(int(p) for p, t in v1["templates"].items() if "cd" in t))
    depths = sorted(manifest["config"]["depths"])
    settings = manifest["config"]["execution"]
    outdir = directory / "test_results"
    outdir.mkdir(exist_ok=True)
    jobs, meta = [], {}
    for row in manifest["splits"]["test"]:
        if (outdir / f"{row['id']}.json").exists():
            continue
        inst = _row_instance(row)
        specs, info = [], []
        for p in depths:
            for name, (mdl, method) in _test_methods(p, model, v1).items():
                a = predict(inst, p, method, mdl)
                specs.append((f"{p}:{name}", a.to_dict()))
                info.append((p, name, gate_cost(inst, a)["decomposed_cx"]))
        jobs.append((row, specs, _exact_max_n(manifest["config"])))
        meta[row["id"]] = (row, inst, info)
    for k, (rid, out) in enumerate(_map(jobs, manifest["config"], runtime), 1):
        row, inst, info = meta[rid]
        z = float(np.sum(np.abs(inst.weights)))
        ex = out.get("__exact__")
        em = {"exact": ex is not None, **(ex or {"ground_energy": None, "max_energy": None})}
        rows = []
        for p in depths:
            rec = {"id": rid, "p": p, "n": inst.n, "m": inst.m, "family": inst.family,
                   "weight_kind": inst.weight_kind, "seed": inst.seed, "norm_weights": z,
                   "rms": features(inst)["rms"], "degree": features(inst)["degree"],
                   "ground_energy": em["ground_energy"], "max_energy": em["max_energy"],
                   "ratio_kind": _ratio_kind(inst), "graph_features": row.get("graph_features")}
            for pp, name, cx in info:
                if pp != p:
                    continue
                e = out[f"{p}:{name}"]["energy"]
                rec[f"energy_{name}"] = e
                rec[f"norm_energy_{name}"] = e / z
                rec[f"ratio_{name}"] = _ratio(e, em)
                rec[f"cx_{name}"] = cx
            rec["gain_same_depth_norm"] = (rec["energy_qaoa_best"] - rec["energy_cd_proposed"]) / z
            if "energy_cd_v1" in rec:
                rec["gain_vs_v1_norm"] = (rec["energy_cd_v1"] - rec["energy_cd_proposed"]) / z
            rows.append(rec)
        dump_json(outdir / f"{rid}.json", rows)
        if k % 50 == 0 or k == len(jobs):
            print(f"[test] {k}/{len(jobs)} test instances", flush=True)
    allrows = []
    for row in manifest["splits"]["test"]:
        path = outdir / f"{row['id']}.json"
        if not path.exists():
            raise RuntimeError(f"Missing test result {path}")
        allrows.extend(load_json(path))
    dump_json(directory / "heldout_results.json", allrows)
    return allrows


# --------------------------------------------------------------------- report
def _ci(x, n_boot=4000, seed=1441):
    x = np.asarray([v for v in x if v is not None], float)
    if x.size == 0:
        return [None, None]
    if x.size == 1:
        return [float(x[0])] * 2
    rng = np.random.default_rng(seed)
    m = x[rng.integers(x.size, size=(n_boot, x.size))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def _stats(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "ci95": _ci(vals)}


def summarize(test_rows):
    groups = defaultdict(list)
    for r in test_rows:
        p = r["p"]
        for g in (f"p{p}|all", f"p{p}|family={r['family']}", f"p{p}|weights={r['weight_kind']}",
                  f"p{p}|n={r['n']}", f"p{p}|family={r['family']}|weights={r['weight_kind']}"):
            groups[g].append(r)
    out = {}
    methods = sorted({k[6:] for r in test_rows for k in r if k.startswith("ratio_") and k != "ratio_kind"} |
                     {k[12:] for r in test_rows for k in r if k.startswith("norm_energy_")})
    for g, rows in sorted(groups.items()):
        gain = [r["gain_same_depth_norm"] for r in rows]
        s = {"pairs": len(rows), "gain_same_depth_norm": _stats(rows, "gain_same_depth_norm"),
             "cd_win_rate": float(np.mean(np.asarray(gain) > 1e-9)),
             "gain_vs_v1_norm": _stats(rows, "gain_vs_v1_norm")}
        for m in methods:
            s[f"ratio_{m}"] = _stats(rows, f"ratio_{m}")
            s[f"norm_energy_{m}"] = _stats(rows, f"norm_energy_{m}")
        out[g] = s
    return out


def _optimized_summary(records):
    best = {}
    for r in records:
        k = (r["id"], r["p"], r["cd"])
        if k not in best or r["energy"] < best[k]["energy"]:
            best[k] = r
    groups = defaultdict(list)
    for (rid, p, cd), r in best.items():
        for g in (f"p{p}|all", f"p{p}|family={r['family']}", f"p{p}|weights={r['weight_kind']}",
                  f"p{p}|n={r['n']}"):
            groups[(g, cd)].append(r)
    out = {}
    for (g, cd), rows in sorted(groups.items()):
        out.setdefault(g, {})["cd" if cd else "qaoa"] = {
            "ratio": _stats(rows, "ratio"), "normalized_energy": _stats(rows, "normalized_energy")}
    return out


def report(directory: Path, manifest: dict, records: list[dict], model: dict, test_rows: list[dict]):
    from .eqdepth_plots import make_figures
    directory = Path(directory)
    summary = summarize(test_rows)
    optimized = _optimized_summary(records)
    dump_json(directory / "summary.json", {"fixed_formula_test": summary, "optimized_train": optimized})
    figures = make_figures(directory, records, test_rows, manifest["config"]["depths"])
    depths = sorted(manifest["config"]["depths"])
    cnt = {s: len(manifest["splits"][s]) for s in ("train", "val", "test")}
    kinds = sorted({r["ratio_kind"] for r in test_rows})
    lines = ["# Equal-depth CD-QAOA GPU study — automated report", "",
             f"- Instances: train={cnt['train']}, val={cnt['val']}, test={cnt['test']} "
             f"(independent seeds; split fingerprints in manifest.json).",
             f"- Depths: {depths}. Comparison: CD-QAOA p vs QAOA p (**same number of layers**). "
             "CD layers contain 3x the two-qubit rotations of a QAOA layer (cx_* columns); "
             "this is NOT an equal-gate-budget comparison.",
             f"- Training optimizations: {len(records)} (all restarts kept; traces in traces/).",
             f"- Ratio definition: {kinds}. E_min/E_max are exact (exhaustive enumeration) for "
             f"n <= {_exact_max_n(manifest['config'])}; ratios are omitted above that.",
             f"- Frozen model sha256: {(directory / 'FROZEN_MODEL.sha256').read_text().split()[0] if (directory / 'FROZEN_MODEL.sha256').exists() else 'n/a'}; "
             f"alpha degree power {model['alpha_degree_power']}, multiplier {model['alpha_multiplier']}, "
             f"csv sign {model['csv_sign']}.", "",
             "## Held-out fixed-formula results (no test-instance optimization)", "",
             "gain = (E_QAOA,best-fixed − E_CD,proposed)/Σ|J| at the same p; positive favours CD.", "",
             "| p | pairs | mean gain [95% CI] | CD win rate | ratio QAOA fixed | ratio CD proposed | ratio CD V1 | gain vs V1 [95% CI] |",
             "|---|---:|---|---:|---:|---:|---:|---|"]

    def fmt(s, key="mean"):
        return "–" if not s else f"{s[key]:.4f}"

    def fci(s):
        return "–" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
    for p in depths:
        s = summary.get(f"p{p}|all")
        if not s:
            continue
        lines.append(f"| {p} | {s['pairs']} | {fci(s['gain_same_depth_norm'])} | {s['cd_win_rate']:.1%} | "
                     f"{fmt(s.get('ratio_qaoa_best'))} | {fmt(s.get('ratio_cd_proposed'))} | "
                     f"{fmt(s.get('ratio_cd_v1'))} | {fci(s.get('gain_vs_v1_norm'))} |")
    lines += ["", "## Optimized (train, best of restarts) — ceiling for fixed formulas", "",
              "| p | QAOA ratio | CD ratio | QAOA E/Σ|J| | CD E/Σ|J| |", "|---|---:|---:|---:|---:|"]
    for p in depths:
        o = optimized.get(f"p{p}|all", {})
        lines.append(f"| {p} | {fmt((o.get('qaoa') or {}).get('ratio'))} | {fmt((o.get('cd') or {}).get('ratio'))} | "
                     f"{fmt((o.get('qaoa') or {}).get('normalized_energy'))} | "
                     f"{fmt((o.get('cd') or {}).get('normalized_energy'))} |")
    lines += ["", "Per family / weight / n breakdowns: summary.json and summary_table.csv.", "",
              "## Figures", "", *[f"- `{f}`" for f in figures], "",
              "## Caveats", "",
              "- Optima are best-of-restarts local optima (L-BFGS-B); see optimizer_audit.json.",
              "- Do not modify the formula after looking at test results; new formulas need a new frozen model "
              "validated on val and a fresh test evaluation."]
    (directory / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (directory / "summary_table.csv").open("w", encoding="utf-8") as f:
        f.write("group,pairs,gain_mean,gain_ci_lo,gain_ci_hi,cd_win_rate,ratio_qaoa_best,ratio_cd_proposed,ratio_cd_v1\n")
        for g, s in summary.items():
            gs = s["gain_same_depth_norm"]
            f.write(f"\"{g}\",{s['pairs']},{gs['mean']},{gs['ci95'][0]},{gs['ci95'][1]},{s['cd_win_rate']},"
                    f"{(s.get('ratio_qaoa_best') or {}).get('mean','')},"
                    f"{(s.get('ratio_cd_proposed') or {}).get('mean','')},"
                    f"{(s.get('ratio_cd_v1') or {}).get('mean','')}\n")
    return summary


# ------------------------------------------------------------------------ CLI
def run(stage: str, config: dict, out: Path, reuse_from: str | None = None, runtime: dict | None = None):
    manifest = create_manifest(config, out)
    stored = json.loads(json.dumps(manifest["config"]))
    for k in RUNTIME_KEYS:      # scheduling options always come from the config file used now
        stored.setdefault("execution", {}).pop(k, None)
        if k in config.get("execution", {}):
            stored["execution"][k] = config["execution"][k]
    manifest = dict(manifest, config=stored)
    if stage == "generate":
        print(f"[generate] {out / 'manifest.json'}: " +
              ", ".join(f"{s}={len(v)}" for s, v in manifest["splits"].items()))
        return
    if stage in ("train", "pipeline"):
        records = train(manifest, out, reuse_from, runtime)
        if stage == "train":
            return
    else:
        records = load_records(out) if (out / "offline_optima.jsonl").exists() else \
            collect_records(manifest, out)
    if stage in ("fit", "pipeline"):
        model = fit(manifest, records, out, runtime)
        if stage == "fit":
            return
    else:
        model = load_json(out / "FROZEN_MODEL.json")
    if stage in ("evaluate", "pipeline"):
        rows = evaluate(manifest, model, out, runtime)
        if stage == "evaluate":
            return
    else:
        rows = load_json(out / "heldout_results.json")
    report(out, manifest, records, model, rows)
    print(f"[report] {out / 'EXPERIMENT_REPORT.md'}", flush=True)
