"""Template-aggregation study for the frozen equal-depth formula (validation selection -> one test pass).

Question: the frozen model uses the MEDOID of the per-instance train optima as the angle template.
Is the medoid a noisy choice (it is one instance's optimum)?  Candidates per depth p:
    own:medoid   own:median   own:tmean20      (train optima of the same run)
    pool:medoid  pool:median  pool:tmean20     (--pool: train optima of all --runs together)
Everything else of the frozen formula (degree factor, rms scaling, alpha degree power/multiplier,
csv sign) is kept exactly as frozen.

Stage `val`  : evaluate all candidates on each run's VALIDATION split, pick per p the candidate with
               the lowest mean E/sum|J| (CD and QAOA separately; QAOA may also pick qaoa_csv), write
               MODEL_V2_<run>.json + .sha256.  Test data are not touched.
Stage `test` : evaluate MODEL_V2 once on the test split, compare with the original frozen-model test
               results (heldout_results.json, not re-run) and write TEST_SUMMARY_<run>.md.

Server (from the package root, conda env with Lightning-GPU, MPS started as for training):
    python scripts/24_template_aggregation.py val  --runs results/eqdepth_small_N20 results/eqdepth_large_N20 \
           results/eqdepth_n18p6_N20 --pool --out results/template_agg --gpus 0,1,2,3,4,5,6,7 --slots-per-gpu 4
    python scripts/24_template_aggregation.py test --runs results/eqdepth_small_N20 results/eqdepth_large_N20 \
           results/eqdepth_n18p6_N20 --out results/template_agg --gpus 0,1,2,3,4,5,6,7 --slots-per-gpu 4
Quick CPU check (no GPU, n<=18 only, first K val rows):  add  --cpu-workers 2 --limit 4
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdtransfer import eqdepth                                   # noqa: E402
from cdtransfer.instances import load_json, dump_json            # noqa: E402
from cdtransfer.simulator import Angles                          # noqa: E402
from cdtransfer.transfer import degree_factor, learned_angles, predict   # noqa: E402

AGGS = ("medoid", "median", "tmean20")


# ------------------------------------------------------------------ templates from train optima
def best_train_rows(run_dir: Path, depths):
    """Best restart per (id, p, cd), streamed from offline_optima.jsonl (only the fields needed)."""
    best = {}
    with open(run_dir / "offline_optima.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["p"] not in depths:
                continue
            k = (r["id"], r["p"], bool(r["cd"]))
            if k not in best or r["energy"] < best[k]["energy"]:
                best[k] = {"id": r["id"], "p": r["p"], "cd": bool(r["cd"]), "energy": r["energy"],
                           "angles": r["angles"], "degree": r["features"]["degree"],
                           "rms": r["features"]["rms"]}
    return best


def vectors(rows, p):
    """Exactly the normalisation of eqdepth.template()."""
    out = []
    for row in rows:
        a = row["angles"]
        g, b, al = np.asarray(a["gamma"]), np.asarray(a["beta"]), np.asarray(a["alpha"])
        if np.mean(g) > 0:
            g, b = -g, -b
        out.append(np.concatenate([g * row["rms"] / degree_factor(row["degree"]), b, al * row["rms"]]))
    return np.asarray(out)


def trim_mean(x, frac=0.2):
    x = np.sort(x, axis=0)
    k = int(np.floor(frac * len(x)))
    return x[k:len(x) - k].mean(0) if len(x) - 2 * k > 0 else x.mean(0)


def aggregate(rows, p, how):
    x = vectors(rows, p)
    if how == "medoid":
        v, extra = x[eqdepth._medoid(x)], {}
    elif how == "median":
        v, extra = np.median(x, 0), {}
    elif how == "tmean20":
        v, extra = trim_mean(x, 0.2), {}
    else:
        raise ValueError(how)
    return {"gamma_unit": v[:p].tolist(), "beta": v[p:2 * p].tolist(), "alpha_unit": v[2 * p:].tolist(),
            "reference_degree": float(np.median([r["degree"] for r in rows])),
            "instances": len(rows), "aggregation": how, **extra}


def candidate_templates(best, depths, tag):
    out = {}
    for agg in AGGS:
        tm = {}
        for p in depths:
            q = [r for (i, pp, cd), r in best.items() if pp == p and not cd]
            c = [r for (i, pp, cd), r in best.items() if pp == p and cd]
            if q and c:
                tm[str(p)] = {"qaoa": aggregate(q, p, agg), "cd": aggregate(c, p, agg)}
        out[f"{tag}:{agg}"] = tm
    return out


# ------------------------------------------------------------------ evaluation backends
def _cpu_eval(job):
    from cdtransfer.simulator import StatevectorSimulator
    row, specs, _ = job
    sim = StatevectorSimulator(eqdepth._row_instance(row))
    return row["id"], {k: {"energy": sim.evaluate(Angles(**a))} for k, a in specs}


def run_jobs(jobs, config, args):
    if args.cpu_workers:
        from multiprocessing import Pool
        with Pool(args.cpu_workers) as pool:
            yield from pool.imap_unordered(_cpu_eval, jobs)
    else:
        runtime = {"gpus": args.gpus, "slots_per_gpu": args.slots_per_gpu, "task_weights": args.task_weights}
        yield from eqdepth._map(jobs, config, runtime)


def boot_ci(x, n_boot=4000, seed=1441):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ stage: val
def stage_val(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in args.runs]
    models = {r: load_json(r / "FROZEN_MODEL.json") for r in runs}
    t0 = time.time()
    best = {}
    for r in runs:
        best[r] = best_train_rows(r, sorted(models[r]["depths"]))
        print(f"[val] {r.name}: {len(best[r])} best train optima loaded ({time.time() - t0:.0f}s)", flush=True)
    pooled = {}
    if args.pool:
        for r in runs:
            pooled.update(best[r])          # same id = same instance (n18p6 imported large's n18 optima)
        all_depths = sorted({p for r in runs for p in models[r]["depths"]})
        pool_c = candidate_templates(pooled, all_depths, "pool")
    report = ["# Template aggregation - validation selection", ""]
    for r in runs:
        model, name = models[r], r.name
        depths = sorted(model["depths"])
        cands = candidate_templates(best[r], depths, "own")
        # sanity: own:medoid must reproduce the frozen template
        for p in depths:
            fz, me = model["templates"][str(p)]["cd"], cands["own:medoid"][str(p)]["cd"]
            if not np.allclose(fz["gamma_unit"], me["gamma_unit"], atol=1e-9):
                print(f"[val] WARNING {name} p={p}: own:medoid differs from FROZEN_MODEL template", flush=True)
        if args.pool:
            cands.update(pool_c)
        manifest = load_json(r / "manifest.json")
        val = manifest["splits"]["val"][: args.limit or None]
        jobs = []
        for row in val:
            inst = eqdepth._row_instance(row)
            specs = []
            for p in depths:
                specs.append((f"{p}|qaoa_csv", predict(inst, p, "qaoa_csv", model).to_dict()))
                for cn, tm in cands.items():
                    if str(p) not in tm:
                        continue
                    specs.append((f"{p}|qaoa|{cn}", learned_angles(inst, p, tm, cd=False).to_dict()))
                    specs.append((f"{p}|cd|{cn}", learned_angles(
                        inst, p, tm, cd=True, alpha_degree_power=float(model["alpha_degree_power"]),
                        alpha_multiplier=float(model["alpha_multiplier"])).to_dict()))
            jobs.append((row, specs, -1))
        meta = {row["id"]: row for row in val}
        res = {}
        with open(out / f"val_{name}.jsonl", "w", encoding="utf-8") as f:
            for k, (rid, vals) in enumerate(run_jobs(jobs, manifest["config"], args), 1):
                row = meta[rid]
                rec = {"id": rid, "n": row["n"], "family": row["family"], "weight_kind": row["weight_kind"],
                       "z": float(np.sum(np.abs(row["weights"]))),
                       **{key: v["energy"] for key, v in vals.items() if not key.startswith("__")}}
                res[rid] = rec
                f.write(json.dumps(rec) + "\n")
                if k % 100 == 0 or k == len(jobs):
                    print(f"[val] {name}: {k}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        recs = list(res.values())
        m2 = copy.deepcopy(model)
        m2["version"] = "eqdepth-0.3.0+template-agg"
        m2["v1_frozen_model_sha256"] = sha(r / "FROZEN_MODEL.json")
        m2["protocol"] = ("templates: per-p aggregation selected on validation among "
                          f"{sorted(cands)}; all other hyper-parameters as in v1; frozen before test")
        m2["template_selection"] = {}
        report += [f"## {name} ({len(recs)} validation instances)", "",
                   "| p | candidate | CD E/Σ|J| | Δ vs own:medoid [95% CI] (neg = better) | QAOA E/Σ|J| |",
                   "|---|---|---:|---|---:|"]
        for p in depths:
            names = [cn for cn in cands if str(p) in cands[cn]]
            ref = np.array([x[f"{p}|cd|own:medoid"] / x["z"] for x in recs])
            cd_score, q_score = {}, {"qaoa_csv": float(np.mean([x[f"{p}|qaoa_csv"] / x["z"] for x in recs]))}
            for cn in names:
                e = np.array([x[f"{p}|cd|{cn}"] / x["z"] for x in recs])
                cd_score[cn] = float(e.mean())
                q_score[cn] = float(np.mean([x[f"{p}|qaoa|{cn}"] / x["z"] for x in recs]))
                mu, lo, hi = boot_ci(e - ref)
                report.append(f"| {p} | {cn} | {e.mean():.5f} | {mu:+.5f} [{lo:+.5f}, {hi:+.5f}] | {q_score[cn]:.5f} |")
            cd_pick = min(cd_score, key=cd_score.get)
            q_pick = min(q_score, key=q_score.get)
            m2["templates"][str(p)]["cd"] = dict(cands[cd_pick][str(p)]["cd"], source=cd_pick)
            if q_pick == "qaoa_csv":
                m2["qaoa_fixed_best_by_depth"][str(p)] = "qaoa_csv"
            else:
                m2["templates"][str(p)]["qaoa"] = dict(cands[q_pick][str(p)]["qaoa"], source=q_pick)
                m2["qaoa_fixed_best_by_depth"][str(p)] = "qaoa_trained"
            m2["template_selection"][str(p)] = {"cd": cd_pick, "qaoa": q_pick,
                                                "cd_val_scores": cd_score, "qaoa_val_scores": q_score}
            report.append(f"| {p} | **selected** | CD: `{cd_pick}` | QAOA: `{q_pick}` | |")
        report.append("")
        path = out / f"MODEL_V2_{name}.json"
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists (frozen); use --force only before any test evaluation")
        dump_json(path, m2)
        (out / f"MODEL_V2_{name}.sha256").write_text(f"{sha(path)}  {path.name}\n")
        print(f"[val] {name}: wrote {path.name}", flush=True)
    (out / "VAL_SELECTION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[val] report: {out / 'VAL_SELECTION.md'} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ stage: test
def stage_test(args):
    out = Path(args.out)
    t0 = time.time()
    lines = ["# Template aggregation - single test evaluation of MODEL_V2", ""]
    for r in [Path(x) for x in args.runs]:
        name = r.name
        path = out / f"MODEL_V2_{name}.json"
        want = (out / f"MODEL_V2_{name}.sha256").read_text().split()[0]
        if sha(path) != want:
            raise SystemExit(f"{path} changed after validation selection")
        m2 = load_json(path)
        depths = sorted(m2["depths"])
        manifest = load_json(r / "manifest.json")
        v1 = {(x["id"], x["p"]): x for x in load_json(r / "heldout_results.json")}
        test = manifest["splits"]["test"][: args.limit or None]
        jobs = []
        for row in test:
            inst = eqdepth._row_instance(row)
            specs = []
            for p in depths:
                specs.append((f"{p}|cd", predict(inst, p, "cd_proposed", m2).to_dict()))
                specs.append((f"{p}|qaoa", predict(inst, p, m2["qaoa_fixed_best_by_depth"][str(p)], m2).to_dict()))
            jobs.append((row, specs, -1))
        rows = []
        with open(out / f"test_{name}.jsonl", "w", encoding="utf-8") as f:
            for k, (rid, vals) in enumerate(run_jobs(jobs, manifest["config"], args), 1):
                for p in depths:
                    o = v1[(rid, p)]
                    rec = {"id": rid, "p": p, "n": o["n"], "family": o["family"], "weight_kind": o["weight_kind"],
                           "z": o["norm_weights"], "ground_energy": o["ground_energy"], "max_energy": o["max_energy"],
                           "e_cd_v2": vals[f"{p}|cd"]["energy"], "e_qaoa_v2": vals[f"{p}|qaoa"]["energy"],
                           "e_cd_v1": o["energy_cd_proposed"], "e_qaoa_v1": o["energy_qaoa_best"]}
                    rows.append(rec)
                    f.write(json.dumps(rec) + "\n")
                if k % 200 == 0 or k == len(jobs):
                    print(f"[test] {name}: {k}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

        def ratio(e, x):
            if x["ground_energy"] is None:
                return np.nan
            return (x["max_energy"] - e) / (x["max_energy"] - x["ground_energy"])
        lines += [f"## {name} ({len(test)} test instances)", "",
                  "| p | gain v1 (CD−QAOA, same p) | gain v2 [95% CI] | CD v2 − CD v1 [95% CI] | QAOA v2 − QAOA v1 | "
                  "ratio QAOA v2 | ratio CD v1 → v2 | CD win v2 |", "|---|---:|---|---|---:|---:|---|---:|"]
        by = {}
        for x in rows:
            by.setdefault(x["p"], []).append(x)
        for p in depths:
            xs = by[p]
            z = np.array([x["z"] for x in xs])
            g1 = (np.array([x["e_qaoa_v1"] for x in xs]) - np.array([x["e_cd_v1"] for x in xs])) / z
            g2 = (np.array([x["e_qaoa_v2"] for x in xs]) - np.array([x["e_cd_v2"] for x in xs])) / z
            dc = (np.array([x["e_cd_v1"] for x in xs]) - np.array([x["e_cd_v2"] for x in xs])) / z
            dq = (np.array([x["e_qaoa_v1"] for x in xs]) - np.array([x["e_qaoa_v2"] for x in xs])) / z
            m2_, l2, h2 = boot_ci(g2)
            md, ld, hd = boot_ci(dc)
            rq = np.nanmean([ratio(x["e_qaoa_v2"], x) for x in xs])
            r1 = np.nanmean([ratio(x["e_cd_v1"], x) for x in xs])
            r2 = np.nanmean([ratio(x["e_cd_v2"], x) for x in xs])
            lines.append(f"| {p} | {g1.mean():+.4f} | {m2_:+.4f} [{l2:+.4f}, {h2:+.4f}] | {md:+.4f} [{ld:+.4f}, {hd:+.4f}]"
                         f" | {dq.mean():+.4f} | {rq:.4f} | {r1:.4f} → {r2:.4f} | {(g2 > 0).mean():.1%} |")
        lines += ["", "Per family (CD v2 − CD v1, positive = v2 better), mean over test instances:", "",
                  "| family | " + " | ".join(f"p={p}" for p in depths) + " |", "|---|" + "---:|" * len(depths)]
        fams = sorted({x["family"] for x in rows})
        for fam in fams:
            cells = []
            for p in depths:
                xs = [x for x in by[p] if x["family"] == fam]
                cells.append(f"{np.mean([(x['e_cd_v1'] - x['e_cd_v2']) / x['z'] for x in xs]):+.4f}")
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        lines.append("")
        print(f"[test] {name} done", flush=True)
    (out / "TEST_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[test] report: {out / 'TEST_SUMMARY.md'} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["val", "test"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/template_agg")
    ap.add_argument("--pool", action="store_true", help="also try templates pooled over all --runs")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--slots-per-gpu", type=int, default=None)
    ap.add_argument("--task-weights", default=None)
    ap.add_argument("--cpu-workers", type=int, default=0, help="NumPy CPU backend for quick checks (n<=18)")
    ap.add_argument("--limit", type=int, default=0, help="only the first K val/test instances (checks)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stage_val(args) if args.stage == "val" else stage_test(args)


if __name__ == "__main__":
    main()
