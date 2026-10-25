"""Parameter concentration and scaling-law diagnostics from TRAIN optima only.

Usage:  python scripts/21_eqdepth_concentration.py --run results/eqdepth_full50
        # pool several runs (e.g. n=10-16 and n=18-22) for the n-scaling analysis:
        python scripts/21_eqdepth_concentration.py --run results/eqdepth_small_n10_16_p6 \
            results/eqdepth_full50_n22_p4 --out results/analysis_n10_22

Reads offline_optima.jsonl (best restart per instance/depth/method), normalizes
with the CURRENT V1 law (gamma_unit = gamma*rms/degree_factor(D), alpha_unit = alpha*rms,
beta unchanged), aligns the (gamma,beta)->(-gamma,-beta) branch, then writes
  analysis/concentration_table.csv   median / IQR per layer by family, weights, n, family x n
  analysis/dispersion_vs_n.csv       IQR vs n inside each family x weight cell (concentration test)
  analysis/feature_correlations.csv  Spearman rho of normalized angles vs static graph features:
                                     if the V1 scaling law were complete, rho ~ 0 for every feature
  analysis/scaling_fits.json         log|gamma*rms| and log|alpha*rms| vs log(degree): fitted exponents
  analysis/fig_schedule_*.png        median schedules (IQR band) per family and per n
Validation / test data are never read.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdtransfer.instances import dump_json
from cdtransfer.transfer import degree_factor

FEATURES = ["n", "degree_mean", "degree_std", "degree_cv", "density", "clustering",
            "adjacency_spectral_radius", "algebraic_connectivity", "weight_cv", "weight_skew",
            "weight_max_over_rms", "weight_mean_over_rms"]


def load_best(runs):
    best = {}
    for run in runs:
        with (Path(run) / "offline_optima.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                k = (r["instance_fingerprint"], r["p"], r["cd"])   # same instance in two runs counted once
                if k not in best or r["energy"] < best[k]["energy"]:
                    best[k] = r
    rows = []
    for r in best.values():
        f, a = r["features"], r["angles"]
        g, b, al = (np.asarray(a[k], float) for k in ("gamma", "beta", "alpha"))
        if np.mean(g) > 0:
            g, b = -g, -b
        rows.append({**r, "gamma_unit": g * f["rms"] / degree_factor(f["degree"]), "beta_n": b,
                     "alpha_unit": al * f["rms"], "gamma_rms": g * f["rms"], "gf": r.get("graph_features") or {}})
    return rows


def q(x):
    x = np.asarray(x, float)
    return np.median(x), np.quantile(x, .25), np.quantile(x, .75)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+", help="one or more result directories")
    ap.add_argument("--out", help="output directory (default: <first run>/analysis)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.run[0]) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = load_best(args.run)
    params = [("gamma_unit", False), ("beta_n", False), ("alpha_unit", True)]
    depths = sorted({r["p"] for r in rows})

    # concentration table
    with (out / "concentration_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "p", "param", "layer", "group_type", "group", "count", "median", "q25", "q75", "iqr"])
        for cd in (False, True):
            for p in depths:
                sub = [r for r in rows if r["cd"] == cd and r["p"] == p]
                for par, cd_only in params:
                    if cd_only and not cd:
                        continue
                    for gt, key in (("all", None), ("family", "family"), ("weights", "weight_kind"),
                                    ("n", "n"), ("family_x_n", ("family", "n"))):
                        groups = defaultdict(list)
                        for r in sub:
                            g = "all" if key is None else ("|".join(str(r[k]) for k in key)
                                                           if isinstance(key, tuple) else r[key])
                            groups[g].append(r)
                        for g, rs in sorted(groups.items(), key=lambda t: str(t[0])):
                            for layer in range(p):
                                med, lo, hi = q([r[par][layer] for r in rs])
                                w.writerow(["cd" if cd else "qaoa", p, par, layer + 1, gt, g, len(rs),
                                            med, lo, hi, hi - lo])

    # dispersion vs n inside each family x weight cell
    with (out / "dispersion_vs_n.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "p", "param", "family", "weights", "n", "count", "mean_layer_iqr"])
        cells = defaultdict(list)
        for r in rows:
            cells[(r["cd"], r["p"], r["family"], r["weight_kind"], r["n"])].append(r)
        for (cd, p, fam, wk, n), rs in sorted(cells.items()):
            for par, cd_only in params:
                if cd_only and not cd:
                    continue
                a = np.asarray([r[par] for r in rs])
                iqr = float(np.mean(np.quantile(a, .75, axis=0) - np.quantile(a, .25, axis=0)))
                w.writerow(["cd" if cd else "qaoa", p, par, fam, wk, n, len(rs), iqr])

    # residual dependence on static features (scaling-law completeness)
    with (out / "feature_correlations.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "p", "param", "layer", "feature", "spearman_rho", "count"])
        for cd in (False, True):
            for p in depths:
                sub = [r for r in rows if r["cd"] == cd and r["p"] == p and r["gf"]]
                for par, cd_only in params:
                    if cd_only and not cd:
                        continue
                    for layer in range(p):
                        y = [r[par][layer] for r in sub]
                        for feat in FEATURES:
                            x = [r["gf"].get(feat, r.get(feat)) for r in sub]
                            if any(v is None for v in x):
                                continue
                            w.writerow(["cd" if cd else "qaoa", p, par, layer + 1, feat, spearman(x, y), len(sub)])

    # log-log exponents vs degree (current law: gamma ~ degree_factor(D)/rms, alpha ~ 1/rms)
    fits = {}
    for cd in (False, True):
        for p in depths:
            sub = [r for r in rows if r["cd"] == cd and r["p"] == p and r["features"]["degree"] > 1.0]
            if len(sub) < 5:
                continue
            ld = np.log([r["features"]["degree"] for r in sub])
            for name, key in (("gamma_rms", "gamma_rms"), ("alpha_rms", "alpha_unit")):
                if name == "alpha_rms" and not cd:
                    continue
                for layer in range(p):
                    y = np.array([abs(r[key][layer]) for r in sub])
                    ok = y > 1e-6
                    if ok.sum() < 5:
                        continue
                    slope, icpt = np.polyfit(ld[ok], np.log(y[ok]), 1)
                    fits[f"{'cd' if cd else 'qaoa'}|p{p}|{name}|layer{layer+1}"] = {
                        "exponent_vs_degree": float(slope), "prefactor": float(np.exp(icpt)), "count": int(ok.sum())}
    fits["_reference"] = {"degree_factor_local_exponent_at_D=3": float(
        np.gradient(np.log([degree_factor(d) for d in (2.9, 3.0, 3.1)]), np.log([2.9, 3.0, 3.1]))[1]),
        "note": "current V1 law predicts gamma*rms exponent = d log degree_factor / d log D (about -0.5 at large D)"}
    dump_json(out / "scaling_fits.json", fits)

    # schedule figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, label in (("family", "family"), ("n", "n")):
        for cd in (False, True):
            pars = [p for p in params if cd or not p[1]]
            fig, axes = plt.subplots(len(pars), len(depths), figsize=(3.2 * len(depths), 2.6 * len(pars)),
                                     squeeze=False)
            for j, p in enumerate(depths):
                sub = [r for r in rows if r["cd"] == cd and r["p"] == p]
                cats = sorted({r[key] for r in sub})
                for i, (par, _) in enumerate(pars):
                    ax = axes[i, j]
                    for c_i, c in enumerate(cats):
                        a = np.asarray([r[par] for r in sub if r[key] == c])
                        layers = np.arange(1, p + 1)
                        ax.plot(layers, np.median(a, 0), "-o", ms=3, color=f"C{c_i % 10}", label=str(c))
                        ax.fill_between(layers, np.quantile(a, .25, 0), np.quantile(a, .75, 0),
                                        color=f"C{c_i % 10}", alpha=.15)
                    ax.set_title(f"{'CD' if cd else 'QAOA'} p={p}: {par}", fontsize=8)
                    ax.tick_params(labelsize=7)
            axes[0, -1].legend(fontsize=6, title=label, title_fontsize=6)
            fig.tight_layout()
            fig.savefig(out / f"fig_schedule_{'cd' if cd else 'qaoa'}_by_{label}.png", dpi=140)
            plt.close(fig)
    print(f"[analysis] {len(rows)} best-of-restart optima -> {out}")


if __name__ == "__main__":
    main()
