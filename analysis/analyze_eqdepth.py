"""One-shot analysis of the equal-depth GPU study (all numbers/figures used in the manuscript).

    cd Clean
    python analysis/analyze_eqdepth.py            # -> analysis/outputs/{tables,figures}, ANALYSIS_REPORT.md

Inputs (all under results/gpu_eqdepth/, compact files are enough):
  eqdepth_{small,large,n18p6}_N20/heldout_results.json(.gz)   per-instance fixed-formula test energies
  eqdepth_{small,large,n18p6}_N20/summary.json                 optimized-train summaries (by family/weight/n)
  eqdepth_{...}/optimizer_audit.json, FROZEN_MODEL.json
  eqdepth_n18p6_N20/best_optima.csv.gz, restart_summary.csv.gz  (from analysis/extract_best_optima.py)
  template_agg/test_eqdepth_*.jsonl                            template-aggregation robustness run
If best_optima.csv.gz also exists for small/large (run extract_best_optima.py on the raw
offline_optima.jsonl), the instance-level optimized analyses are extended to those runs automatically.

Dependencies: numpy, pandas, matplotlib, scikit-learn.
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "gpu_eqdepth"
OUT = ROOT / "analysis" / "outputs"
TAB, FIG = OUT / "tables", OUT / "figures"
RUNS = {"small": "eqdepth_small_N20", "large": "eqdepth_large_N20", "n18p6": "eqdepth_n18p6_N20"}
RUN_LABEL = {"small": "n=10–16", "large": "n=18–22", "n18p6": "n=18 (p≤6)"}
FAMILIES = ["tree", "ring", "regular3", "regular4", "regular5", "regular6", "er3", "er05", "community", "bipartite"]
WEIGHTS = ["unit", "uniform", "bimodal", "lognormal", "heavy_tail"]
# validated reference palette (slots 1-3; see dataviz palette) + neutral ink
C_CD, C_Q, C_3, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
RNG = np.random.default_rng(1441)
REPORT = []


# ------------------------------------------------------------------ helpers
def load_json(path: Path):
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    gz = Path(str(path) + ".gz")
    return json.load(gzip.open(gz, "rt", encoding="utf-8"))


def boot(x, n=4000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def heldout(run):
    rows = []
    for r in load_json(RES / RUNS[run] / "heldout_results.json"):
        g = r.pop("graph_features", None) or {}
        r.update({"f_" + k: v for k, v in g.items()})
        rows.append(r)
    d = pd.DataFrame(rows)
    d["run"] = run
    return d


def table(df, name, title, floatfmt=4):
    df.to_csv(TAB / f"{name}.csv", index=False)
    REPORT.append(f"### {title}\n\n`tables/{name}.csv`\n")
    REPORT.append(df.to_markdown(index=False, floatfmt=f".{floatfmt}f") + "\n")


def style(ax, xlabel=None, ylabel=None, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    REPORT.append(f"![{name}](figures/{name}.png)\n")


# ------------------------------------------------------------------ analyses
def dataset_table(H):
    rows = []
    for run, d in H.items():
        model = load_json(RES / RUNS[run] / "FROZEN_MODEL.json")
        man_cfg = json.load(open(ROOT / "code/CDQAOA_EqualDepth_LightningGPU_8x/configs/rounds" / f"{RUNS[run]}.json"))
        ds = man_cfg["datasets"]
        audit = load_json(RES / RUNS[run] / "optimizer_audit.json")
        rows.append({"run": RUNS[run], "n": ",".join(map(str, ds["train"]["n"])),
                     "depths": ",".join(map(str, model["depths"])),
                     "families": len(ds["train"]["families"]), "weights": len(ds["train"]["weights"]),
                     "train": len(ds["train"]["n"]) * 50 * ds["train"]["samples_per_cell"],
                     "val": len(ds["val"]["n"]) * 50 * ds["val"]["samples_per_cell"],
                     "test": d.id.nunique(), "optimizations": audit["records"],
                     "not_converged_%": 100 * len(audit["not_converged"]) / audit["records"],
                     "gpu_slot_hours": audit["total_wall_seconds"] / 3600,
                     "cd_opt_worse_than_qaoa_opt": len(audit["cd_best_worse_than_qaoa_best"])})
    table(pd.DataFrame(rows), "T1_dataset", "T1. Data sets and optimisation effort", 2)


def same_depth(H):
    rows = []
    for run, d in H.items():
        opt = load_json(RES / RUNS[run] / "summary.json")["optimized_train"]
        for p, s in d.groupby("p"):
            m, lo, hi = boot(s.gain_same_depth_norm)
            o = opt[f"p{p}|all"]
            rows.append({"run": run, "p": p, "pairs": len(s), "gain_fixed": m, "ci_lo": lo, "ci_hi": hi,
                         "cd_win": (s.gain_same_depth_norm > 0).mean(),
                         "ratio_qaoa_fixed": s.ratio_qaoa_best.mean(), "ratio_cd_fixed": s.ratio_cd_proposed.mean(),
                         "ratio_qaoa_opt": o["qaoa"]["ratio"]["mean"], "ratio_cd_opt": o["cd"]["ratio"]["mean"],
                         "gain_opt_norm": o["qaoa"]["normalized_energy"]["mean"] - o["cd"]["normalized_energy"]["mean"]})
    t = pd.DataFrame(rows)
    t["gap_closed_fixed"] = (t.ratio_cd_fixed - t.ratio_qaoa_fixed) / (1 - t.ratio_qaoa_fixed)
    t["gap_closed_opt"] = (t.ratio_cd_opt - t.ratio_qaoa_opt) / (1 - t.ratio_qaoa_opt)
    t["captured_by_formula"] = (t.ratio_cd_fixed - t.ratio_qaoa_fixed) / (t.ratio_cd_opt - t.ratio_qaoa_opt)
    table(t, "T2_same_depth", "T2. Same depth: fixed formula (test) and optimized ceiling (train)")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for run, ls in (("small", "-"), ("large", "--")):
        s = t[t.run == run]
        ax[0].plot(s.p, s.ratio_cd_opt, ls, color=C_CD, lw=1.2, alpha=0.45)
        ax[0].plot(s.p, s.ratio_qaoa_opt, ls, color=C_Q, lw=1.2, alpha=0.45)
        ax[0].plot(s.p, s.ratio_cd_fixed, ls, marker="o", ms=4, color=C_CD, lw=2,
                   label=f"CD fixed ({RUN_LABEL[run]})")
        ax[0].plot(s.p, s.ratio_qaoa_fixed, ls, marker="s", ms=4, color=C_Q, lw=2,
                   label=f"QAOA fixed ({RUN_LABEL[run]})")
        ax[1].plot(s.p, s.gap_closed_opt, ls, color=MUTED, lw=1.5, marker="^", ms=4, label=f"optimized ({RUN_LABEL[run]})")
        ax[1].plot(s.p, s.gap_closed_fixed, ls, color=C_CD, lw=2, marker="o", ms=4, label=f"fixed formula ({RUN_LABEL[run]})")
    style(ax[0], "layers p", "mean MaxCut approximation ratio", "(a) same depth; faint = optimized ceiling")
    style(ax[1], "layers p", "fraction of QAOA gap to 1 closed by CD", "(b) CD advantage, optimized vs fixed")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_ylim(0, 0.42)
    fig.tight_layout()
    save(fig, "F2_same_depth")
    return t


def breakdown(H):
    rows = []
    for run, d in H.items():
        for dim in ("family", "weight_kind", "n"):
            for (p, v), s in d.groupby(["p", dim]):
                m, lo, hi = boot(s.gain_same_depth_norm, 2000)
                rows.append({"run": run, "dim": dim, "value": v, "p": p, "pairs": len(s), "gain": m,
                             "ci_lo": lo, "ci_hi": hi, "cd_win": (s.gain_same_depth_norm > 0).mean(),
                             "ratio_qaoa_fixed": s.ratio_qaoa_best.mean(), "ratio_cd_fixed": s.ratio_cd_proposed.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "T3_breakdown_long.csv", index=False)
    for dim, order in (("family", FAMILIES), ("weight_kind", WEIGHTS), ("n", None)):
        s = t[t.dim == dim].copy()
        s["value"] = s["value"].astype(str)
        w = s.pivot_table(index=["run", "value"], columns="p", values="gain").reset_index()
        if order:
            w["o"] = w.value.map({k: i for i, k in enumerate(order)})
            w = w.sort_values(["run", "o"]).drop(columns="o")
        w.columns = [str(c) if not isinstance(c, int) else f"p{c}" for c in w.columns]
        table(w, f"T3_gain_by_{dim}", f"T3. Same-depth fixed gain by {dim} (mean E/Σ|J|)")

    # heatmap figure family x p for small and large
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    lim = 0.16
    for ax, run in zip(axes, ("small", "large")):
        s = t[(t.run == run) & (t.dim == "family")].pivot(index="value", columns="p", values="gain").reindex(FAMILIES)
        im = ax.imshow(s.values, cmap=div_cmap(), vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(s.shape[1]), [f"p={c}" for c in s.columns])
        ax.set_yticks(range(len(FAMILIES)), FAMILIES)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                v = s.values[i, j]
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.09 else INK)
        style(ax, title=f"({'ab'[run == 'large']}) {RUN_LABEL[run]}")
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("gain (E_QAOA − E_CD)/Σ|J|; blue = CD better", fontsize=8)
    save(fig, "F4_family_heatmap")
    return t


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("d", [C_Q, "#f4f3f0", C_CD])


def equal_budget(H, best):
    rows = []
    for run, d in H.items():
        q = d.pivot(index="id", columns="p", values="energy_qaoa_best")
        c = d.pivot(index="id", columns="p", values="energy_cd_proposed")
        z = d.groupby("id").norm_weights.first()
        for pc in sorted(c.columns):
            for k in (2, 3):
                if k * pc in q.columns:
                    g = (q[k * pc] - c[pc]) / z
                    m, lo, hi = boot(g)
                    rows.append({"run": run, "source": "fixed (test)", "cd_p": pc, "qaoa_p": k * pc,
                                 "matching": "compiled level-3 CX" if k == 2 else "unoptimized CX",
                                 "gain": m, "ci_lo": lo, "ci_hi": hi, "cd_win": (g > 0).mean(), "pairs": len(g)})
    for run, b in best.items():
        Q = b[~b.cd].pivot(index="id", columns="p", values="energy")
        C = b[b.cd].pivot(index="id", columns="p", values="energy")
        Z = b.groupby("id").sum_abs_weights.first()
        for pc in sorted(C.columns):
            for k in (1, 2, 3):
                if k * pc in Q.columns:
                    g = (Q[k * pc] - C[pc]) / Z
                    m, lo, hi = boot(g)
                    rows.append({"run": run, "source": "optimized (train)", "cd_p": pc, "qaoa_p": k * pc,
                                 "matching": {1: "same depth", 2: "compiled level-3 CX", 3: "unoptimized CX"}[k],
                                 "gain": m, "ci_lo": lo, "ci_hi": hi, "cd_win": (g > 1e-9).mean(), "pairs": len(g)})
    t = pd.DataFrame(rows)
    table(t, "T4_equal_budget", "T4. Paired comparisons at matched two-qubit budget (gain>0 favours CD)")

    # CD p1 vs QAOA p2 by n x family
    rows = []
    for run in ("small", "large"):
        d = H[run]
        q = d.pivot(index="id", columns="p", values="energy_qaoa_best")
        c = d.pivot(index="id", columns="p", values="energy_cd_proposed")
        meta = d.groupby("id")[["n", "family", "norm_weights"]].first()
        g = ((q[2] - c[1]) / meta.norm_weights).rename("g")
        x = pd.concat([g, meta], axis=1)
        for (n, fam), s in x.groupby(["n", "family"]):
            rows.append({"n": n, "family": fam, "gain_cd1_vs_qaoa2": s.g.mean(), "cd_win": (s.g > 0).mean()})
        for n, s in x.groupby("n"):
            rows.append({"n": n, "family": "ALL", "gain_cd1_vs_qaoa2": s.g.mean(), "cd_win": (s.g > 0).mean()})
    u = pd.DataFrame(rows)
    w = u.pivot(index="family", columns="n", values="gain_cd1_vs_qaoa2").reindex(FAMILIES + ["ALL"]).reset_index()
    w.columns = [str(cc) for cc in w.columns]
    table(w, "T5_cd1_vs_qaoa2_by_n", "T5. CD p=1 vs QAOA p=2 (equal compiled CX) by n and family")
    opt_rows = []
    for run, b in best.items():
        if run == "n18p6":
            continue
        Q = b[~b.cd].pivot(index="id", columns="p", values="energy")
        C = b[b.cd].pivot(index="id", columns="p", values="energy")
        meta = b.groupby("id")[["n", "family", "sum_abs_weights"]].first()
        g = ((Q[2] - C[1]) / meta.sum_abs_weights).rename("g")
        x = pd.concat([g, meta], axis=1)
        for (n, fam), s in x.groupby(["n", "family"]):
            opt_rows.append({"n": n, "family": fam, "gain": s.g.mean(), "cd_win": (s.g > 1e-9).mean()})
        for n, s in x.groupby("n"):
            opt_rows.append({"n": n, "family": "ALL", "gain": s.g.mean(), "cd_win": (s.g > 1e-9).mean()})
    uo = pd.DataFrame(opt_rows)
    if len(uo):
        w = uo.pivot(index="family", columns="n", values="gain").reindex(FAMILIES + ["ALL"]).reset_index()
        w.columns = [str(cc) for cc in w.columns]
        table(w, "T5b_cd1_vs_qaoa2_by_n_optimized",
              "T5b. CD p=1 vs QAOA p=2, optimized (train, best of 8), by n and family")
        w = uo.pivot(index="family", columns="n", values="cd_win").reindex(FAMILIES + ["ALL"]).reset_index()
        w.columns = [str(cc) for cc in w.columns]
        table(w, "T5c_cd1_vs_qaoa2_winrate_optimized", "T5c. CD win rate, CD p=1 vs QAOA p=2, optimized")

    # Figure: ratio vs two-qubit rotations per edge (small run, fixed + optimized)
    s2 = same_depth_cache["small"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for a, (cdcost, ttl, xmax) in zip(ax, ((3, "(a) unoptimized decomposition: CD layer = 3", 18.5),
                                           (2, "(b) level-3 compiled: CD layer = 2", 12.5))):
        a.plot(s2.p, s2.ratio_qaoa_fixed, "-s", color=C_Q, ms=4, lw=2, label="QAOA fixed")
        a.plot(s2.p * cdcost, s2.ratio_cd_fixed, "-o", color=C_CD, ms=4, lw=2, label="CD fixed")
        a.plot(s2.p, s2.ratio_qaoa_opt, ":", color=C_Q, lw=1.5, label="QAOA optimized")
        a.plot(s2.p * cdcost, s2.ratio_cd_opt, ":", color=C_CD, lw=1.5, label="CD optimized")
        style(a, "two-qubit gate cost per edge (QAOA-layer units)", None, ttl)
        a.set_xlim(0.5, xmax)
    ax[0].set_ylabel("mean approximation ratio (n=10–16)", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "F3_equal_budget")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    allr = u[u.family == "ALL"]
    for fam, col in (("bipartite", C_CD), ("ring", C_3), ("regular3", MUTED), ("tree", "#6250d6")):
        s = u[u.family == fam]
        ax.plot(s.n, s.gain_cd1_vs_qaoa2, "-o", ms=3.5, color=col, lw=1.5, label=fam)
    ax.plot(allr.n, allr.gain_cd1_vs_qaoa2, "-", color=INK, lw=2.5, label="all families")
    if len(uo):
        ao = uo[uo.family == "ALL"]
        ax.plot(ao.n, ao.gain, "--", color=INK, lw=2, label="all families, optimized")
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "n", "(E_QAOA,p=2 − E_CD,p=1)/Σ|J|", "CD p=1 vs QAOA p=2 (equal compiled CX)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "F6_cd1_vs_qaoa2_vs_n")
    return t


def transfer_loss(H):
    rows = []
    for run, d in H.items():
        opt = load_json(RES / RUNS[run] / "summary.json")["optimized_train"]
        fx = load_json(RES / RUNS[run] / "summary.json")["fixed_formula_test"]
        for key, o in opt.items():
            if key not in fx or ("family=" not in key) or "|weights=" in key or "|n=" in key:
                continue
            p, fam = key.split("|")
            f = fx[key]
            rows.append({"run": run, "p": int(p[1:]), "family": fam.split("=")[1],
                         "opt_cd": o["cd"]["ratio"]["mean"], "opt_qaoa": o["qaoa"]["ratio"]["mean"],
                         "fixed_cd": f["ratio_cd_proposed"]["mean"], "fixed_qaoa": f["ratio_qaoa_best"]["mean"]})
    t = pd.DataFrame(rows)
    t["transfer_loss_cd"] = t.opt_cd - t.fixed_cd
    t["transfer_loss_qaoa"] = t.opt_qaoa - t.fixed_qaoa
    t["opt_advantage"] = t.opt_cd - t.opt_qaoa
    t["fixed_advantage"] = t.fixed_cd - t.fixed_qaoa
    t.to_csv(TAB / "T6_transfer_loss_long.csv", index=False)
    w = t[t.run.isin(["small", "large"])].pivot_table(index="family", columns=["run", "p"],
                                                        values="transfer_loss_cd").reindex(FAMILIES)
    w.columns = [f"{r}_p{p}" for r, p in w.columns]
    table(w.reset_index(), "T6_transfer_loss_cd",
          "T6. CD transfer loss = optimized ratio (train) − fixed-formula ratio (test), by family")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    s = t[t.run == "n18p6"]
    xs = np.arange(len(FAMILIES))
    for i, (p, col) in enumerate(zip((1, 3, 6), (C_3, C_CD, INK))):
        v = s[s.p == p].set_index("family").reindex(FAMILIES)
        ax.bar(xs + (i - 1) * 0.27, v.transfer_loss_cd, 0.25, color=col, label=f"CD, p={p}",
               edgecolor="white", linewidth=0.5)
    v6 = s[s.p == 6].set_index("family").reindex(FAMILIES)
    ax.plot(xs + 0.27, v6.transfer_loss_qaoa, "_", color=C_Q, ms=12, mew=2, label="QAOA, p=6")
    ax.set_xticks(xs, FAMILIES, rotation=40, ha="right")
    style(ax, None, "optimized − fixed approximation ratio", "Transfer loss of the fixed formula by family (n=18)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "F5_transfer_loss")
    return t


def heterogeneity(H):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    d = pd.concat([H["small"], H["large"]])
    F = {"f_degree_mean": "mean degree", "f_degree_cv": "degree CV", "f_is_bipartite": "bipartite",
         "f_clustering": "clustering", "f_weight_cv": "weight CV", "f_weight_max_over_rms": "max|J|/rms",
         "f_weight_skew": "weight skew", "n": "n", "f_algebraic_connectivity": "algebraic connectivity",
         "f_density": "density", "f_degree_min": "min degree"}
    rows, quart = [], []
    for p in (1, 2, 3, 4, 6):
        s = d[d.p == p]
        X, y = s[list(F)].astype(float), s.gain_same_depth_norm
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
        r2 = cross_val_score(m, X, y, cv=5, scoring="r2").mean()
        m.fit(X, y)
        pi = permutation_importance(m, X, y, n_repeats=5, random_state=0)
        for k, v in zip(F, pi.importances_mean):
            rows.append({"p": p, "feature": F[k], "importance": v, "cv_r2": r2, "instances": len(s)})
        for feat in ("f_degree_cv", "f_weight_max_over_rms"):
            qbin = pd.qcut(s[feat].rank(method="first"), 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
            for qb, ss in s.groupby(qbin, observed=True):
                quart.append({"p": p, "feature": F[feat], "quartile": qb, "gain": ss.gain_same_depth_norm.mean(),
                              "cd_win": (ss.gain_same_depth_norm > 0).mean(), "instances": len(ss)})
    imp = pd.DataFrame(rows)
    w = imp.pivot(index="feature", columns="p", values="importance")
    w = w.loc[w.max(axis=1).sort_values(ascending=False).index].reset_index()
    w.columns = [str(c) if not isinstance(c, int) else f"p{c}" for c in w.columns]
    table(w, "T7_feature_importance", "T7. Permutation importance of graph features for the same-depth fixed gain "
          f"(gradient boosting; 5-fold CV R² = {', '.join(f'p{p}: {r:.2f}' for p, r in imp.groupby('p').cv_r2.first().items())})")
    qt = pd.DataFrame(quart)
    w = qt.pivot_table(index=["feature", "quartile"], columns="p", values="gain", observed=True).reset_index()
    w.columns = [str(c) if not isinstance(c, int) else f"p{c}" for c in w.columns]
    table(w, "T8_gain_by_heterogeneity_quartile", "T8. Same-depth fixed gain by heterogeneity quartile (n=10–22)")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for a, feat in zip(ax, ("degree CV", "max|J|/rms")):
        s = qt[qt.feature == feat]
        for p, col in zip((1, 2, 3, 4, 6), ("#9ec5f0", "#6aa6e8", C_CD, "#1d5aa5", INK)):
            v = s[s.p == p]
            a.plot(range(4), v.gain, "-o", ms=3.5, color=col, lw=1.6, label=f"p={p}")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xticks(range(4), ["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        style(a, f"{feat} quartile", None, f"({'ab'[feat != 'degree CV']}) by {feat}")
    ax[0].set_ylabel("same-depth fixed gain /Σ|J|", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "F7_heterogeneity")


def templates():
    rows = []
    for run in RUNS:
        m = load_json(RES / RUNS[run] / "FROZEN_MODEL.json")
        for p, t in m["templates"].items():
            for meth in ("qaoa", "cd"):
                e = t[meth]
                for l in range(int(p)):
                    rows.append({"run": run, "p": int(p), "method": meth, "layer": l + 1,
                                 "gamma_unit": e["gamma_unit"][l], "beta": e["beta"][l], "alpha_unit": e["alpha_unit"][l],
                                 "median_gamma_unit": e["median_gamma_unit"][l], "median_beta": e["median_beta"][l],
                                 "median_alpha_unit": e["median_alpha_unit"][l],
                                 "reference_degree": e["reference_degree"], "medoid_id": e["medoid_record_id"]})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "T9_frozen_templates_long.csv", index=False)
    cd = t[(t.method == "cd") & (t.run == "small")][["p", "layer", "gamma_unit", "beta", "alpha_unit"]].astype({"p": str, "layer": str})
    table(cd, "T9_cd_template_small", "T9. Frozen CD angle table (run small, medoid template; γ = γ_unit·f_D(D)/s_J, "
          "α = α_unit/s_J·(D/D0)^−0.5)", 5)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for run, ls, mk in (("small", "-", "o"), ("large", "--", "s"), ("n18p6", ":", "^")):
        for p, col in ((4, C_CD), (6, INK)):
            s = t[(t.run == run) & (t.p == p) & (t.method == "cd")]
            if s.empty:
                continue
            x = (s.layer - 0.5) / p
            for a, key in zip(ax, ("gamma_unit", "beta", "alpha_unit")):
                a.plot(x, s[key], ls, marker=mk, ms=3.5, color=col, lw=1.4,
                       label=f"p={p}, {RUN_LABEL[run]}" if key == "gamma_unit" else None)
    for a, ttl in zip(ax, ("(a) γ_unit", "(b) β", "(c) α_unit")):
        style(a, "layer position (l−½)/p", None, ttl)
    ax[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "F8_cd_templates")


def init_dependence(best_restarts):
    rows, byn = [], []
    for run, r in best_restarts.items():
        z = r.groupby("id").sum_abs_weights.first()
        nn = r.groupby("id").n.first()
        a = r.groupby(["id", "p", "cd"]).energy.min()
        rnd = r[r.init_style == "random"].groupby(["id", "p", "cd"]).energy.min()
        g = ((rnd - a) / z.reindex(a.index.get_level_values(0)).values)
        win = r.loc[r.groupby(["id", "p", "cd"]).energy.idxmin()]
        for (p, cd), s in g.groupby(level=[1, 2]):
            ws = win[(win.p == p) & (win.cd == cd)].init_style.value_counts(normalize=True)
            rows.append({"run": run, "p": p, "method": "CD" if cd else "QAOA",
                         "random_only_gap_mean": s.mean(), "frac_random_only_worse": (s > 1e-6).mean(),
                         **{f"best_from_{k}": ws.get(k, 0.0) for k in ("random", "ramp", "interp", "from_qaoa")},
                         "not_converged_%": 100 * (~r[(r.p == p) & (r.cd == cd)].success).mean()})
        if run == "n18p6":
            continue
        gn = g.reset_index()
        gn.columns = ["id", "p", "cd", "gap"]
        gn["n"] = gn.id.map(nn)
        for (n, p, cd), s in gn.groupby(["n", "p", "cd"]):
            byn.append({"n": n, "p": p, "method": "CD" if cd else "QAOA", "random_only_gap_mean": s.gap.mean(),
                        "frac_random_only_worse": (s.gap > 1e-6).mean()})
    t = pd.DataFrame(rows)
    table(t, "T10_init_dependence", "T10. Dependence of the optimized ceiling on the depth-chain initialisations")
    u = pd.DataFrame(byn)
    if len(u):
        w = u.pivot_table(index=["method", "p"], columns="n", values="random_only_gap_mean").reset_index()
        w.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"n{c}" for c in w.columns]
        table(w, "T10b_init_dependence_by_n", "T10b. Random-start-only excess energy (/Σ|J|) by n")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for run, ls in (("small", "-"), ("large", "--"), ("n18p6", ":")):
        for meth, col, mk in (("CD", C_CD, "o"), ("QAOA", C_Q, "s")):
            s = t[(t.method == meth) & (t.run == run)]
            if len(s):
                ax[0].plot(s.p, s.random_only_gap_mean, ls + mk, color=col, ms=4, lw=1.8,
                           label=f"{meth}, {RUN_LABEL[run]}")
    style(ax[0], "layers p", "(best random − best overall)/Σ|J|", "(a) by depth")
    ax[0].legend(fontsize=6.5, frameon=False)
    if len(u):
        for p, col in ((2, "#9ec5f0"), (4, C_CD), (6, INK)):
            for meth, ls in (("CD", "-o"), ("QAOA", "--s")):
                s = u[(u.p == p) & (u.method == meth)]
                if len(s):
                    ax[1].plot(s.n, s.random_only_gap_mean, ls, color=col, ms=3.5, lw=1.5,
                               label=f"{meth}, p={p}")
        style(ax[1], "n", None, "(b) by size (solid CD, dashed QAOA)")
        ax[1].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "F9_init_dependence")


def template_aggregation():
    rows = []
    for run in RUNS:
        path = RES / "template_agg" / f"test_{RUNS[run]}.jsonl"
        if not path.exists():
            continue
        d = pd.read_json(path, lines=True)
        for p, s in d.groupby("p"):
            g1 = (s.e_qaoa_v1 - s.e_cd_v1) / s.z
            g2 = (s.e_qaoa_v2 - s.e_cd_v2) / s.z
            dc = (s.e_cd_v1 - s.e_cd_v2) / s.z
            m, lo, hi = boot(dc)
            rows.append({"run": run, "p": p, "gain_v1": g1.mean(), "gain_v2": g2.mean(), "cd_v2_minus_v1": m,
                         "ci_lo": lo, "ci_hi": hi, "qaoa_v2_minus_v1": ((s.e_qaoa_v1 - s.e_qaoa_v2) / s.z).mean()})
    if not rows:
        return
    t = pd.DataFrame(rows)
    sel = []
    for run in RUNS:
        mp = RES / "template_agg" / f"MODEL_V2_{RUNS[run]}.json"
        if mp.exists():
            m = load_json(mp)
            for p, s in m["template_selection"].items():
                sel.append({"run": run, "p": int(p), "cd_template": s["cd"], "qaoa_choice": s["qaoa"]})
    if sel:
        t = t.merge(pd.DataFrame(sel), on=["run", "p"], how="left")
    table(t, "T11_template_aggregation", "T11. Template aggregation chosen on validation (MODEL_V2) vs frozen medoid (v1), test set")


def optimizer_audit():
    rows = []
    for run in RUNS:
        a = load_json(RES / RUNS[run] / "optimizer_audit.json")
        nc = pd.DataFrame(a["not_converged"])
        rows.append({"run": run, "optimizations": a["records"], "not_converged": len(nc),
                     "abnormal": int(nc.message.str.startswith("ABNORMAL").sum()) if len(nc) else 0,
                     "maxiter_hit": int(nc.message.str.contains("ITERATIONS").sum()) if len(nc) else 0,
                     "instance_p_method_all8_failed": int((nc.groupby(["id", "p", "cd"]).size() == 8).sum()) if len(nc) else 0,
                     "eval_calls": a["total_eval_calls"], "gpu_slot_hours": a["total_wall_seconds"] / 3600})
    table(pd.DataFrame(rows), "T12_optimizer_audit", "T12. Optimizer audit (complex64 L-BFGS-B, 8 restarts)", 1)


same_depth_cache = {}


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.append("# Equal-depth CD-QAOA GPU study — analysis report\n\n"
                  "Generated by `analysis/analyze_eqdepth.py`. gain = (E_QAOA − E_CD)/Σ|J| (positive favours CD); "
                  "ratio = MaxCut approximation ratio (exact extrema). CIs: instance bootstrap, 95%.\n")
    H = {run: heldout(run) for run in RUNS}
    best, restarts = {}, {}
    for run in RUNS:
        b = RES / RUNS[run] / "best_optima.csv.gz"
        if b.exists():
            best[run] = pd.read_csv(b)
        r = RES / RUNS[run] / "restart_summary.csv.gz"
        if r.exists():
            restarts[run] = pd.read_csv(r)
    REPORT.append("## 1. Data\n")
    dataset_table(H)
    REPORT.append("## 2. Same depth\n")
    t = same_depth(H)
    for run in RUNS:
        same_depth_cache[run] = t[t.run == run]
    breakdown(H)
    REPORT.append("## 3. Equal two-qubit budget\n")
    equal_budget(H, best)
    REPORT.append("## 4. Where the fixed formula loses\n")
    transfer_loss(H)
    heterogeneity(H)
    REPORT.append("## 5. Angle templates and optimisation\n")
    templates()
    if restarts:
        init_dependence(restarts)
    optimizer_audit()
    template_aggregation()
    (OUT / "ANALYSIS_REPORT.md").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"wrote {OUT / 'ANALYSIS_REPORT.md'}, {len(list(TAB.glob('*.csv')))} tables, "
          f"{len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
