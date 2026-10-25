"""Side check beyond n = 22: optimized CD p=1 vs QAOA p=2 at n = 20-28 (paper Appendix C, Fig. 10).

    cd Clean
    python analysis/analyze_largen.py     # -> analysis/outputs/{tables/T13*,figures/F10*}, ANALYSIS_REPORT_largen.md

Inputs:
  results/gpu_eqdepth/cd1_vs_qaoa2_largen/records.jsonl          new run (scripts/25_cd1_vs_qaoa2_largen.py)
  results/gpu_eqdepth/eqdepth_{small,large}_N20/restart_summary.csv.gz   main runs, every restart
The new run uses a reduced protocol (CD p=1: QAOA warm start + ramp; QAOA p=2: INTERP + ramp) and
only unit and lognormal weights.  The same reduced protocol is emulated on the main runs (best of the
same start types instead of best of 8) to measure its bias, and the main runs are restricted to the
same two weight laws so that both data sets can be put on one axis.
Dependencies: numpy, pandas, matplotlib.
"""
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
FAMILIES = ["tree", "ring", "regular3", "regular4", "regular5", "regular6", "er3", "er05", "community", "bipartite"]
C_CD, C_Q, C_3, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
RNG = np.random.default_rng(1441)
REPORT = ["# Side check: CD p=1 vs QAOA p=2 beyond n = 22\n"]
REDUCED = {True: {"from_qaoa", "ramp"}, False: {"interp", "ramp"}}


def boot(x, n=4000):
    x = np.asarray(x, float)
    m = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def table(df, name, title, floatfmt=".4f"):
    df.to_csv(TAB / f"{name}.csv", index=False)
    REPORT.append(f"### {title}\n\n`tables/{name}.csv`\n")
    REPORT.append(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


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


def slope(df, col):
    """d gain / d n with family and weight-law fixed effects (OLS)."""
    X = pd.get_dummies(df[["family", "weight_kind"]].astype(str)).astype(float)
    X = X.iloc[:, :-1]  # drop one weight dummy (collinear with the family dummies)
    X["n"] = df.n.values
    return np.linalg.lstsq(X.values, df[col].values, rcond=None)[0][-1]


def slope_ci(df, col, B=1000):
    b = [slope(df.iloc[RNG.integers(0, len(df), len(df))], col) for _ in range(B)]
    return slope(df, col), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main_runs():
    rows = []
    for run in ("eqdepth_small_N20", "eqdepth_large_N20"):
        d = pd.read_csv(RES / run / "restart_summary.csv.gz",
                        usecols=["id", "n", "family", "weight_kind", "p", "cd", "init_style", "energy", "sum_abs_weights"])
        d = d[(d.cd & (d.p == 1)) | (~d.cd & (d.p == 2))]
        full = d.groupby(["id", "cd"]).energy.min().unstack()
        keep = [s in REDUCED[c] for s, c in zip(d.init_style, d.cd)]
        red = d[keep].groupby(["id", "cd"]).energy.min().unstack()
        m = d.groupby("id")[["n", "family", "weight_kind", "sum_abs_weights"]].first()
        m["gain_full"] = (full[False] - full[True]) / m.sum_abs_weights
        m["gain_reduced"] = (red[False] - red[True]) / m.sum_abs_weights
        rows.append(m.reset_index())
    return pd.concat(rows, ignore_index=True)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    r = pd.read_json(RES / "cd1_vs_qaoa2_largen" / "records.jsonl", lines=True)
    if "error" in r.columns:
        r = r[r["error"].isna()]
    r["loss_cd1"] = (r.E_cd1_fixed - r.E_cd1_opt) / r.z
    r["gain_cd1_vs_qaoa1"] = (r.E_qaoa1_opt - r.E_cd1_opt) / r.z
    m = main_runs()
    mu = m[m.weight_kind.isin(["unit", "lognormal"])]

    # T13: new run by n
    rows = []
    for n, s in r.groupby("n"):
        go, lo, hi = boot(s.gain_opt)
        gf, lf, hf = boot(s.gain_fixed)
        rows.append({"n": n, "graphs": len(s), "gain_opt": go, "opt_lo": lo, "opt_hi": hi,
                     "cd_win_opt": (s.gain_opt > 1e-9).mean(), "gain_fixed": gf, "fixed_lo": lf, "fixed_hi": hf,
                     "cd_win_fixed": (s.gain_fixed > 0).mean(), "transfer_loss_cd1": s.loss_cd1.mean(),
                     "gain_cd1_vs_qaoa1_opt": s.gain_cd1_vs_qaoa1.mean(), "seconds_per_graph": s.seconds.mean()})
    for lab, s in (("24-28", r[r.n >= 24]),):
        go, lo, hi = boot(s.gain_opt)
        gf, lf, hf = boot(s.gain_fixed)
        rows.append({"n": lab, "graphs": len(s), "gain_opt": go, "opt_lo": lo, "opt_hi": hi,
                     "cd_win_opt": (s.gain_opt > 1e-9).mean(), "gain_fixed": gf, "fixed_lo": lf, "fixed_hi": hf,
                     "cd_win_fixed": (s.gain_fixed > 0).mean(), "transfer_loss_cd1": s.loss_cd1.mean(),
                     "gain_cd1_vs_qaoa1_opt": s.gain_cd1_vs_qaoa1.mean(), "seconds_per_graph": s.seconds.mean()})
    t13 = pd.DataFrame(rows)
    table(t13, "T13_largen_by_n", "T13. New run (unit + lognormal weights, reduced starts): CD p=1 vs QAOA p=2 by n")

    t13b = r.pivot_table(index="family", columns="n", values="gain_opt", aggfunc="mean").reindex(FAMILIES)
    t13b.columns = [f"n{c}" for c in t13b.columns]
    table(t13b.reset_index(), "T13b_largen_by_family", "T13b. New run: optimized gain by family and n")

    new_w = r.pivot_table(index="n", columns="weight_kind", values="gain_opt", aggfunc="mean")
    old_w = m.pivot_table(index="n", columns="weight_kind", values="gain_full", aggfunc="mean")
    t13c = pd.concat([old_w.add_prefix("main_"), new_w.add_prefix("new_")], axis=1).reset_index()
    table(t13c, "T13c_largen_by_weight", "T13c. Optimized gain by weight law: main runs (best of 8) and new run")

    prot = m.groupby("n").agg(full_all=("gain_full", "mean"), reduced_all=("gain_reduced", "mean")).join(
        mu.groupby("n").agg(full_unit_lognormal=("gain_full", "mean"), reduced_unit_lognormal=("gain_reduced", "mean"),
                            graphs_unit_lognormal=("gain_full", "size")))
    prot["reduced_minus_full"] = prot.reduced_unit_lognormal - prot.full_unit_lognormal
    table(prot.reset_index(), "T13d_protocol_check",
          "T13d. Main runs: reduced start set (as in the new run) vs best of 8 starts")

    tr = []
    for lab, df, col in (("new run, n=20-28", r, "gain_opt"),
                         ("main runs, n=10-22 (unit+lognormal)", mu, "gain_full"),
                         ("main runs, n=16-22 (unit+lognormal)", mu[mu.n >= 16], "gain_full"),
                         ("main runs, n=10-22 (all weights)", m, "gain_full")):
        b, lo, hi = slope_ci(df, col)
        tr.append({"data": lab, "graphs": len(df), "slope_per_qubit": b, "lo": lo, "hi": hi})
    table(pd.DataFrame(tr), "T13e_largen_trend",
          "T13e. Trend of the optimized gain with n (family and weight-law fixed effects, bootstrap 95% CI)",
          floatfmt=".5f")

    # F10
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
    a = ax[0]
    mo = mu.groupby("n").gain_full.apply(lambda x: boot(x)).apply(pd.Series)
    a.errorbar(mo.index - 0.15, mo[0], yerr=[mo[0] - mo[1], mo[2] - mo[0]], fmt="-s", color=MUTED, ms=4, lw=1.5,
               capsize=2, label="main runs (best of 8)")
    nr = t13[t13.n.apply(lambda v: isinstance(v, (int, np.integer)))]
    a.errorbar(nr.n.astype(int) + 0.15, nr.gain_opt, yerr=[nr.gain_opt - nr.opt_lo, nr.opt_hi - nr.gain_opt],
               fmt="-o", color=C_CD, ms=4, lw=2, capsize=2, label="new run (reduced starts)")
    a.axhline(0, color=MUTED, lw=0.8)
    style(a, "n", "(E_QAOA,p=2 − E_CD,p=1)/Σ|J|", "(a) optimized, unit + lognormal weights")
    a.legend(fontsize=7, frameon=False, loc="lower right")
    a = ax[1]
    for w, col in (("unit", C_3), ("lognormal", C_Q)):
        a.plot(old_w.index, old_w[w], "--s", color=col, ms=3.5, lw=1.3, alpha=0.8, label=f"{w}, main runs")
        a.plot(new_w.index, new_w[w], "-o", color=col, ms=4, lw=2, label=f"{w}, new run")
    a.axhline(0, color=MUTED, lw=0.8)
    style(a, "n", None, "(b) by weight law (optimized)")
    a.set_ylim(-0.016, 0.017)
    a.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    for a in ax:
        a.set_xticks(range(10, 29, 2))
    fig.tight_layout()
    fig.savefig(FIG / "F10_cd1_vs_qaoa2_largen.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    REPORT.append("![F10](figures/F10_cd1_vs_qaoa2_largen.png)\n")
    (OUT / "ANALYSIS_REPORT_largen.md").write_text("\n".join(REPORT), encoding="utf-8")
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
