#!/usr/bin/env python3
"""Extra figures/tables for the PRR revision of the formula-based DC-QAOA paper.

Reads only archived files in this repository; no quantum simulation, no GPU.
From the repository root:

    python analysis/prr_figures.py                    # p=1, layer 1
    python analysis/prr_figures.py --p 4 --layer 1

Outputs (default: analysis/outputs_prr/):
  P1_validation_grid.{png,pdf,csv}  validation energy vs CD degree exponent b
                                    and multiplier c, three cohorts
  P2_collapse_p{p}_l{layer}.{png,pdf} trained angles vs mean degree, before/after
                                    the RMS-weight and degree rescaling
  P2_collapse_by_family_p{p}_l{layer}.csv  per-family medians of the rescaled angles
  P3_exponents.csv                  fitted degree exponent of alpha_l*s_J and
                                    gamma_l*s_J per depth/layer, graph bootstrap
  P4_transfer_loss.csv              population-level transfer loss from T6

Sources:
  results/gpu_eqdepth/eqdepth_{small_N20,large_N20,n18p6_N20}/FROZEN_MODEL.json
  exports/cd-qaoa-gpu-transfer/data/{small,large,extra}_best_optima.csv.gz
  exports/cd-qaoa-gpu-transfer/assets/unified_gpu_model.json
  analysis/outputs/tables/T6_transfer_loss_long.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COHORTS = {"small": "eqdepth_small_N20", "large": "eqdepth_large_N20", "n18p6": "eqdepth_n18p6_N20"}
COHORT_LABEL = {"small": "10–16 qubits, p ≤ 6", "large": "18–22 qubits, p ≤ 4", "n18p6": "18 qubits, p = 6"}
FAMILIES = ["tree", "ring", "regular3", "regular4", "regular5", "regular6",
            "er3", "er05", "community", "bipartite"]
BOOT = 500
SEED = 20261024


# ------------------------------------------------------------------ helpers
def f_D(D):
    D = np.asarray(D, dtype=float)
    out = np.ones_like(D)
    m = D > 1
    out[m] = np.arctan(1.0 / np.sqrt(D[m] - 1.0))
    return out


def _parse_vec(s):
    return np.asarray(json.loads(s), dtype=float)


def load_frozen_models(root: Path) -> dict:
    return {c: json.loads((root / "results" / "gpu_eqdepth" / d / "FROZEN_MODEL.json").read_text(encoding="utf-8"))
            for c, d in COHORTS.items()}


def load_unified_model(root: Path) -> dict:
    return json.loads((root / "exports" / "cd-qaoa-gpu-transfer" / "assets" / "unified_gpu_model.json")
                      .read_text(encoding="utf-8"))


def load_training_optima(root: Path) -> pd.DataFrame:
    """Best-of-8 CD-QAOA training optimum per (graph, p), sign-aligned as in Sec. 2.3."""
    data = root / "exports" / "cd-qaoa-gpu-transfer" / "data"
    frames = []
    for name in ("small", "large", "extra"):
        df = pd.read_csv(data / f"{name}_best_optima.csv.gz", dtype={"cd": str, "gamma": str, "beta": str, "alpha": str})
        df = df[df["cd"].str.strip().str.lower() == "true"]
        if name == "extra":              # extra p<=4 duplicates large (GUIDE_zh.md, Sec. 5)
            df = df[df["p"] == 6]
        df = df.assign(source=name)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[df.groupby(["id", "p"])["energy"].idxmin()].reset_index(drop=True)

    rows = []
    for r in df.itertuples(index=False):
        g, b, a = _parse_vec(r.gamma), _parse_vec(r.beta), _parse_vec(r.alpha)
        flipped = False
        if g.mean() > 0:                 # equivalent branch: (γ,β,α) -> (−γ,−β,α)
            g, b, flipped = -g, -b, True
        rows.append({"id": r.id, "n": int(r.n), "family": r.family, "weight_kind": r.weight_kind,
                     "p": int(r.p), "D": float(r.degree), "s": float(r.rms), "source": r.source,
                     "gamma": g, "beta": b, "alpha": a, "flipped": flipped})
    out = pd.DataFrame(rows)
    print(f"[optima] {len(out)} (graph, p) CD records; {out.flipped.mean():.1%} sign-flipped; "
          f"D range {out.D.min():.2f}–{out.D.max():.2f}")
    return out


def layer_values(df: pd.DataFrame, p: int, layer: int, D0: float) -> pd.DataFrame:
    """Raw (weight-scale removed) and fully rescaled layer-`layer` angles at depth p."""
    d = df[df.p == p].copy()
    l = layer - 1
    d["g_raw"] = [g[l] * s for g, s in zip(d.gamma, d.s)]            # γ_l s_J
    d["g_col"] = d.g_raw / f_D(d.D.values)                            # γ_l s_J / f_D(D)  -> γ_unit
    d["a_raw"] = [a[l] * s for a, s in zip(d.alpha, d.s)]            # α_l s_J
    d["a_col"] = d.a_raw * np.sqrt(d.D.values / D0)                   # α_l s_J (D/D0)^{1/2} -> α_unit
    return d


# ------------------------------------------------------------------ P1: validation grid
def fig_validation_grid(models: dict, out: Path):
    recs = []
    for c, m in models.items():
        for r in m["alpha_grid_val"]:
            recs.append({"cohort": c, "exponent": r["alpha_degree_power"], "multiplier": r["alpha_multiplier"],
                         "validation_normalized_energy": r["validation_normalized_energy"],
                         "selected": (r["alpha_degree_power"] == m["alpha_degree_power"]
                                      and r["alpha_multiplier"] == m["alpha_multiplier"])})
    t = pd.DataFrame(recs).sort_values(["cohort", "multiplier", "exponent"])
    t.to_csv(out / "P1_validation_grid.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, c in zip(axes, ("small", "large", "n18p6")):
        s = t[t.cohort == c]
        for mult, mk in zip((0.5, 1.0, 1.5), ("s", "o", "^")):
            q = s[s.multiplier == mult]
            ax.plot(q.exponent, q.validation_normalized_energy, marker=mk, ms=5, lw=1.2, label=f"$c_\\alpha={mult}$")
        best = s[s.selected].iloc[0]
        ax.plot(best.exponent, best.validation_normalized_energy, "o", ms=11, mfc="none", mec="k", mew=1.2)
        ax.axvline(-0.5, color="0.4", ls=":", lw=1)
        ax.set_title(COHORT_LABEL[c], fontsize=9)
        ax.set_xlabel("CD degree exponent $b$")
        ax.set_xticks([-0.5, 0, 0.5, 1.0])
    axes[0].set_ylabel(r"validation $\langle H_C\rangle/\sum_e J_e$  (lower is better)")
    axes[0].text(0.06, 0.96, "dotted: second-moment scale $b=-1/2$\ncircle: selected on validation",
                 transform=axes[0].transAxes, fontsize=7, va="top", color="0.3")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "P1_validation_grid.png", dpi=220)
    fig.savefig(out / "P1_validation_grid.pdf")
    plt.close(fig)
    sel = t[t.selected][["cohort", "exponent", "multiplier", "validation_normalized_energy"]]
    print("[P1] selected per cohort:\n", sel.to_string(index=False))


# ------------------------------------------------------------------ P2: collapse figure
def bin_stats(x, y, width=0.5, min_count=10):
    """Median and IQR of y in degree bins of fixed width; returns (D_center, median, q25, q75, count)."""
    centers = np.round(np.asarray(x) / width) * width
    rows = []
    for c in np.unique(centers):
        m = centers == c
        if m.sum() < min_count:
            continue
        q = np.quantile(np.asarray(y)[m], [0.25, 0.5, 0.75])
        rows.append((np.median(np.asarray(x)[m]), q[1], q[0], q[2], int(m.sum())))
    return tuple(np.asarray(v) for v in zip(*rows))


def _binned(ax, x, y, color="k"):
    xs, med, lo, hi, _ = bin_stats(x, y)
    ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt="o-", color=color, ms=3.5, lw=1.3, capsize=2, zorder=5,
                label="median, IQR (0.5-wide D bins)")


def _spread(y):
    q = np.quantile(y, [0.25, 0.5, 0.75])
    return (q[2] - q[0]) / abs(q[1]) if q[1] != 0 else np.nan


def fig_collapse(df: pd.DataFrame, unified: dict, p: int, layer: int, out: Path):
    D0 = float(unified["reference_degree"])
    tpl = unified["templates"][str(p)]
    g_unit, a_unit = tpl["gamma_unit"][layer - 1], tpl["alpha_unit"][layer - 1]
    d = layer_values(df, p, layer, D0)
    Dgrid = np.linspace(d.D.min(), d.D.max(), 200)
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i) for i, f in enumerate(FAMILIES)}

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    panels = [
        (axes[0, 0], "g_raw", r"$\gamma_%d\,s_J$" % layer, g_unit * f_D(Dgrid), "rule: $\\gamma^{\\rm unit}f_D(D)$"),
        (axes[0, 1], "g_col", r"$\gamma_%d\,s_J/f_D(D)$" % layer, np.full_like(Dgrid, g_unit), "table $\\gamma^{\\rm unit}$"),
        (axes[1, 0], "a_raw", r"$\alpha_%d\,s_J$" % layer, a_unit * np.sqrt(D0 / Dgrid), "rule: $\\alpha^{\\rm unit}(D/D_0)^{-1/2}$"),
        (axes[1, 1], "a_col", r"$\alpha_%d\,s_J\,(D/D_0)^{1/2}$" % layer, np.full_like(Dgrid, a_unit), "table $\\alpha^{\\rm unit}$"),
    ]
    for ax, col, ylabel, curve, clabel in panels:
        for fam in FAMILIES:
            s = d[d.family == fam]
            ax.scatter(s.D, s[col], s=5, alpha=0.25, color=colors[fam], label=fam, linewidths=0)
        _binned(ax, d.D.values, d[col].values)
        (rule_line,) = ax.plot(Dgrid, curve, "--", color="crimson", lw=1.4, label=clabel, zorder=6)
        ax.legend(handles=[rule_line], fontsize=7, frameon=False, loc="upper right")
        ax.set_ylabel(ylabel)
        ax.text(0.02, 0.04, f"IQR/|median| = {_spread(d[col].values):.2f}", transform=ax.transAxes, fontsize=8)
        # view = central 96 % of points; the statistics use all points.  Points far outside are
        # mostly symmetry-equivalent branches (e.g. gamma -> gamma - pi for unit weights).
        lo, hi = np.quantile(d[col], [0.02, 0.98]); pad = 0.25 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        outside = int(((d[col] < lo - pad) | (d[col] > hi + pad)).sum())
        ax.text(0.98, 0.12, f"{outside} of {len(d)} outside view", transform=ax.transAxes, fontsize=7,
                ha="right", color="0.4")
    axes[0, 0].set_title("weight scale removed", fontsize=10)
    axes[0, 1].set_title("weight and degree scale removed", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("mean degree $D$")
    h, l = axes[1, 1].get_legend_handles_labels()
    keep = [i for i, lab in enumerate(l) if not lab.startswith(("rule", "table"))]
    fig.legend([h[i] for i in keep], [l[i] for i in keep], loc="lower center", ncol=6, fontsize=7, frameon=False,
               markerscale=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Trained CD-QAOA optima, depth p={p}, layer {layer}: {len(d)} training graphs", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out / f"P2_collapse_p{p}_l{layer}.png", dpi=220)
    fig.savefig(out / f"P2_collapse_p{p}_l{layer}.pdf")
    plt.close(fig)

    fam = (d.groupby("family")[["D", "g_raw", "g_col", "a_raw", "a_col"]].median()
           .reindex(FAMILIES).assign(count=d.groupby("family").size().reindex(FAMILIES)))
    fam.to_csv(out / f"P2_collapse_by_family_p{p}_l{layer}.csv")
    print(f"[P2] p={p} layer {layer}: spread IQR/|median|  γ raw {_spread(d.g_raw):.2f} -> rescaled {_spread(d.g_col):.2f};"
          f"  α raw {_spread(d.a_raw):.2f} -> rescaled {_spread(d.a_col):.2f}")
    xs, med, lo, hi, cnt = bin_stats(d.D.values, d.a_raw.values)
    print("[P2] α_1 s_J by degree bin:", ", ".join(f"D≈{x:.1f}: {m:+.3f} (n={c})" for x, m, c in zip(xs, med, cnt)))
    print(f"[P2] medians: γ_unit {d.g_col.median():+.4f} (table {g_unit:+.4f});  α_unit {d.a_col.median():+.4f} (table {a_unit:+.4f})")


# ------------------------------------------------------------------ P3: fitted exponents
def _loglog_slope(D, y):
    """Slope of log|median_bin(y)| vs log D over 0.5-wide degree bins, weighted by bin counts."""
    xs, med, _, _, cnt = bin_stats(D, y)
    keep = (med != 0) & (xs > 1)
    xs, med, cnt = xs[keep], np.abs(med[keep]), cnt[keep]
    if len(xs) < 3:
        return np.nan
    w = np.sqrt(cnt)
    X = np.column_stack([np.ones_like(xs), np.log(xs)]) * w[:, None]
    return np.linalg.lstsq(X, np.log(med) * w, rcond=None)[0][1]


def table_exponents(df: pd.DataFrame, D0: float, out: Path):
    rng = np.random.default_rng(SEED)
    rows = []
    for p in sorted(df.p.unique()):
        for layer in range(1, p + 1):
            d = layer_values(df, p, layer, D0)
            d = d[d.D > 1]
            for qty, col in (("alpha_l*s_J", "a_raw"), ("gamma_l*s_J", "g_raw")):
                y = d[col].values; D = d.D.values
                keep = np.ones(len(y), bool)                      # bin medians are robust to branch outliers
                slope = _loglog_slope(D, y)
                boots = [_loglog_slope(D[i], y[i]) for i in (rng.integers(0, len(D), len(D)) for _ in range(BOOT))]
                rule = -0.5 if col == "a_raw" else _loglog_slope(D, -f_D(D))  # slope implied by the rule on the same D
                rows.append({"p": p, "layer": layer, "quantity": qty, "n_records": int(keep.sum()),
                             "fitted_exponent": slope, "ci_lo": np.quantile(boots, 0.025),
                             "ci_hi": np.quantile(boots, 0.975), "rule_exponent": rule})
    t = pd.DataFrame(rows)
    t.to_csv(out / "P3_exponents.csv", index=False)
    print("[P3] fitted degree exponents (layer 1 only shown):\n",
          t[t.layer == 1].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ------------------------------------------------------------------ P4: transfer loss
def table_transfer_loss(root: Path, out: Path):
    t = pd.read_csv(root / "analysis" / "outputs" / "tables" / "T6_transfer_loss_long.csv")
    g = (t.groupby(["run", "p"])[["opt_cd", "fixed_cd", "opt_qaoa", "fixed_qaoa"]].mean()
         .assign(transfer_loss_cd=lambda x: x.opt_cd - x.fixed_cd,
                 fixed_cd_minus_opt_qaoa=lambda x: x.fixed_cd - x.opt_qaoa).reset_index())
    g.to_csv(out / "P4_transfer_loss.csv", index=False)
    print("[P4] optimized (train split) vs fixed formula (test split), cohort means over families:\n",
          g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ------------------------------------------------------------------ main
def main():
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=here.parents[1], help="repository root (default: parent of analysis/)")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "analysis" / "outputs_prr", help="default: analysis/outputs_prr")
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--layer", type=int, default=1)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    fig_validation_grid(load_frozen_models(a.root), out)
    unified = load_unified_model(a.root)
    optima = load_training_optima(a.root)
    fig_collapse(optima, unified, a.p, a.layer, out)
    table_exponents(optima, float(unified["reference_degree"]), out)
    table_transfer_loss(a.root, out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
