"""Figures for the equal-depth study (extensions of fig01 / fig02 of the CPU paper)."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np

METHODS = [("qaoa_best", "QAOA p (fixed)", "o"), ("cd_proposed", "CD p (fixed, new)", "s"),
           ("cd_v1", "CD p (fixed, V1)", "^")]


def _mean_ci(x, n_boot=2000, seed=7):
    x = np.asarray([v for v in x if v is not None], float)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(x.size, size=(n_boot, x.size))].mean(1) if x.size > 1 else x
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def _metric(rows):
    """Exact ratio if available for every row, else normalized energy."""
    if rows and all(r.get("ratio_qaoa_best") is not None for r in rows):
        return "ratio", rows[0]["ratio_kind"].split(" ")[0]
    return "norm_energy", "E / sum|J| (lower is better)"


def _grouped(path, rows, depths, key, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = sorted({r[key] for r in rows})
    fig, axes = plt.subplots(len(depths), 1, figsize=(max(7, 0.9 * len(cats) + 3), 2.8 * len(depths)),
                             squeeze=False)
    for ax, p in zip(axes[:, 0], depths):
        sub = [r for r in rows if r["p"] == p]
        prefix, label = _metric(sub)
        for k, (m, name, marker) in enumerate(METHODS):
            pts = []
            for c in cats:
                vals = [r.get(f"{prefix}_{m}") for r in sub if r[key] == c]
                pts.append(_mean_ci(vals))
            pts = np.asarray(pts)
            if np.all(np.isnan(pts[:, 0])):
                continue
            xs = np.arange(len(cats)) + (k - 1) * 0.18
            ax.errorbar(xs, pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]],
                        fmt=marker, ms=5, capsize=2, label=name)
        wins = [np.mean([r["gain_same_depth_norm"] > 1e-9 for r in sub if r[key] == c]) for c in cats]
        for i, w in enumerate(wins):
            ax.annotate(f"{w:.0%}", (i, 1.0), xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=7)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=20, fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(f"CD p={p} vs QAOA p={p} (same layer); top: CD win rate", fontsize=9, pad=12)
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _optimized(path, records, depths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = {}
    for r in records:
        k = (r["id"], r["p"], r["cd"])
        if k not in best or r["energy"] < best[k]["energy"]:
            best[k] = r
    use_ratio = all(r.get("ratio") is not None for r in best.values())
    field = "ratio" if use_ratio else "normalized_energy"
    fams = sorted({r["family"] for r in best.values()})
    fig, ax = plt.subplots(figsize=(8, 4))
    for cd, marker in ((False, "o"), (True, "s")):
        for j, fam in enumerate(fams):
            ys = [_mean_ci([r[field] for r in best.values() if r["p"] == p and r["cd"] == cd
                            and r["family"] == fam])[0] for p in depths]
            ax.plot(depths, ys, marker=marker, ls="-" if cd else "--", color=f"C{j}",
                    label=f"{fam} {'CD' if cd else 'QAOA'}")
    ax.set(xlabel="layers p", ylabel=field, title="Optimized (train, best of restarts): CD p vs QAOA p")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(out: Path, records, test_rows, depths) -> list[str]:
    figdir = Path(out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    depths = sorted(depths)
    names = []
    if test_rows:
        _grouped(figdir / "fig01_graph_structures_same_layer.png", test_rows, depths, "family",
                 "Fixed parameters on held-out graphs: graph families")
        _grouped(figdir / "fig02_weight_distributions_same_layer.png", test_rows, depths, "weight_kind",
                 "Fixed parameters on held-out graphs: positive weight distributions")
        _grouped(figdir / "fig01b_qubits_same_layer.png", test_rows, depths, "n",
                 "Fixed parameters on held-out graphs: qubit number")
        names += ["figures/fig01_graph_structures_same_layer.png",
                  "figures/fig02_weight_distributions_same_layer.png",
                  "figures/fig01b_qubits_same_layer.png"]
    if records:
        _optimized(figdir / "fig03_optimized_same_layer.png", records, depths)
        names.append("figures/fig03_optimized_same_layer.png")
    return names
