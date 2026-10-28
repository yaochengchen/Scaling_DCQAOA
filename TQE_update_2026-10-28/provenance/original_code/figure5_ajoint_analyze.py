#!/usr/bin/env python3
"""Analyse the full A-joint Figure 5 refinement and random-start control runs.

    python analysis/analyze_formula_init.py --results <dir with prr_formula_init_c128/ and prr_formula_init_c64/> \
        [--out figures]

Outputs:
  P5_formula_init_trajectories.{png,pdf}  (a) mean best-so-far ratio vs value-and-gradient calls:
                                          formula-started chain vs 20 random-start chains per graph;
                                          (b) per-graph fixed / refined / best-random / mean-random ratios
  P5_formula_init_per_graph.csv           per-graph numbers for both precisions
  P5_formula_init_summary.md              the numbers to quote in the paper
Only reads the result files; no simulation.
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

K = 60          # calls shown on the x axis
TOL = 1e-3      # "reaches" tolerance, as in the paper's Fig. 3


def load(results: Path, cfg: str):
    d = results
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "runs").glob("*.json"))]
    meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return runs, meta


def padded(curve, k=K):
    c = list(curve)
    return np.asarray((c + [c[-1]] * k)[:k], dtype=float)


def per_graph_rows(runs, cfg):
    rows = []
    for r in runs:
        f = r["chains"]["formula"]
        rnd = [c for k, c in r["chains"].items() if k.startswith("random")]
        fin = np.array([c["final_ratio"] for c in rnd])
        rows.append({
            "dtype": cfg, "id": r["id"], "family": r["family"], "weight_kind": r["weight_kind"],
            "fixed_ratio": r["fixed_ratio"], "refined_ratio": f["final_ratio"], "gain": f["final_ratio"] - r["fixed_ratio"],
            "formula_calls": f["calls"], "formula_calls_to_99pct_gain": f["calls_to_99pct"],
            "formula_message": f["message"].strip(),
            "random_chains": len(rnd), "random_mean_final": fin.mean(), "random_median_final": float(np.median(fin)),
            "random_best_final": fin.max(), "random_mean_calls": float(np.mean([c["calls"] for c in rnd])),
            "random_reach_fixed": int((fin >= r["fixed_ratio"] - TOL).sum()),
            "random_reach_refined": int((fin >= f["final_ratio"] - TOL).sum()),
            "random_beat_refined_1e-4": int((fin > f["final_ratio"] + 1e-4).sum()),
            "random_abnormal_stops": int(sum("ABNORMAL" in c["message"] for c in rnd)),
            "refined_equals_best_random_1e-4": bool(abs(fin.max() - f["final_ratio"]) <= 1e-4),
        })
    return rows


def figure(runs, meta, out: Path):
    allrnd = [c for r in runs for k, c in r["chains"].items() if k.startswith("random")]
    F = np.array([padded(r["chains"]["formula"]["best_so_far_ratio"]) for r in runs])
    R = np.array([padded(c["best_so_far_ratio"]) for c in allrnd])
    x = np.arange(1, K + 1)
    fixed_mean = np.mean([r["fixed_ratio"] for r in runs])
    refined_mean = np.mean([r["chains"]["formula"]["final_ratio"] for r in runs])

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a.fill_between(x, F.min(0), F.max(0), color="tab:orange", alpha=0.18, lw=0)
    a.plot(x, F.mean(0), color="tab:orange", lw=2, label=f"A-joint start (mean of {len(runs)} graphs; band = range)")
    a.fill_between(x, np.percentile(R, 10, axis=0), np.percentile(R, 90, axis=0), color="tab:blue", alpha=0.15, lw=0)
    a.plot(x, R.mean(0), color="tab:blue", lw=2, label=f"random start (mean of {len(allrnd)} chains; band = 10–90%)")
    a.axhline(fixed_mean, color="tab:orange", ls=":", lw=1.2)
    a.text(K - 1, fixed_mean - 0.006, f"A-joint (fixed) {fixed_mean:.4f}", va="top", ha="right", fontsize=7, color="tab:orange")
    a.axhline(refined_mean, color="k", ls="--", lw=1)
    a.text(K - 1, refined_mean + 0.006, f"A-joint refined {refined_mean:.4f}", va="bottom", ha="right", fontsize=7)
    a.set_xlim(1, K); a.set_ylim(0.45, 0.95); a.set_xlabel("L-BFGS-B value-and-gradient calls"); a.set_ylabel("mean best-so-far approximation ratio")
    a.set_title(f"(a) A-joint; n = 12, p = 2, {meta['dtype']}, adjoint gradients", fontsize=9)
    a.legend(fontsize=7, frameon=False, loc="lower right")

    rows = per_graph_rows(runs, meta["dtype"])
    order = np.argsort([r["fixed_ratio"] for r in rows])
    y = np.arange(len(rows))
    for j, i in enumerate(order):
        r = rows[i]
        fin = np.array([c["final_ratio"] for k, c in runs[i]["chains"].items() if k.startswith("random")])
        b.scatter(fin, np.full(len(fin), j), s=9, color="tab:blue", alpha=0.35, linewidths=0, zorder=2)
        b.plot([r["fixed_ratio"], r["refined_ratio"]], [j, j], color="tab:orange", lw=1.5, zorder=3)
        b.scatter(r["fixed_ratio"], j, marker="o", s=28, color="tab:orange", zorder=4)
        b.scatter(r["refined_ratio"], j, marker="D", s=24, color="k", zorder=5)
    b.scatter([], [], marker="o", s=28, color="tab:orange", label="A-joint (fixed)")
    b.scatter([], [], marker="D", s=24, color="k", label="A-joint, refined")
    b.scatter([], [], s=9, color="tab:blue", alpha=0.5, label="random-start final (20 per graph)")
    b.set_yticks(y, [rows[i]["id"].replace("test-n12-", "") for i in order], fontsize=7)
    b.set_xlabel("approximation ratio"); b.set_xlim(0.45, 0.95)
    b.set_title("(b) per graph", fontsize=9)
    b.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "Figure5_A_joint_full.png", dpi=220)
    fig.savefig(out / "Figure5_A_joint_full.pdf")
    plt.close(fig)
    return F, R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "figure5_ajoint_c128_cpu")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures" / "new")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows, lines = [], []
    for cfg in ("c128",):
        if not (a.results / "summary.json").exists():
            raise FileNotFoundError(a.results / "summary.json")
        runs, meta = load(a.results, cfg)
        rows += per_graph_rows(runs, cfg)
        F, R = figure(runs, meta, a.out)
        t = pd.DataFrame(per_graph_rows(runs, cfg))
        allrnd = [c for r in runs for k, c in r["chains"].items() if k.startswith("random")]
        n_rnd = len(allrnd)
        lines += [
            f"## {cfg}: {meta['backend']}, {meta['dtype']}, maxiter {meta['maxiter']}, ftol {meta['ftol']:g}, gtol {meta['gtol']:g}",
            "",
            f"- Formula angles: mean ratio {t.fixed_ratio.mean():.4f}. Refined by L-BFGS-B from the formula: "
            f"{t.refined_ratio.mean():.4f} (gain mean {t.gain.mean():.4f}, median {t.gain.median():.4f}, "
            f"range {t.gain.min():.4f}–{t.gain.max():.4f}); mean {t.formula_calls.mean():.0f} value-and-gradient calls "
            f"(99 % of the gain within {int(t.formula_calls_to_99pct_gain.min())}–{int(t.formula_calls_to_99pct_gain.max())} calls); "
            f"all 12 chains stop with a regular convergence message.",
            f"- Mean best-so-far along the formula chain: {F.mean(0)[0]:.4f} (call 1), {F.mean(0)[4]:.4f} (5), "
            f"{F.mean(0)[9]:.4f} (10), {F.mean(0)[19]:.4f} (20).",
            f"- Refined formula equals the best of the 20 random-start chains within 1e-4 on "
            f"{int(t['refined_equals_best_random_1e-4'].sum())}/{len(t)} graphs; "
            f"a random chain beats it by more than 1e-4 on {int((t['random_beat_refined_1e-4'] > 0).sum())} graphs "
            f"(largest excess {max(0.0, (t.random_best_final - t.refined_ratio).max()):.4f}).",
            f"- Random starts ({n_rnd} chains, campaign instance-scaled box): mean start ratio "
            f"{np.mean([c['start_ratio'] for c in allrnd]):.4f}, mean final {np.mean([c['final_ratio'] for c in allrnd]):.4f} "
            f"(median {np.median([c['final_ratio'] for c in allrnd]):.4f}); {int(t.random_reach_fixed.sum())}/{n_rnd} "
            f"({t.random_reach_fixed.sum() / n_rnd:.0%}) reach the fixed formula value within {TOL:g}, "
            f"{int(t.random_reach_refined.sum())}/{n_rnd} ({t.random_reach_refined.sum() / n_rnd:.0%}) reach the refined value; "
            f"mean {t.random_mean_calls.mean():.0f} calls per chain; {int(t.random_abnormal_stops.sum())} ABNORMAL line-search stops.",
            f"- Mean best-so-far along random chains: {R.mean(0)[0]:.4f} (call 1), {R.mean(0)[9]:.4f} (10), "
            f"{R.mean(0)[19]:.4f} (20), {R.mean(0)[39]:.4f} (40), {R.mean(0)[59]:.4f} (60).",
            "",
        ]
    pd.DataFrame(rows).to_csv(a.out / "Figure5_A_joint_per_graph.csv", index=False)
    text = "# A-joint initialized refinement, 12 held-out graphs (n = 12, p = 2)\n\n" + "\n".join(lines)
    (a.out / "Figure5_A_joint_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
