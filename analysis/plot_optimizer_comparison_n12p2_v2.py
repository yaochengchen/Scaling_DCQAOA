"""Versioned, print-readable rendering of the saved n=12, p=2 traces.

No optimizer or circuit is run here. The original plot and script are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_optimizer_comparison_n12p2 import COLORS, METHODS, attainment_ylim

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/unified_variational_n12p2_optimizer_compare_pennylane_starts20_eval500/summary.json"
OUTPUT = ROOT / "paper/figures/fixed_vs_variational_unified_n12p2_optimizers_pennylane_starts20_eval500_v2"


def nonattainment_label(method: str, item: dict) -> str:
    reached = sum(hit is not None for hit in item["run_hits"])
    return f"{method}: {reached} attained / {item['runs'] - reached} not attained"


def plot(summary: dict, destination: Path) -> None:
    if set(summary) != set(METHODS):
        raise ValueError("Expected exactly three optimizers")
    budget = summary[METHODS[0]]["budget_evaluations"]
    if any(summary[m]["budget_evaluations"] != budget or summary[m]["runs"] != 240
           for m in METHODS):
        raise ValueError("Incompatible run counts or evaluation budgets")

    fig = plt.figure(figsize=(11.2, 8.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, .83), wspace=.13, hspace=.13)
    first = fig.add_subplot(grid[0, :])
    second = fig.add_subplot(grid[1, 0])
    third = fig.add_subplot(grid[1, 1])
    x = np.arange(1, budget + 1)
    first.axhline(summary[METHODS[0]]["mean_fixed_ratio"], color="#dd9b12", lw=2.8,
                  label="Formula-based DC-QAOA mean")
    for method in METHODS:
        item = summary[method]
        first.fill_between(x, item["p10_best_curve"], item["p90_best_curve"],
                           color=COLORS[method], alpha=.13, linewidth=0)
        first.plot(x, item["mean_best_curve"], color=COLORS[method], lw=2.1,
                   label=f"{method} mean best-so-far")
    first.set(title="(a) Mean best-so-far trajectory and 10th–90th percentile range",
              xlabel="Target-instance objective evaluations",
              ylabel="Mean MaxCut approximation ratio", xlim=(0, budget))
    first.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)

    timeline = np.arange(budget + 1)
    for method in METHODS:
        item = summary[method]
        hits = item["run_hits"]
        cumulative = np.array([sum(hit is not None and hit <= k for hit in hits)
                               for k in timeline]) / item["runs"]
        second.step(timeline, cumulative, where="post", color=COLORS[method], lw=2.1,
                    label=f"{method}: {sum(hit is not None for hit in hits)}/{item['runs']}")
    second.set(title="(b) Cumulative target attainment",
               xlabel="Target-instance objective evaluations",
               ylabel="Fraction of all 240 runs (zoomed)",
               xlim=(0, budget), ylim=attainment_ylim(summary))
    second.legend(frameon=False, fontsize=9, loc="upper left")

    rng = np.random.default_rng(1441)
    for index, method in enumerate(METHODS):
        hits = [hit for hit in summary[method]["run_hits"] if hit is not None]
        third.scatter(hits, index + rng.uniform(-.12, .12, len(hits)),
                      s=39, color=COLORS[method], alpha=.83, zorder=3)
    third.set_yticks(range(len(METHODS)),
                     [nonattainment_label(m, summary[m]) for m in METHODS], fontsize=8.5)
    third.invert_yaxis()
    third.set(title="(c) First attainment among successful runs",
              xlabel="Actual evaluations at first attainment", xlim=(0, budget))

    for axis in (first, second, third):
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.title.set_fontsize(11)
    fig.suptitle("Held-out MaxCut, n=12, p=2: optimizer comparison against one frozen GPU table",
                 fontsize=13)
    fig.supxlabel("12 graphs × 20 shared starts = 240 runs per method; at most 500 actual evaluations per run.\n"
                  "Nonattainment counts include runs stopped before 500; target tolerance 0.001.",
                  fontsize=9, color="#555555")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot(json.loads(SUMMARY.read_text(encoding="utf-8")), OUTPUT)
