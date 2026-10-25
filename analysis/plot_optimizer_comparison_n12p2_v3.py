"""Compact main-text Fig. 3(a,b), using the archived optimization summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_optimizer_comparison_n12p2 import COLORS, METHODS, attainment_ylim
from plot_optimizer_comparison_n12p2_v2 import ROOT, SUMMARY

OUTPUT = ROOT / "paper/figures/fixed_vs_variational_unified_n12p2_optimizers_pennylane_starts20_eval500_v3"


def build_figure(summary: dict):
    if set(summary) != set(METHODS):
        raise ValueError("Expected three optimizer summaries")
    budget = summary[METHODS[0]]["budget_evaluations"]
    if any(summary[m]["budget_evaluations"] != budget or summary[m]["runs"] != 240
           for m in METHODS):
        raise ValueError("Incompatible cohorts or budgets")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    first, second = axes
    x = np.arange(1, budget + 1)
    first.axhline(summary[METHODS[0]]["mean_fixed_ratio"], color="#dd9b12",
                  linewidth=2.2, label="Formula-based DC-QAOA")
    for method in METHODS:
        item = summary[method]
        first.fill_between(x, item["p10_best_curve"], item["p90_best_curve"],
                           color=COLORS[method], alpha=.13, linewidth=0)
        first.plot(x, item["mean_best_curve"], color=COLORS[method],
                   linewidth=1.7, label=method)
    first.set(title="(a) Mean best-so-far", xlabel="Objective evaluations",
              ylabel="Mean MaxCut approximation ratio", xlim=(0, budget))
    first.legend(frameon=False, fontsize=7.1, loc="lower right")

    timeline = np.arange(budget + 1)
    for method in METHODS:
        item = summary[method]
        hits = item["run_hits"]
        cumulative = np.array([sum(hit is not None and hit <= k for hit in hits)
                               for k in timeline]) / item["runs"] * 100
        second.step(timeline, cumulative, where="post", color=COLORS[method],
                    linewidth=1.8,
                    label=f"{method}: {sum(hit is not None for hit in hits)}/240")
    second.set(title="(b) Runs attaining own-graph target",
               xlabel="Objective evaluations", ylabel="Runs attaining target (%)",
               xlim=(0, budget), ylim=tuple(100*y for y in attainment_ylim(summary)))
    second.legend(frameon=False, fontsize=7.1, loc="upper left")

    for ax in axes:
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(8.5)
        ax.yaxis.label.set_fontsize(8.5)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.23)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw(summary: dict, destination: Path = OUTPUT) -> None:
    fig = build_figure(summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=300,
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(json.loads(SUMMARY.read_text(encoding="utf-8")))
