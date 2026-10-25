"""Three-panel comparison from saved per-call optimizer summaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("L-BFGS-B", "COBYLA", "Adam")
COLORS = {"L-BFGS-B": "#6750a4", "COBYLA": "#21867a", "Adam": "#cf6b43"}


def attainment_ylim(summary: dict) -> tuple[float, float]:
    """Keep zero visible while making rare successes legible."""
    maximum = max(sum(hit is not None for hit in item["run_hits"]) / item["runs"]
                  for item in summary.values())
    return (0, min(1., max(.05, 1.5 * maximum)))


def plot_comparison(summary: dict, destination: Path) -> None:
    if set(summary) != set(METHODS):
        raise ValueError("Expected exactly three optimizer summaries")
    budget = summary[METHODS[0]]["budget_evaluations"]
    if any(summary[method]["budget_evaluations"] != budget or
           summary[method]["runs"] != summary[METHODS[0]]["runs"] or
           not np.isclose(summary[method]["mean_fixed_ratio"],
                          summary[METHODS[0]]["mean_fixed_ratio"], atol=1e-10)
           for method in METHODS):
        raise ValueError("Incompatible optimizer cohorts")

    fig, axes = plt.subplots(1, 3, figsize=(15.1, 4.7), layout="constrained")
    first, second, third = axes
    x = np.arange(1, budget + 1)
    target = summary[METHODS[0]]["mean_fixed_ratio"]
    first.axhline(target, color="#e0a11d", lw=2.5, label="Formula-based DC-QAOA")
    for method in METHODS:
        item = summary[method]
        first.fill_between(x, item["p10_best_curve"], item["p90_best_curve"],
                           color=COLORS[method], alpha=.13, linewidth=0)
        first.plot(x, item["mean_best_curve"], color=COLORS[method], lw=2.1,
                   label=f"{method} mean best-so-far")
    first.set(title="(a) Mean and 10–90% trajectory range",
              xlabel="Target-instance objective evaluations",
              ylabel="Mean MaxCut approximation ratio", xlim=(0, budget))
    first.legend(frameon=False, fontsize=8, loc="lower right")

    timeline = np.arange(budget + 1)
    for method in METHODS:
        item = summary[method]
        hits = item["run_hits"]
        cumulative = [sum(hit is not None and hit <= count for hit in hits) / len(hits)
                      for count in timeline]
        second.step(timeline, cumulative, where="post", color=COLORS[method], lw=2,
                    label=f"{method}: {sum(hit is not None for hit in hits)}/{len(hits)}")
    second.set(title="(b) Cumulative target attainment",
               xlabel="Target-instance objective evaluations",
               ylabel="Fraction of all runs reaching target (zoomed)",
               xlim=(0, budget), ylim=attainment_ylim(summary))
    second.legend(frameon=False, fontsize=8, loc="upper left")

    third.axvline(budget, color="#777777", lw=1, ls="--")
    rng = np.random.default_rng(1441)
    for index, method in enumerate(METHODS):
        item = summary[method]
        hits = [hit for hit in item["run_hits"] if hit is not None]
        if hits:
            third.scatter(hits, index + rng.uniform(-.16, .16, len(hits)),
                          s=21, color=COLORS[method], alpha=.72, zorder=3)
        censored = item["runs"] - len(hits)
        if censored:
            third.scatter([budget + 8], [index], marker=">", facecolors="none",
                          edgecolors=COLORS[method], s=55, zorder=3)
            third.text(budget + 14, index, f"{censored}/{item['runs']} >{budget}",
                       color=COLORS[method], fontsize=8, va="center")
    third.set_yticks(range(len(METHODS)), METHODS)
    third.invert_yaxis()
    third.set(title="(c) First attainment or not reached",
              xlabel="Actual evaluations to fixed target", xlim=(0, budget + 83))
    third.text(.02, -.20, "Open triangle: not reached within budget",
               transform=third.transAxes, fontsize=8, color="#666666")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Held-out MaxCut, n=12, p=2: optimizer comparison against one frozen GPU table",
                 fontsize=13)
    graph_count = summary[METHODS[0]]["graphs"]
    run_count = summary[METHODS[0]]["runs"]
    if run_count % graph_count:
        raise ValueError("Run count is not divisible by graph count")
    fig.text(.5, -.025,
             f"{graph_count} graphs × {run_count // graph_count} shared random starts = "
             f"{run_count} runs per method; "
             f"max {budget} actual evaluations/run; target tolerance 0.001.",
             ha="center", fontsize=8, color="#555555")
    fig.text(.5, -.065,
             "Shading in (a): 10th–90th percentile of individual best-so-far trajectories; "
             "ended runs retain their last value.",
             ha="center", fontsize=8, color="#555555")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
