"""Replot saved n=16,p=2 unified-table trajectories as honest panels (a)–(c)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unified_variational_n16p2 import OUT, FIGURE, TOLERANCE, grouped_attainment, summarize_runs

DESTINATION = FIGURE.with_name(FIGURE.name + "_three_panel")
GROUP_LABELS = {("regular3", "unit"): "3-regular / unit",
                ("er3", "unit"): "ER3 / unit",
                ("regular3", "uniform"): "3-regular / uniform",
                ("er3", "uniform"): "ER3 / uniform"}
COLORS = ("#2d728f", "#bb6a39", "#63935e", "#8b64a8")


def panel_titles() -> tuple[str, str, str]:
    return ("(a) Objective-evaluation trajectories",
            "(b) Cumulative target attainment",
            "(c) Time to fixed target by graph group")


def figure_title(traces: list[dict]) -> str:
    sizes = {(trace["n"], trace["p"]) for trace in traces}
    if len(sizes) != 1:
        raise ValueError("Mixed n or p in one figure")
    n, p = next(iter(sizes))
    return f"Unified GPU-angle table vs random-start variational DC-QAOA, n={n}, p={p}"


def validate_saved_summary(saved: dict, fresh: dict) -> bool:
    """A trace reorder can change hit-list order and tiny floating sums."""
    exact = ("runs", "graphs", "budget_evaluations", "tolerance",
             "mean_trajectory_hit", "errors")
    if any(saved[key] != fresh[key] for key in exact):
        return False
    if sorted((-1 if hit is None else hit) for hit in saved["run_hits"]) != sorted(
        (-1 if hit is None else hit) for hit in fresh["run_hits"]
    ):
        return False
    numeric = ("mean_fixed_ratio", "run_hit_fraction", "mean_last_best_ratio")
    if any(not np.isclose(saved[key], fresh[key], atol=1e-12) for key in numeric):
        return False
    return bool(np.allclose(saved["mean_best_curve"], fresh["mean_best_curve"], atol=1e-12))


def plot_three_panel(traces: list[dict], summary: dict, destination: Path) -> None:
    budget = summary["budget_evaluations"]
    groups = grouped_attainment(traces, budget)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), layout="constrained")
    ax_a, ax_c, ax_b = axes
    title_a, title_c, title_b = panel_titles()

    # (a) Every horizontal coordinate is an actual target-instance objective call.
    for trace in traces:
        values = [event["best_ratio"] for event in trace["evaluations"]]
        ax_a.plot(range(1, len(values) + 1), values, color="#a79bc9", alpha=.15, lw=.7)
    ax_a.plot(range(1, budget + 1), summary["mean_best_curve"],
              color="#6651a7", lw=2.5, label="Variational mean best-so-far")
    ax_a.axhline(summary["mean_fixed_ratio"], color="#e0a11d", lw=2.5,
                 label="Formula-based fixed target")
    ax_a.set(title=title_a, xlabel="Evaluations on target graph",
             ylabel="MaxCut approximation ratio", xlim=(0, budget))
    ax_a.legend(loc="lower right", fontsize=8, frameon=False)

    # (c) A dot records a first hit; a hollow triangle beyond the budget marks
    # right-censored runs, never a false attainment at the budget boundary.
    ax_b.axvline(budget, color="#777777", ls="--", lw=1)
    for index, group in enumerate(groups):
        hits = [hit for hit in group["hits"] if hit is not None]
        if hits:
            offsets = np.linspace(-.12, .12, len(hits))
            ax_b.scatter(hits, index + offsets, marker="o", color=COLORS[index],
                         s=36, zorder=3)
        if group["not_reached"]:
            ax_b.scatter([budget + 7], [index], marker=">", facecolors="none",
                         edgecolors=COLORS[index], s=55, zorder=3)
            ax_b.text(budget + 11, index, f"{group['not_reached']}/{group['runs']} >{budget}",
                      va="center", fontsize=8, color=COLORS[index])
    ax_b.set_yticks(range(len(groups)),
                    [GROUP_LABELS[(group["family"], group["weight_kind"])] for group in groups])
    ax_b.set(xlim=(0, budget + 58), xlabel="First attainment: actual evaluations",
             title=title_b)
    ax_b.invert_yaxis()
    ax_b.text(.02, -.20, "Open triangle: not reached within budget", transform=ax_b.transAxes,
              fontsize=8, color="#666666")

    # (b) Denominator includes every run, also those censored at the budget.
    x = np.arange(budget + 1)
    for index, group in enumerate(groups):
        attained = group["runs"] - group["not_reached"]
        label = GROUP_LABELS[(group["family"], group["weight_kind"])]
        ax_c.step(x, group["cumulative"], where="post", color=COLORS[index], lw=1.8,
                  label=f"{label}: {attained}/{group['runs']}")
    pooled = [sum(hit is not None and hit <= count for hit in summary["run_hits"])
              / summary["runs"] for count in x]
    ax_c.step(x, pooled, where="post", color="#333333", lw=2.2,
              label=f"All: {sum(hit is not None for hit in summary['run_hits'])}/{summary['runs']}")
    ax_c.set(title=title_c, xlabel="Evaluations on target graph",
             ylabel="Fraction of all runs reaching target", xlim=(0, budget),
             ylim=(0, max(.15, 1.12 * max(max(group["cumulative"]) for group in groups))))
    ax_c.legend(loc="upper left", fontsize=7.5, frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(figure_title(traces), fontsize=13)
    fig.text(.5, -.025,
             f"{summary['graphs']} held-out graphs, {summary['runs']} starts; L-BFGS-B; "
             f"max {budget} evaluations/run; attainment within {TOLERANCE:g} of each fixed value. "
             f"{sum(hit is not None for hit in summary['run_hits'])}/{summary['runs']} reached target.",
             ha="center", fontsize=8, color="#555555")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    traces = [json.loads(path.read_text(encoding="utf-8"))
              for path in sorted((OUT / "runs").glob("*.json"))]
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if len(traces) != 36 or summary["runs"] != len(traces) or summary["graphs"] != 12:
        raise ValueError("Expected the complete 12-graph, 36-run saved experiment")
    fresh = summarize_runs(traces, summary["budget_evaluations"])
    if not validate_saved_summary(summary, fresh):
        raise ValueError("Saved summary does not match saved trajectories")
    plot_three_panel(traces, fresh, DESTINATION)
    print(DESTINATION.with_suffix(".png"))
    print(DESTINATION.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
