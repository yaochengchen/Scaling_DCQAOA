"""Compact side-by-side Fig. 2 from archived, unchanged graph-level results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_quality_depth_and_size_unified import ROOT, SERIES
from plot_pooled_depth_unified import SOURCE, summarize_pooled_depths
from plot_p2_by_n_unified import summarize_p2_by_n

OUTPUT = ROOT / "paper/figures/fig02_quality_depth_and_size_unified_v3"


def build_figure():
    depth_rows = summarize_pooled_depths(SOURCE)
    size_rows = summarize_p2_by_n(SOURCE)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True,
                             layout="constrained")
    for ax, rows, x_key, connected in (
        (axes[0], depth_rows, "p", True),
        (axes[1], size_rows, "n", False),
    ):
        xs = np.array([row[x_key] for row in rows])
        for key, label, color, marker in SERIES:
            means = np.array([row[f"{key}_mean"] for row in rows])
            lower = np.array([row[f"{key}_ci_lo"] for row in rows])
            upper = np.array([row[f"{key}_ci_hi"] for row in rows])
            ax.errorbar(xs, means, yerr=np.vstack((means-lower, upper-means)),
                        fmt=marker, linestyle="-" if connected else "none",
                        color=color, label=label, linewidth=1.6, markersize=5,
                        elinewidth=1, capsize=2.5, zorder=3)
        ax.set_xticks(xs)
        ax.set_ylim(.75, .93)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_title("(a) By depth: 7,000 graphs/point", fontsize=9, loc="left")
    axes[0].set_xlabel("Layers, $p$", fontsize=9)
    axes[0].set_ylabel("Mean MaxCut approximation ratio", fontsize=9)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.3)
    axes[1].set_title("(b) By size: $p=2$, 1,000 graphs/point", fontsize=9, loc="left")
    axes[1].set_xlabel("Qubits, $n$", fontsize=9)
    return fig


def draw(destination: Path = OUTPUT) -> None:
    fig = build_figure()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
