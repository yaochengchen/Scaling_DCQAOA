"""Plot held-out equal-layer quality by depth and a selected p=2 size slice."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_pooled_depth_unified import SOURCE, summarize_pooled_depths
from plot_p2_by_n_unified import summarize_p2_by_n


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "paper" / "figures" / "fig02_quality_depth_and_size_unified"
SERIES = (
    ("qaoa", "Fixed-angle QAOA", "#e87837", "s"),
    ("dc", "Formula-based DC-QAOA", "#2476cf", "o"),
)


def draw() -> None:
    depth_rows = summarize_pooled_depths(SOURCE)
    size_rows = summarize_p2_by_n(SOURCE)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharey=True)

    for ax, rows, x_key, connected in (
        (axes[0], depth_rows, "p", True),
        (axes[1], size_rows, "n", False),
    ):
        xs = np.array([row[x_key] for row in rows])
        for key, label, color, marker in SERIES:
            means = np.array([row[f"{key}_mean"] for row in rows])
            lower = np.array([row[f"{key}_ci_lo"] for row in rows])
            upper = np.array([row[f"{key}_ci_hi"] for row in rows])
            ax.errorbar(
                xs, means, yerr=np.vstack((means - lower, upper - means)),
                fmt=marker,
                linestyle="-" if connected else "none",
                linewidth=1.8, markersize=6, capsize=3, elinewidth=1.1,
                color=color, label=label, zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_ylim(0.75, 0.93)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("(a) Depth dependence · 7,000 held-out graphs per point", loc="left")
    axes[0].set_xlabel("Layers, p")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_title("(b) Size dependence · p=2, 1,000 graphs per point", loc="left")
    axes[1].set_xlabel("Number of qubits, n")
    fig.supylabel("Mean MaxCut approximation ratio", x=0.02)
    fig.text(
        0.5, 0.01,
        "Both panels: 95% graph-bootstrap CI. Panel (b) shows four selected sizes.",
        ha="center", fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 1), h_pad=2.0)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = DESTINATION.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    draw()
