"""Render manuscript-style success-probability figures from the frozen campaign.

The reviewed three-method figures and the manuscript are left untouched.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "pennylane_probability_1050"
FIGURES = ROOT / "figures" / "pennylane_probability_1050_manuscript_style"
MANUSCRIPT_FIGURES = ROOT / "manuscript" / "paper" / "figures"
DEPTHS = (1, 2, 3, 4)
SIZES = (10, 12, 14, 16, 18, 20, 22)
FAMILIES = ("tree", "ring", "regular3", "regular4", "regular5", "regular6",
            "er3", "er05", "community", "bipartite")
WEIGHTS = ("bimodal", "heavy_tail", "lognormal", "uniform", "unit")
BLUE = "#2476cf"
ORANGE = "#e87837"
GREEN = "#1baf7a"


def load_data():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))
    with (RESULTS / "paired_probabilities_1050.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert verification["records"] == len(rows) == 4200
    assert verification["graphs"] == len({r["id"] for r in rows}) == 1050
    assert verification["campaign"] == summary["campaign"]
    assert {r["backend"] for r in rows} == {"pennylane.lightning.qubit"}
    assert len({(r["id"], r["p"]) for r in rows}) == 4200
    assert {int(r["p"]) for r in rows} == set(DEPTHS)
    assert {int(r["n"]) for r in rows} == set(SIZES)
    assert {r["family"] for r in rows} == set(FAMILIES)
    assert {r["weight_kind"] for r in rows} == set(WEIGHTS)
    for field in ("probability_u1", "probability_qaoa"):
        values = np.array([float(r[field]) for r in rows])
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
    for p in DEPTHS:
        group = [r for r in rows if int(r["p"]) == p]
        assert len(group) == summary["depth"][str(p)]["count"] == 1050
        for index, name in ((0, "probability_u1"), (2, "probability_qaoa")):
            assert np.isclose(np.mean([float(r[name]) for r in group]),
                              summary["depth"][str(p)]["mean"][index], atol=1e-12)
    for n in SIZES:
        group = [r for r in rows if int(r["n"]) == n and int(r["p"]) == 2]
        assert len(group) == summary["size_p2"][str(n)]["count"] == 150
    return summary, rows


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def probability_panel(ax, xs, entries, connected):
    for index, label, color, marker in (
        (2, "Fixed-angle QAOA", ORANGE, "s"),
        (0, "Power-law DC-QAOA", BLUE, "o"),
    ):
        means = np.array([e["mean"][index] for e in entries]) * 100
        intervals = np.array([e["ci95"][index] for e in entries]) * 100
        errors = np.maximum(0, np.array([means - intervals[:, 0], intervals[:, 1] - means]))
        ax.errorbar(xs, means, yerr=errors, fmt=marker,
                    linestyle="-" if connected else "none", color=color,
                    label=label, linewidth=1.6, markersize=5,
                    elinewidth=1, capsize=2.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_ylim(0, 25)
    ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def figure3(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True,
                             layout="constrained")
    probability_panel(axes[0], DEPTHS,
                      [summary["depth"][str(p)] for p in DEPTHS], True)
    probability_panel(axes[1], SIZES,
                      [summary["size_p2"][str(n)] for n in SIZES], False)
    axes[0].set_title("(a) By depth: 1,050 graphs/point", fontsize=9, loc="left")
    axes[0].set_xlabel("Layers, $p$", fontsize=9)
    axes[0].set_ylabel("Mean optimal-solution probability (%)", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.3)
    axes[1].set_title("(b) By size: $p=2$, 150 graphs/point", fontsize=9, loc="left")
    axes[1].set_xlabel("Qubits, $n$", fontsize=9)
    save(fig, "Figure3_powerlaw_vs_qaoa_probability_1050")


def family_matrices(rows):
    matrices = []
    for xkey, xs in (("p", DEPTHS), ("n", SIZES)):
        matrix = np.zeros((len(FAMILIES), len(xs)))
        for i, family in enumerate(FAMILIES):
            for j, x in enumerate(xs):
                selected = [r for r in rows if r["family"] == family
                            and int(r[xkey]) == x and (xkey == "p" or int(r["p"]) == 2)]
                assert len(selected) == (105 if xkey == "p" else 15)
                matrix[i, j] = np.mean([float(r["probability_u1"])
                                        - float(r["probability_qaoa"]) for r in selected]) * 100
        matrices.append(matrix)
    return matrices


def figure6(rows):
    matrices = family_matrices(rows)
    scale = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), sharey=True)
    for ax, matrix, xs, title, xlabel in zip(
        axes, matrices, (DEPTHS, SIZES),
        ("(a) Depth dependence · pooled over $n=10$–$22$",
         "(b) Size dependence · $p=2$"),
        ("Layers $p$", "Qubits $n$"),
    ):
        image = ax.imshow(matrix, cmap="RdBu", vmin=-scale, vmax=scale,
                          aspect="auto")
        ax.set_yticks(range(len(FAMILIES)), FAMILIES)
        ax.set_xticks(range(len(xs)), xs)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(np.arange(-.5, len(xs), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(FAMILIES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(FAMILIES)):
            for j in range(len(xs)):
                value = matrix[i, j]
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center",
                        fontsize=7.4,
                        color="white" if abs(value) > .6 * scale else "#202020")
    fig.subplots_adjust(left=.13, right=.87, bottom=.13, top=.9, wspace=.2)
    colorbar_axis = fig.add_axes([.9, .18, .018, .64])
    fig.colorbar(image, cax=colorbar_axis,
                 label="Mean paired probability difference (percentage points)")
    save(fig, "Figure6_powerlaw_minus_qaoa_probability_1050")
    return matrices


def figure7(summary, rows):
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    xs = np.arange(len(WEIGHTS))
    for offset, p, color, hatch in ((-.18, 1, BLUE, ""),
                                    (.18, 4, GREEN, "///")):
        entries = [summary["weights"][f"{weight}_p{p}"] for weight in WEIGHTS]
        means = np.array([entry["delta_new_old_new_qaoa"][1] for entry in entries]) * 100
        ci = np.array([entry["delta_ci95"][1] for entry in entries]) * 100
        errors = np.maximum(0, np.array([means - ci[:, 0], ci[:, 1] - means]))
        for j, weight in enumerate(WEIGHTS):
            group = [r for r in rows if int(r["p"]) == p and r["weight_kind"] == weight]
            assert len(group) == entries[j]["count"] == 210
            observed = np.mean([float(r["probability_u1"])
                                - float(r["probability_qaoa"]) for r in group]) * 100
            assert np.isclose(observed, means[j], atol=1e-12)
        ax.bar(xs + offset, means, width=.34, label=f"$p={p}$",
               color=color, edgecolor="#303030", hatch=hatch,
               yerr=errors, capsize=3, linewidth=.75,
               error_kw={"ecolor": "#303030", "elinewidth": 1})
    ax.axhline(0, color="#444444", linewidth=.9)
    ax.set_xticks(xs, ("Bimodal", "Heavy tail", "Lognormal", "Uniform", "Unit"))
    ax.set_ylabel("Mean paired probability difference (percentage points)")
    ax.set_title("Same-layer gain by edge-weight distribution", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    fig.text(.5, .01,
             "Each bar: 210 test graphs ($n=10$–$22$); 95% graph bootstrap CI.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "Figure7_powerlaw_minus_qaoa_probability_1050")


def main():
    summary, rows = load_data()
    figure3(summary)
    matrices = figure6(rows)
    figure7(summary, rows)
    manuscript_names = (
        ("Figure3_powerlaw_vs_qaoa_probability_1050", "fig03_powerlaw_success_probability_1050"),
        ("Figure6_powerlaw_minus_qaoa_probability_1050", "fig06_powerlaw_family_success_gain_1050"),
        ("Figure7_powerlaw_minus_qaoa_probability_1050", "fig07_powerlaw_weight_success_gain_1050"),
    )
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    for source, target in manuscript_names:
        for suffix in ("pdf", "png"):
            shutil.copy2(FIGURES / f"{source}.{suffix}",
                         MANUSCRIPT_FIGURES / f"{target}.{suffix}")
    print(f"Rendered three manuscript-style figures from {len(rows):,} verified records")
    print(f"Family-cell difference range: {min(float(m.min()) for m in matrices):.3f} to "
          f"{max(float(m.max()) for m in matrices):.3f} percentage points")
    print(FIGURES)
    print(MANUSCRIPT_FIGURES)


if __name__ == "__main__":
    main()
