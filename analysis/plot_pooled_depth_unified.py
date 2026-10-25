"""Plot the p=1–4 comparison on the same 7,000 archived test graphs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis" / "outputs_unified" / "tables" / "T1_paired_records_unified.csv"
DESTINATION = ROOT / "paper" / "figures" / "fig02_same_depth_pooled_unified"
DEPTHS = (1, 2, 3, 4)


def summarize_pooled_depths(source: Path, expected_graphs: int = 7000) -> list[dict]:
    """Check a common graph set and compute pointwise graph-bootstrap intervals."""
    groups = {p: {} for p in DEPTHS}
    with Path(source).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            p = int(row["p"])
            if p not in groups:
                continue
            graph_id = row["id"]
            if graph_id in groups[p]:
                raise ValueError(f"Duplicate graph {graph_id} at p={p}")
            groups[p][graph_id] = (float(row["ratio_qaoa_fixed"]),
                                   float(row["ratio_cd_unified"]))
    graph_ids = set(groups[1])
    if any(len(groups[p]) != expected_graphs or set(groups[p]) != graph_ids for p in DEPTHS):
        raise ValueError(f"Expected the same {expected_graphs} graphs at p=1–4")
    ordered_ids = sorted(graph_ids)
    values = np.asarray([[groups[p][graph_id] for p in DEPTHS]
                         for graph_id in ordered_ids], dtype=float)
    if not np.isfinite(values).all() or np.any((values < -1e-8) | (values > 1 + 1e-8)):
        raise ValueError("Invalid approximation ratio")
    means = values.mean(axis=0)
    rng = np.random.default_rng(1441)
    boot_means = np.empty((2000, len(DEPTHS), 2))
    for draw in range(len(boot_means)):
        boot_means[draw] = values[rng.integers(expected_graphs, size=expected_graphs)].mean(axis=0)
    lows, highs = np.quantile(boot_means, [0.025, 0.975], axis=0)
    return [{"p": p, "count": expected_graphs,
             "qaoa_mean": float(means[i, 0]), "qaoa_ci_lo": float(lows[i, 0]),
             "qaoa_ci_hi": float(highs[i, 0]), "dc_mean": float(means[i, 1]),
             "dc_ci_lo": float(lows[i, 1]), "dc_ci_hi": float(highs[i, 1])}
            for i, p in enumerate(DEPTHS)]


if __name__ == "__main__":
    rows = summarize_pooled_depths(SOURCE)
    ps = np.asarray(DEPTHS)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for key, low, high, label, color, marker in (
        ("qaoa_mean", "qaoa_ci_lo", "qaoa_ci_hi", "Fixed-angle QAOA", "#e87837", "s"),
        ("dc_mean", "dc_ci_lo", "dc_ci_hi", "Formula-based DC-QAOA", "#2476cf", "o"),
    ):
        means = np.asarray([row[key] for row in rows])
        lower = np.asarray([row[low] for row in rows])
        upper = np.asarray([row[high] for row in rows])
        ax.errorbar(ps, means, yerr=np.vstack((means - lower, upper - means)),
                    fmt=marker + "-", linewidth=1.9, markersize=6, capsize=3,
                    elinewidth=1.1, color=color, label=label)
    ax.set_xticks(ps)
    ax.set_xlabel("Layers, p")
    ax.set_ylabel("Mean MaxCut approximation ratio")
    ax.set_title("Same-layer comparison across 7,000 held-out graphs", loc="left")
    ax.set_ylim(0.75, 0.93)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.12, 0.015,
             "Each point uses the same 7,000 graphs (n=10–22); bars: 95% bootstrap CI.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(DESTINATION.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    for row in rows:
        print(f"p={row['p']}: QAOA={row['qaoa_mean']:.6f}, "
              f"DC-QAOA={row['dc_mean']:.6f}, graphs={row['count']}")
    print(DESTINATION.with_suffix(".png"))
    print(DESTINATION.with_suffix(".pdf"))
