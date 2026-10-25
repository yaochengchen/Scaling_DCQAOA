"""Plot p=2 size dependence from existing frozen unified-table test records."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis" / "outputs_unified" / "tables" / "T1_paired_records_unified.csv"
DESTINATION = ROOT / "paper" / "figures" / "fig06_p2_by_n_unified"
SIZES = (10, 14, 18, 22)


def summarize_p2_by_n(source: Path, expected_per_n: int = 1000) -> list[dict]:
    """Return graph-level means and pointwise paired-bootstrap intervals."""
    groups = defaultdict(list)
    with Path(source).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["p"]) == 2 and int(row["n"]) in SIZES:
                groups[int(row["n"])].append((float(row["ratio_qaoa_fixed"]),
                                              float(row["ratio_cd_unified"])))
    output = []
    if set(groups) != set(SIZES):
        raise ValueError(f"Expected {expected_per_n} p=2 graphs at each n in {SIZES}")
    for n in SIZES:
        values = np.asarray(groups[n], dtype=float)
        if len(values) != expected_per_n or not np.isfinite(values).all():
            raise ValueError(f"Expected {expected_per_n} p=2 graphs at n={n}")
        rng = np.random.default_rng(1441 + n)
        indices = rng.integers(len(values), size=(2000, len(values)))
        boot_means = values[indices].mean(axis=1)
        low, high = np.quantile(boot_means, [0.025, 0.975], axis=0)
        means = values.mean(axis=0)
        output.append({"n": n, "count": len(values),
                       "qaoa_mean": float(means[0]), "qaoa_ci_lo": float(low[0]),
                       "qaoa_ci_hi": float(high[0]), "dc_mean": float(means[1]),
                       "dc_ci_lo": float(low[1]), "dc_ci_hi": float(high[1])})
    return output


if __name__ == "__main__":
    summary = summarize_p2_by_n(SOURCE)
    ns = np.array([row["n"] for row in summary])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for mean_key, low_key, high_key, label, color, marker in (
        ("qaoa_mean", "qaoa_ci_lo", "qaoa_ci_hi", "Fixed-angle QAOA", "#e87837", "s"),
        ("dc_mean", "dc_ci_lo", "dc_ci_hi", "Formula-based DC-QAOA", "#2476cf", "o"),
    ):
        means = np.array([row[mean_key] for row in summary])
        lows = np.array([row[low_key] for row in summary])
        highs = np.array([row[high_key] for row in summary])
        ax.errorbar(ns, means, yerr=np.vstack((means - lows, highs - means)),
                    fmt=marker, linestyle="none", color=color, markersize=5.5,
                    capsize=3, elinewidth=1.1, label=label, zorder=3)
    ax.set_xticks(ns)
    ax.set_xlabel("Number of qubits, n")
    ax.set_ylabel("Mean MaxCut approximation ratio")
    ax.set_title("Same-layer comparison at p=2", loc="left")
    ax.set_ylim(0.81, 0.90)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.text(0.12, 0.015, "Each point summarizes 1,000 test graphs; bars: 95% bootstrap CI.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(DESTINATION.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(DESTINATION.with_suffix(".png"))
    print(DESTINATION.with_suffix(".pdf"))
