"""Versioned Fig. 5 redraw from existing paired records; no circuit simulation."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_unified_table import _bootstrap_ci, _style, ROOT

PAIRS = ROOT / "analysis/outputs_unified/tables/T1_paired_records_unified.csv"
OUTPUT = ROOT / "paper/figures/fig05_weight_distributions_unified_v2"
KINDS = ("bimodal", "heavy_tail", "lognormal", "uniform", "unit")


def weight_style(p: int) -> dict:
    if p == 1:
        return {"color": "#2a78d6", "edgecolor": "#303030", "hatch": ""}
    if p == 4:
        return {"color": "#1baf7a", "edgecolor": "#303030", "hatch": "///"}
    raise ValueError("Only p=1 and p=4 appear in Fig. 5")


def plot(records: list[dict], destination: Path) -> None:
    groups = defaultdict(list)
    for row in records:
        p = int(row["p"])
        if p in (1, 4):
            groups[(row["weight_kind"], p)].append(float(row["gain_fixed_norm"]))
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    xs = np.arange(len(KINDS))
    for offset, p in ((-.18, 1), (.18, 4)):
        means, lower, upper = [], [], []
        for kind in KINDS:
            values = groups[(kind, p)]
            if len(values) != 1400:
                raise ValueError(f"Expected 1,400 graphs for {kind}, p={p}")
            mean = float(np.mean(values))
            lo, hi = _bootstrap_ci(values)
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        ax.bar(xs + offset, means, width=.34, label=f"p={p}",
               yerr=np.array([lower, upper]), capsize=3,
               linewidth=.75, error_kw={"ecolor": "#303030", "elinewidth": 1},
               **weight_style(p))
    ax.axhline(0, color="#444444", linewidth=.9)
    ax.set_xticks(xs, ("Bimodal", "Heavy tail", "Lognormal", "Uniform", "Unit"))
    _style(ax, y="Mean paired energy gain / ΣJ",
           title="Same-layer gain by edge-weight distribution")
    ax.legend(frameon=False, ncol=2)
    fig.text(.5, .01, "Each bar: 1,400 held-out graphs (n=10–22); 95% graph bootstrap CI.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    with PAIRS.open(encoding="utf-8", newline="") as handle:
        plot(list(csv.DictReader(handle)), OUTPUT)
