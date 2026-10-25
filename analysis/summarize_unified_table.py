"""Recalculate paired held-out statistics for the frozen unified angle table."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_unified_table import EXPECTED_MODEL_SHA256, ROOT, load_archived_inputs

OUTPUT = ROOT / "analysis" / "outputs_unified"
FIGURES = ROOT / "paper" / "figures"
FAMILIES = ("tree", "ring", "regular3", "regular4", "regular5", "regular6",
            "er3", "er05", "community", "bipartite")
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def score_pair(energy: float, historical: dict) -> dict:
    minimum = float(historical["ground_energy"])
    maximum = float(historical["max_energy"])
    qaoa = float(historical["energy_qaoa_best"])
    if maximum <= minimum or not np.isfinite(energy):
        raise ValueError("Invalid extrema or unified energy")
    tolerance = 1e-8 * max(1.0, abs(minimum), abs(maximum))
    if energy < minimum - tolerance or energy > maximum + tolerance:
        raise ValueError(f"Unified energy outside exact spectrum: {energy}, [{minimum}, {maximum}]")
    ratio_cd = (maximum - energy) / (maximum - minimum)
    ratio_qaoa = (maximum - qaoa) / (maximum - minimum)
    result = {"ratio_cd_unified": ratio_cd, "ratio_qaoa_fixed": ratio_qaoa,
              "gain_fixed_norm": (qaoa - energy) / float(historical["norm_weights"]),
              "cd_win": float(energy < qaoa - 1e-12)}
    if "ratio_cd_opt" in historical and "ratio_qaoa_opt" in historical:
        result.update({"transfer_loss_cd": float(historical["ratio_cd_opt"]) - ratio_cd,
                       "transfer_loss_qaoa": float(historical["ratio_qaoa_opt"]) - ratio_qaoa,
                       "ratio_cd_opt": float(historical["ratio_cd_opt"]),
                       "ratio_qaoa_opt": float(historical["ratio_qaoa_opt"])})
    return result


def _bootstrap_ci(values, seed=1441, draws=2000):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = arr[rng.integers(len(arr), size=len(arr))].mean()
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty group")
    gain = np.asarray([r["gain_fixed_norm"] for r in rows], dtype=float)
    lo, hi = _bootstrap_ci(gain)
    out = {"pairs": len(rows), "gain_fixed_norm": float(gain.mean()),
           "gain_ci_lo": lo, "gain_ci_hi": hi}
    for key in ("ratio_cd_unified", "ratio_qaoa_fixed"):
        out[key] = float(np.mean([r[key] for r in rows]))
    out["cd_win"] = float(np.mean([r.get("cd_win", float(
        r["ratio_cd_unified"] > r["ratio_qaoa_fixed"])) for r in rows]))
    for key in ("transfer_loss_cd", "transfer_loss_qaoa", "ratio_cd_opt", "ratio_qaoa_opt"):
        if all(key in r for r in rows):
            out[key] = float(np.mean([r[key] for r in rows]))
    return out


def collect_pairs(root: Path = ROOT) -> list[dict]:
    _, cohorts = load_archived_inputs(root)
    all_pairs = []
    result_dir = root / "results" / "unified_table_heldout"
    for cohort in cohorts:
        for source in cohort["rows"]:
            path = result_dir / cohort["name"] / f"{source['id']}.json"
            if not path.exists():
                raise FileNotFoundError(f"Incomplete unified evaluation: {path}")
            records = json.loads(path.read_text(encoding="utf-8"))
            if len(records) != len(cohort["depths"]):
                raise ValueError(f"Wrong depth count: {path}")
            for record, p in zip(records, cohort["depths"]):
                if (record["id"] != source["id"] or record["p"] != p or
                    record["fingerprint"] != source["fingerprint"] or
                    record["model_sha256"] != EXPECTED_MODEL_SHA256):
                    raise ValueError(f"Record identity or model mismatch: {path}")
                old = cohort["old_results"][(source["id"], p)]
                pair = {"cohort": cohort["name"], "id": source["id"], "p": p,
                        "n": source["n"], "family": source["family"],
                        "weight_kind": source["weight_kind"],
                        "energy_cd_unified": record["energy_cd_unified"],
                        "energy_qaoa_fixed": old["energy_qaoa_best"],
                        "degree_cv": old["graph_features"]["degree_cv"],
                        "weight_cv": old["graph_features"]["weight_cv"]}
                pair.update(score_pair(record["energy_cd_unified"], old))
                all_pairs.append(pair)
    if len(all_pairs) != 32000 or len({(r["id"], r["p"]) for r in all_pairs}) != 32000:
        raise ValueError("Expected exactly 32,000 unique graph-depth pairs")
    return all_pairs


def _group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [dict(zip(keys, key), **summarize_group(group))
            for key, group in sorted(groups.items())]


def attach_training_reference(groups: list[dict], reference_rows: list[dict]):
    """Match the old paper's transfer-loss definition: train optimum minus fixed test."""
    reference = {(r["run"], r["family"], int(r["p"])): r for r in reference_rows}
    for group in groups:
        key = (group["cohort"], group["family"], group["p"])
        if key not in reference:
            raise ValueError(f"Missing training optimum summary for {key}")
        old = reference[key]
        group["transfer_loss_cd"] = float(old["opt_cd"]) - group["ratio_cd_unified"]
        group["transfer_loss_qaoa"] = float(old["opt_qaoa"]) - group["ratio_qaoa_fixed"]


def _csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style(ax, x=None, y=None, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.7)
    ax.set_axisbelow(True)
    if x: ax.set_xlabel(x)
    if y: ax.set_ylabel(y)
    if title: ax.set_title(title, loc="left")


def _save(fig, name):
    fig.tight_layout()
    destination = FIGURES / name
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(destination)


def make_weight_figure(paired_rows):
    """Pool both test-size cohorts at the graph level for the main weight figure."""
    kinds = ("bimodal", "heavy_tail", "lognormal", "uniform", "unit")
    groups = defaultdict(list)
    for row in paired_rows:
        if int(row["p"]) in (1, 4):
            groups[(row["weight_kind"], int(row["p"]))].append(
                float(row["gain_fixed_norm"])
            )
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    xs = np.arange(len(kinds))
    for offset, p, color in ((-0.18, 1, BLUE), (0.18, 4, GREEN)):
        means, lower, upper = [], [], []
        for kind in kinds:
            values = groups[(kind, p)]
            if len(values) != 1400:
                raise ValueError(f"Expected 1,400 graphs for {kind}, p={p}")
            mean = float(np.mean(values))
            lo, hi = _bootstrap_ci(values)
            means.append(mean)
            lower.append(mean - lo)
            upper.append(hi - mean)
        ax.bar(xs + offset, means, width=0.34, color=color, label=f"p={p}",
               yerr=np.array([lower, upper]), capsize=3,
               error_kw={"ecolor": "#303030", "elinewidth": 1})
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_xticks(xs, ("Bimodal", "Heavy tail", "Lognormal", "Uniform", "Unit"))
    _style(ax, y="Mean paired energy gain / ΣJ",
           title="Same-layer gain by edge-weight distribution")
    ax.legend(frameon=False, ncol=2)
    fig.text(0.5, 0.01, "Each bar: 1,400 held-out graphs (n=10–22); 95% graph bootstrap CI.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf", "svg"):
        destination = FIGURES / f"fig05_weight_distributions_unified.{suffix}"
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        print(destination)
    plt.close(fig)


def make_figures(by_depth, by_family, by_weight, model, paired_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), sharey=True)
    for ax, cohort, label in zip(axes, ("small", "large"), ("n=10–16", "n=18–22")):
        rows = sorted((r for r in by_depth if r["cohort"] == cohort), key=lambda r: r["p"])
        ps = [r["p"] for r in rows]
        ax.plot(ps, [r["ratio_qaoa_fixed"] for r in rows], "o-", color=ORANGE,
                label="Fixed-angle QAOA")
        ax.plot(ps, [r["ratio_cd_unified"] for r in rows], "o-", color=BLUE,
                label="Formula-based DC-QAOA")
        ax.set_xticks(ps)
        _style(ax, "Layers p", "Mean MaxCut approximation ratio" if cohort == "small" else None,
               label)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "fig02_same_depth_unified.png")

    depth_groups = defaultdict(list)
    size_groups = defaultdict(list)
    for row in paired_rows:
        if row["p"] in (1, 2, 3, 4):
            depth_groups[(row["family"], row["p"])].append(row["gain_fixed_norm"])
        if row["p"] == 2 and row["n"] in (10, 14, 18, 22):
            size_groups[(row["family"], row["n"])].append(row["gain_fixed_norm"])
    depths, sizes = (1, 2, 3, 4), (10, 14, 18, 22)
    for family in FAMILIES:
        for p in depths:
            if len(depth_groups[(family, p)]) != 700:
                raise ValueError(f"Incomplete family-depth cell: {family}, p={p}")
        for n in sizes:
            if len(size_groups[(family, n)]) != 100:
                raise ValueError(f"Incomplete family-size cell: {family}, n={n}")
    matrices = [
        np.array([[np.mean(depth_groups[(f, p)]) for p in depths] for f in FAMILIES]),
        np.array([[np.mean(size_groups[(f, n)]) for n in sizes] for f in FAMILIES]),
    ]
    max_abs = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), sharey=True)
    for ax, matrix, labels, title, xlabel in zip(
        axes, matrices, (depths, sizes),
        ("(a) Depth dependence · pooled over n = 10–22", "(b) Size dependence · p = 2"),
        ("Layers p", "Qubits n"),
    ):
        im = ax.imshow(matrix, cmap="RdBu", vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_yticks(range(len(FAMILIES)), FAMILIES)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(FAMILIES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(FAMILIES)):
            for j in range(len(labels)):
                value = matrix[i, j]
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7.4,
                        color="white" if abs(value) > .6 * max_abs else "#202020")
    fig.subplots_adjust(left=.13, right=.87, bottom=.13, top=.9, wspace=.2)
    colorbar_axis = fig.add_axes([.9, .18, .018, .64])
    fig.colorbar(im, cax=colorbar_axis, label="Mean paired energy gain / ΣJ")
    for suffix in ("png", "pdf", "svg"):
        destination = FIGURES / f"fig03_family_heatmap_unified.{suffix}"
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        print(destination)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    xs = np.arange(len(FAMILIES))
    lookup = {(r["family"], r["p"]): r for r in by_family if r["cohort"] == "large"}
    for offset, p, color in ((-0.23, 1, GREEN), (0, 3, BLUE), (0.23, 4, "#303030")):
        ax.bar(xs + offset, [lookup[(f, p)]["transfer_loss_cd"] for f in FAMILIES],
               width=0.22, color=color, label=f"DC-QAOA p={p}")
    ax.plot(xs + 0.36, [lookup[(f, 4)]["transfer_loss_qaoa"] for f in FAMILIES],
            "_", color=ORANGE, markersize=10, markeredgewidth=2,
            label="QAOA p=4")
    ax.set_xticks(xs, FAMILIES, rotation=40, ha="right")
    _style(ax, y="Training optimized − test fixed mean ratio",
           title="Cross-split transfer gap by graph family (n=18–22)")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _save(fig, "fig04_transfer_loss_unified.png")

    make_weight_figure(paired_rows)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for p, color in ((1, "#707070"), (2, GREEN), (3, ORANGE), (4, BLUE), (6, "#303030")):
        row = model["templates"][str(p)]
        x = (np.arange(p) + 0.5) / p
        for ax, key in zip(axes, ("gamma_unit", "beta", "alpha_unit")):
            ax.plot(x, row[key], "o-", color=color, markersize=3.5, label=f"p={p}")
    for ax, title in zip(axes, ("γ unit", "β", "α unit")):
        _style(ax, x="Relative layer position (l−½)/p", title=title)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "fig08_unified_angle_table.png")


def main():
    model, _ = load_archived_inputs()
    rows = collect_pairs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "tables").mkdir(exist_ok=True)
    by_depth = _group(rows, ("cohort", "p"))
    by_size = _group(rows, ("n", "p"))
    by_family = _group(rows, ("cohort", "family", "p"))
    by_weight = _group(rows, ("cohort", "weight_kind", "p"))
    with (ROOT / "analysis" / "outputs" / "tables" / "T6_transfer_loss_long.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        attach_training_reference(by_family, list(csv.DictReader(handle)))
    _csv(OUTPUT / "tables" / "T2_same_depth_unified.csv", by_depth)
    _csv(OUTPUT / "tables" / "T3_by_n_unified.csv", by_size)
    _csv(OUTPUT / "tables" / "T4_by_family_unified.csv", by_family)
    _csv(OUTPUT / "tables" / "T5_by_weight_unified.csv", by_weight)
    _csv(OUTPUT / "tables" / "T1_paired_records_unified.csv", rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_figures(by_depth, by_family, by_weight, model, rows)
    p1 = summarize_group([r for r in rows if r["p"] == 1])
    report = ["# Unified-table held-out evaluation", "",
              f"Model SHA-256: `{EXPECTED_MODEL_SHA256}`. No angle optimization was run.",
              f"Unique graph-depth pairs: {len(rows):,}; unique test graphs: "
              f"{len({r['id'] for r in rows}):,}.", "",
              "## Overall p=1 across 7,000 graphs", "",
              f"Mean approximation ratio: QAOA {p1['ratio_qaoa_fixed']:.6f}; "
              f"unified DC-QAOA {p1['ratio_cd_unified']:.6f}.",
              f"Mean normalized energy gain: {p1['gain_fixed_norm']:.6f} "
              f"[{p1['gain_ci_lo']:.6f}, {p1['gain_ci_hi']:.6f}]; "
              f"DC win fraction {p1['cd_win']:.4%}.", "", "## By cohort and depth", "",
              "| Cohort | p | Pairs | QAOA ratio | Unified DC ratio | Gain / ΣJ [95% CI] | DC wins |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in by_depth:
        report.append(f"| {r['cohort']} | {r['p']} | {r['pairs']:,} | "
                      f"{r['ratio_qaoa_fixed']:.6f} | {r['ratio_cd_unified']:.6f} | "
                      f"{r['gain_fixed_norm']:.6f} [{r['gain_ci_lo']:.6f}, "
                      f"{r['gain_ci_hi']:.6f}] | {r['cd_win']:.2%} |")
    report += ["", "The test graphs and old QAOA comparator/extrema came from the two archived "
               "GPU cohorts. The unified table was constructed after the original cohort "
               "analyses, using training records only, and locked before this evaluation. "
               "No claim of a previously untouched research-wide test split is made."]
    (OUTPUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUTPUT / "REPORT.md")


if __name__ == "__main__":
    main()
