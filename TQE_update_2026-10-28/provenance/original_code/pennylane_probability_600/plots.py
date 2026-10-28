"""Aggregate verified PennyLane results into the supplementary probability figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import campaign
import core

ROOT = campaign.ROOT
OUT = campaign.OUT
FIG = ROOT / "figures" / "pennylane_probability_low_degree_p6_600"
BLUE = "#2476cf"
ORANGE = "#e87837"


def original_record(b, gid, p):
    source = campaign.SOURCE / "records" / f"{campaign.taskkey(gid, p)}.json"
    r = json.loads(source.read_text(encoding="utf-8"))
    body = dict(r)
    if body.pop("content_sha256", None) != core.object_digest(body):
        raise ValueError(f"Bad original record checksum: {source}")
    if (r["campaign"] != b["imported_campaign"] or
            r["fingerprint"] != b["graphs"][gid]["fingerprint"] or
            r["backend"] != core.BACKEND):
        raise ValueError(f"Wrong original record: {source}")
    return {kind: r["methods"][kind]["probability"] for kind in ("u1", "qaoa")}


def bootstrap(matrix, strata):
    """Graph bootstrap within each size; same graph draws for all depths/methods."""
    rng = np.random.default_rng(20261027)
    draws = np.empty((2000,) + matrix.shape[1:], dtype=float)
    strata = np.asarray(strata)
    unique = sorted(set(strata))
    for i in range(2000):
        take = np.concatenate([rng.choice(np.flatnonzero(strata == n), size=int(np.sum(strata == n)), replace=True)
                               for n in unique])
        draws[i] = matrix[take].mean(axis=0)
    return np.quantile(draws, [.025, .975], axis=0)


def extract_main(b):
    ids = b["selected_ids"]
    ns = np.array([b["graphs"][gid]["n"] for gid in ids])
    values = np.empty((600, 5, 2))
    rows = []
    for i, gid in enumerate(ids):
        for j, p in enumerate(campaign.DEPTHS):
            if p == 6:
                pair = {kind: campaign.check_method(b, gid, p, kind)["result"]["probability"]
                        for kind in ("u1", "qaoa")}
            else:
                pair = original_record(b, gid, p)
            values[i, j] = [pair["u1"], pair["qaoa"]]
            rows.append({"id": gid, "n": int(ns[i]), "p": p,
                         "family": b["graphs"][gid]["family"],
                         "weight_kind": b["graphs"][gid]["weight_kind"],
                         "probability_powerlaw": pair["u1"],
                         "probability_qaoa": pair["qaoa"],
                         "paired_difference": pair["u1"] - pair["qaoa"]})
    return ns, values, rows


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_main(ns, values):
    depth_mean = values.mean(axis=0)
    depth_ci = bootstrap(values, ns)
    size_mean = np.stack([values[ns == n, -1].mean(axis=0) for n in (10, 12, 14, 16)])
    size_ci = np.stack([bootstrap(values[ns == n, -1], ns[ns == n]) for n in (10, 12, 14, 16)])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True, layout="constrained")
    for ax, xs, mean, ci, connected in (
        (axes[0], campaign.DEPTHS, depth_mean, depth_ci, True),
        (axes[1], (10, 12, 14, 16), size_mean, size_ci.transpose(1, 0, 2), False),
    ):
        for method, label, color, marker in ((1, "Fixed-angle QAOA", ORANGE, "s"),
                                             (0, "Power-law DC-QAOA", BLUE, "o")):
            y = mean[:, method] * 100
            bounds = ci[:, :, method] * 100
            errs = np.maximum(0, np.array([y - bounds[0], bounds[1] - y]))
            ax.errorbar(xs, y, yerr=errs, fmt=marker, linestyle="-" if connected else "none",
                        color=color, linewidth=1.6, markersize=5, capsize=2.5,
                        elinewidth=1, label=label)
        ax.set_xticks(xs)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    ymax = max(float(np.max(depth_ci[1])), float(np.max(size_ci[:, 1]))) * 100 * 1.15
    axes[0].set_ylim(0, max(10, ymax))
    axes[0].set_title("(a) By depth: the same 600 graphs", loc="left", fontsize=9)
    axes[0].set_xlabel("Layers, $p$")
    axes[0].set_ylabel("Mean optimal-solution probability (%)")
    axes[0].legend(frameon=False, fontsize=7.3, loc="upper left")
    axes[1].set_title("(b) By size at $p=6$: 150 graphs/point", loc="left", fontsize=9)
    axes[1].set_xlabel("Qubits, $n$")
    save(fig, "supplement_p6_600_powerlaw_vs_qaoa_probability")
    d = values[:, :, 0] - values[:, :, 1]
    dci = bootstrap(d, ns)
    return {"depths": {str(p): {"mean": depth_mean[j].tolist(),
                                "ci95": depth_ci[:, j].T.tolist(),
                                "mean_paired_difference": float(d[:, j].mean()),
                                "paired_difference_ci95": dci[:, j].tolist()}
                       for j, p in enumerate(campaign.DEPTHS)},
            "p6_by_n": {str(n): {"mean": size_mean[j].tolist(),
                                  "ci95": size_ci[j].T.tolist(),
                                  "paired_difference": float(size_mean[j, 0] - size_mean[j, 1])}
                        for j, n in enumerate((10, 12, 14, 16))}}


def render_low(b):
    gids = b["low_ids"]
    rows = []
    values = np.empty((11, 5, 3))
    for i, gid in enumerate(gids):
        for j, p in enumerate(campaign.DEPTHS):
            vals = [campaign.check_method(b, gid, p, kind)["result"]["probability"]
                    for kind in ("u1", "qaoa", "old")]
            values[i, j] = vals
            rows.append({"id": gid, "n": b["graphs"][gid]["n"], "p": p,
                         "probability_powerlaw": vals[0], "probability_qaoa": vals[1],
                         "probability_previous_rule": vals[2],
                         "paired_difference_qaoa": vals[0] - vals[1],
                         "paired_difference_previous_rule": vals[0] - vals[2]})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), layout="constrained")
    for k, label, color, marker in ((0, "Power-law DC-QAOA", BLUE, "o"),
                                     (1, "Fixed-angle QAOA", ORANGE, "s")):
        axes[0].plot(campaign.DEPTHS, values[:, :, k].mean(axis=0) * 100,
                     color=color, marker=marker, linewidth=1.6, label=label)
    axes[0].set(xlabel="Layers, $p$", ylabel="Mean optimal-solution probability (%)",
                title="(a) Eleven low-degree graphs", xticks=campaign.DEPTHS)
    axes[0].legend(frameon=False, fontsize=7.5)
    diff = (values[:, :, 0] - values[:, :, 1]) * 100
    for row in diff:
        axes[1].plot(campaign.DEPTHS, row, color="#bfcbd6", lw=.7, alpha=.65)
    axes[1].plot(campaign.DEPTHS, diff.mean(axis=0), color=BLUE,
                 marker="o", lw=1.8, label="Mean paired difference")
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].set(xlabel="Layers, $p$", ylabel="Power-law minus QAOA (percentage points)",
                title="(b) Paired probability differences", xticks=campaign.DEPTHS)
    axes[1].legend(frameon=False, fontsize=7.5)
    for ax in axes:
        ax.grid(axis="y", color="#e5e5e5", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "supplement_low_degree_11_probability_diagnostic")
    summary = {str(p): {"mean_powerlaw": float(values[:, j, 0].mean()),
                        "mean_qaoa": float(values[:, j, 1].mean()),
                        "mean_previous_rule": float(values[:, j, 2].mean()),
                        "mean_powerlaw_minus_qaoa": float((values[:, j, 0] - values[:, j, 1]).mean()),
                        "mean_powerlaw_minus_previous_rule": float((values[:, j, 0] - values[:, j, 2]).mean())}
               for j, p in enumerate(campaign.DEPTHS)}
    return rows, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(b):
    ns, values, paired = extract_main(b)
    if len(paired) != 3000 or len({(r["id"], r["p"]) for r in paired}) != 3000:
        raise ValueError("The 600-graph five-depth panel must have 3000 pairs")
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "paired_probability_600_five_depths.csv", paired)
    main_summary = render_main(ns, values)
    low_rows, low_summary = render_low(b)
    write_csv(OUT / "low_degree_probability_55.csv", low_rows)
    summary = {"campaign": b["campaign"], "main_graphs": 600,
               "main_graph_depth_pairs": 3000, "main_methods": ("u1", "qaoa"),
               "bootstrap_replicates": 2000,
               "bootstrap": "size-stratified graph bootstrap; same graph draws for both methods and all depths",
               "main": main_summary, "low_degree_graphs": 11,
               "low_degree_graph_depth_pairs": 55, "low_degree": low_summary}
    core.atomic_json(OUT / "summary.json", summary)
    text = "# 600 图 $p=6$ 和 11 图低度专项：PennyLane 最优解概率结果\n\n"
    text += "全部新概率由 PennyLane `lightning.qubit`、`complex128`、`shots=None` 计算。"
    text += "主表固定为同一批 600 图；额外低度图不进入该均值。\n\n"
    text += "|深度 $p$|Power-law DC-QAOA|Fixed-angle QAOA|配对差（百分点）|\n|---:|---:|---:|---:|\n"
    for p in campaign.DEPTHS:
        item = main_summary["depths"][str(p)]
        text += f"|{p}|{item['mean'][0]*100:.3f}%|{item['mean'][1]*100:.3f}%|{item['mean_paired_difference']*100:+.3f}|\n"
    text += "\n图中误差棒是 2,000 次按规模分层、保持图级配对的逐点 bootstrap 区间。"
    text += "11 图低度结果为专项描述性诊断，样本量较小。\n\n"
    text += "![600-graph depth-six probability](../figures/pennylane_probability_low_degree_p6_600/supplement_p6_600_powerlaw_vs_qaoa_probability.png)\n\n"
    text += "![low-degree diagnostic](../figures/pennylane_probability_low_degree_p6_600/supplement_low_degree_11_probability_diagnostic.png)\n"
    (ROOT / "reports" / "PennyLane_600图p6与低度概率_运行结果.md").write_text(text, encoding="utf-8")
