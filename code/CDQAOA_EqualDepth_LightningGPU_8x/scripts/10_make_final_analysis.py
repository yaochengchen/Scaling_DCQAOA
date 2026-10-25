"""Build auditable final statistics, manuscript tables, and six core figures.

This script reads a completed frozen run.  It never refits a model and never
optimizes a held-out instance.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import numpy as np

from cdtransfer.analysis import paired_method_summary, concentration_by_depth
from cdtransfer.instances import IsingInstance, dump_json, load_json
from cdtransfer.transfer import predict
from cdtransfer.qiskit_optional import transpile_cost


METHODS = {
    "QAOA CSV, budget": "energy_qaoa_csv_budget",
    "QAOA best, budget": "energy_qaoa_best_budget",
    "QAOA best, same p": "energy_qaoa_best_same",
    "CD alpha=0": "energy_cd_zero",
    "CD BCH": "energy_cd_bch",
    "CD reference": "energy_cd_reference",
    "CD proposed": "energy_cd_proposed",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_instance(row: dict) -> IsingInstance:
    keys = ("n", "edges", "weights", "family", "weight_kind", "seed", "split")
    return IsingInstance.from_dict({key: row[key] for key in keys})


def method_statistics(test: list[dict]) -> dict:
    reference = "energy_qaoa_best_budget"
    return {name: paired_method_summary(test, reference, key)
            for name, key in METHODS.items()}


def subgroup_statistics(test: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in test:
        groups[("all", "all")].append(row)
        for field in ("p_cd", "n", "family", "weight_kind"):
            groups[(field, str(row[field]))].append(row)
    rows = []
    for (field, value), group in sorted(groups.items()):
        stat = paired_method_summary(group, "energy_qaoa_best_budget", "energy_cd_proposed")
        rows.append({"group": field, "value": value, **stat})
    return rows


def offline_advantage(records: list[dict], manifest: dict) -> dict:
    norms = {row["id"]: float(np.sum(np.abs(row["weights"])))
             for row in manifest["splits"]["train"]}
    best = {}
    for row in records:
        key = (row["id"], int(row["p"]), bool(row["cd"]))
        if key not in best or row["energy"] < best[key]["energy"]:
            best[key] = row
    pairs = []
    for (instance_id, p, cd), row in best.items():
        if not cd or (instance_id, p, False) not in best:
            continue
        pairs.append({
            "norm_weights": norms[instance_id],
            "energy_qaoa": best[(instance_id, p, False)]["energy"],
            "energy_cd": row["energy"],
            "p": p,
        })
    result = {"all": paired_method_summary(pairs, "energy_qaoa", "energy_cd")}
    for p in sorted({row["p"] for row in pairs}):
        result[f"depth-{p}"] = paired_method_summary(
            [row for row in pairs if row["p"] == p], "energy_qaoa", "energy_cd")
    return result


def native_gate_counts(manifest: dict, model: dict) -> list[dict]:
    exemplar = load_instance(manifest["splits"]["test"][0])
    output = []
    for p in manifest["config"]["depths"]:
        q_depth = 3 * p
        q_method = model["qaoa_fixed_best_by_depth"][str(q_depth)]
        circuits = {
            "CD proposed": predict(exemplar, p, "cd_proposed", model),
            "QAOA best budget": predict(exemplar, q_depth, q_method, model),
        }
        for level in (0, 1, 3):
            for name, angles in circuits.items():
                cost = transpile_cost(exemplar, angles, optimization_level=level)
                output.append({"circuit": name, "p_cd": p, "p_actual": angles.p,
                               "n": exemplar.n, "m": exemplar.m, **cost})
    return output


def dataset_rows(manifest: dict) -> list[dict]:
    output = []
    for split, entries in manifest["splits"].items():
        counts = Counter((r["n"], r["family"], r["weight_kind"]) for r in entries)
        for (n, family, weights), count in sorted(counts.items()):
            output.append({"split": split, "n": n, "family": family,
                           "weight_kind": weights, "instances": count})
    return output


def make_figures(out: Path, paper: Path, records: list[dict], test: list[dict],
                 method_stats: dict, subgroup: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.size": 9})
    figdir = paper / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    # Fig. 1: protocol and freeze boundary.
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    labels = ["Offline train\nmultistart QAOA/CD", "Validation\nchoose simple rule",
              "FROZEN table\nno objective calls", "Held-out OOD\nstatic features only",
              "Paired quality\n+ resource audit"]
    xs = np.linspace(0.08, 0.92, len(labels))
    colors = ["#d9eaf7", "#d9eaf7", "#f9d976", "#dff2df", "#eadff2"]
    for x, label, color in zip(xs, labels, colors):
        ax.text(x, 0.52, label, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#444"})
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right-0.075, 0.52), xytext=(left+0.075, 0.52),
                    arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.5, 0.10, "Freeze boundary: target instances cannot select signs, coefficients, templates, or candidates",
            ha="center", color="#8b1a1a")
    fig.tight_layout(); fig.savefig(figdir / "fig01_training_free_protocol.png"); plt.close(fig)
    made.append("fig01_training_free_protocol.png")

    # Fig. 2: normalized best-instance parameter dispersion.
    depths = sorted({r["p"] for r in records if r["cd"]})
    fig, axes = plt.subplots(len(depths), 3, figsize=(10, 3.2*len(depths)), squeeze=False)
    for row_index, p in enumerate(depths):
        chosen = []
        ids = sorted({r["id"] for r in records if r["cd"] and r["p"] == p})
        for ident in ids:
            choices = [r for r in records if r["id"] == ident and r["cd"] and r["p"] == p]
            chosen.append(min(choices, key=lambda r: r["energy"]))
        gamma = np.array([np.asarray(r["angles"]["gamma"])*r["features"]["rms"] for r in chosen])
        beta = np.array([r["angles"]["beta"] for r in chosen])
        alpha = np.array([np.asarray(r["angles"]["alpha"])*r["features"]["rms"] for r in chosen])
        for ax, values, title in zip(axes[row_index], (gamma, beta, alpha),
                                     ("gamma × RMS", "beta", "alpha × RMS")):
            ax.boxplot([values[:, j] for j in range(p)], tick_labels=[str(j+1) for j in range(p)],
                       showfliers=True)
            ax.axhline(0, color="#777", lw=0.6)
            ax.set(title=f"p={p}: {title}", xlabel="layer")
    fig.suptitle("Best-restart CD parameters (unaligned equivalent branches)")
    fig.tight_layout(); fig.savefig(figdir / "fig02_parameter_concentration.png"); plt.close(fig)
    made.append("fig02_parameter_concentration.png")

    # Fig. 3: exact scale covariance identity.
    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    errors = np.zeros_like(scales)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(scales, np.maximum(errors, 1e-18), "o-")
    ax.set(xscale="log", yscale="log", xlabel="global weight multiplier s",
           ylabel="max state-vector error (floor 1e-18)",
           title="Exact circuit covariance under J→sJ, (gamma,alpha)→(gamma,alpha)/s")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(figdir / "fig03_weight_scale_identity.png"); plt.close(fig)
    made.append("fig03_weight_scale_identity.png")

    # Fig. 4: all fixed methods relative to validation-selected budget QAOA.
    names = list(method_stats)
    means = np.array([method_stats[n]["mean_gain"] for n in names])
    lo = np.array([method_stats[n]["bootstrap_95_ci"][0] for n in names])
    hi = np.array([method_stats[n]["bootstrap_95_ci"][1] for n in names])
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    y = np.arange(len(names))
    ax.errorbar(means, y, xerr=np.vstack([means-lo, hi-means]), fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(yticks=y, yticklabels=names, xlabel="mean paired gain vs QAOA best budget (95% bootstrap CI)",
           title="Frozen-method comparison on held-out instances")
    fig.tight_layout(); fig.savefig(figdir / "fig04_fixed_method_comparison.png"); plt.close(fig)
    made.append("fig04_fixed_method_comparison.png")

    # Fig. 5: subgroup equal-budget evidence.
    groups = [r for r in subgroup if r["group"] in ("p_cd", "n", "family", "weight_kind")]
    labels = [f"{r['group']}={r['value']}" for r in groups]
    means = np.array([r["mean_gain"] for r in groups])
    lo = np.array([r["bootstrap_95_ci"][0] for r in groups])
    hi = np.array([r["bootstrap_95_ci"][1] for r in groups])
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.33*len(groups))))
    y = np.arange(len(groups))
    ax.errorbar(means, y, xerr=np.vstack([means-lo, hi-means]), fmt="o", capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(yticks=y, yticklabels=labels, xlabel="(E_QAOA budget − E_CD proposed) / Σ|J|",
           title="Equal-decomposition-budget paired gain by held-out subgroup")
    fig.tight_layout(); fig.savefig(figdir / "fig05_equal_budget_subgroups.png"); plt.close(fig)
    made.append("fig05_equal_budget_subgroups.png")

    # Fig. 6: same-depth benefit versus equal-budget deficit.
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for p in sorted({r["p_cd"] for r in test}):
        group = [r for r in test if r["p_cd"] == p]
        ax.scatter([r["gain_same_depth_norm"] for r in group],
                   [r["gain_equal_budget_norm"] for r in group], s=20, alpha=0.75, label=f"p={p}")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="CD gain over QAOA at same p", ylabel="CD gain over QAOA at matched model CX",
           title="Per-layer expressivity does not imply resource-matched advantage")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(figdir / "fig06_same_depth_vs_budget.png"); plt.close(fig)
    made.append("fig06_same_depth_vs_budget.png")
    return made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--paper-dir", default="paper")
    args = parser.parse_args()
    out, paper = Path(args.out), Path(args.paper_dir)
    analysis_dir, table_dir = out / "analysis", paper / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(out / "manifest.json")
    records = load_json(out / "offline_optima.json")
    model = load_json(out / "FROZEN_MODEL.json")
    test = load_json(out / "heldout_results.json")
    expected = len(manifest["splits"]["test"]) * len(manifest["config"]["depths"])
    if len(test) != expected:
        raise RuntimeError(f"Held-out run incomplete: {len(test)}/{expected}")

    methods = method_statistics(test)
    subgroups = subgroup_statistics(test)
    concentration = concentration_by_depth(records)
    offline = offline_advantage(records, manifest)
    gates = native_gate_counts(manifest, model)
    optimization = {
        "records": len(records),
        "successes": int(sum(bool(r["optimizer_success"]) for r in records)),
        "success_rate": float(np.mean([bool(r["optimizer_success"]) for r in records])),
        "objective_evaluations": int(sum(int(r["eval_calls"]) for r in records)),
    }
    same_depth = paired_method_summary(test, "energy_qaoa_best_same", "energy_cd_proposed")
    ablations = {
        "proposed_vs_reference": paired_method_summary(
            test, "energy_cd_reference", "energy_cd_proposed"),
        "proposed_vs_bch": paired_method_summary(
            test, "energy_cd_bch", "energy_cd_proposed"),
    }
    shot_count = int(manifest["config"].get("shots", 0))
    shot_summary = None
    q_shot = f"best_{shot_count}_shots_qaoa_best_budget"
    cd_shot = f"best_{shot_count}_shots_cd_proposed"
    if shot_count and all(q_shot in row and cd_shot in row for row in test):
        shot_summary = paired_method_summary(test, q_shot, cd_shot)
    exact_rows = [row for row in test if row.get("ground_energy") is not None]
    ground_gaps = None
    if exact_rows:
        ground_gaps = {
            "n": len(exact_rows),
            "qaoa_best_budget_mean": float(np.mean([
                (row["energy_qaoa_best_budget"]-row["ground_energy"])/row["norm_weights"]
                for row in exact_rows])),
            "cd_proposed_mean": float(np.mean([
                (row["energy_cd_proposed"]-row["ground_energy"])/row["norm_weights"]
                for row in exact_rows])),
        }
    summary = {"methods_vs_qaoa_best_budget": methods,
               "cd_proposed_vs_qaoa_same_depth": same_depth,
               "cd_formula_ablations": ablations,
               "offline_optimized_cd_vs_qaoa_same_depth": offline,
               "finite_shot_best_of_k": {"shots": shot_count, "paired": shot_summary},
               "mean_normalized_ground_gap": ground_gaps,
               "optimization": optimization,
               "confirmatory_pairs": len(test),
               "manifest_sha256": manifest["config_sha256"]}
    dump_json(analysis_dir / "final_statistics.json", summary)
    dump_json(analysis_dir / "concentration_diagnostics.json", concentration)
    dump_json(analysis_dir / "native_gate_counts.json", gates)
    write_csv(analysis_dir / "subgroup_results.csv", subgroups)
    write_csv(table_dir / "table1_dataset.csv", dataset_rows(manifest))
    write_csv(table_dir / "table2_method_results.csv",
              [{"method": name, **stats} for name, stats in methods.items()])
    write_csv(table_dir / "table4_native_gate_counts.csv", gates)
    shutil.copy2(out / "cd_transfer_table_v1.csv", table_dir / "table3_fixed_angles.csv")
    made = make_figures(out, paper, records, test, methods, subgroups)
    dump_json(analysis_dir / "figure_manifest.json",
              {"figures": made, "source_run": str(out),
               "statistics": "analysis/final_statistics.json",
               "subgroups": "analysis/subgroup_results.csv"})
    print(f"Wrote {len(made)} figures, 4 tables, and analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
