"""Pure statistical helpers for final research analysis.

These functions consume saved records only.  They do not run circuits, fit
models, or inspect target instances beyond quantities already recorded.
"""
from __future__ import annotations

import numpy as np

from .optimization import select_best_per_instance
from .transfer import degree_factor


def _bootstrap_mean_ci(values, *, n_boot: int, seed: int = 1441) -> list[float]:
    x = np.asarray(values, dtype=float)
    if x.size == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(x.size, size=(n_boot, x.size))], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_method_summary(rows: list[dict], reference_key: str, candidate_key: str,
                          *, n_boot: int = 5000) -> dict:
    """Summarize normalized paired gain; positive means candidate is better."""
    gain = np.asarray([
        (float(row[reference_key]) - float(row[candidate_key])) /
        max(float(row["norm_weights"]), 1e-15)
        for row in rows
    ])
    if gain.size == 0:
        raise ValueError("At least one paired row is required")
    tol = 1e-9
    return {
        "n": int(gain.size),
        "mean_gain": float(np.mean(gain)),
        "median_gain": float(np.median(gain)),
        "bootstrap_95_ci": _bootstrap_mean_ci(gain, n_boot=n_boot),
        "win_rate": float(np.mean(gain > tol)),
        "tie_rate": float(np.mean(np.abs(gain) <= tol)),
        "loss_rate": float(np.mean(gain < -tol)),
        "min_gain": float(np.min(gain)),
        "max_gain": float(np.max(gain)),
    }


def concentration_by_depth(records: list[dict]) -> dict[str, dict]:
    """Describe best-per-instance CD angles after the declared normalization.

    Raw Euclidean dispersion does not resolve periodic/equivalent branches;
    the returned label records this limitation explicitly.
    """
    output: dict[str, dict] = {}
    depths = sorted({int(r["p"]) for r in records if bool(r["cd"])})
    for p in depths:
        all_rows = [r for r in records if bool(r["cd"]) and int(r["p"]) == p]
        best = select_best_per_instance(records, p=p, cd=True)
        gamma_unit, beta, alpha_unit = [], [], []
        for row in best:
            f = row["features"]
            rms = float(f["rms"])
            df = degree_factor(float(f["degree"]))
            a = row["angles"]
            gamma_unit.append(np.asarray(a["gamma"], dtype=float) * rms / df)
            beta.append(np.asarray(a["beta"], dtype=float))
            alpha_unit.append(np.asarray(a["alpha"], dtype=float) * rms)
        g = np.asarray(gamma_unit)
        b = np.asarray(beta)
        a = np.asarray(alpha_unit)
        stacked = np.concatenate([g, b, a], axis=1)
        distances = np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=2)
        medoid_index = int(np.argmin(np.sum(distances, axis=1)))
        output[str(p)] = {
            "n_instances": len(best),
            "n_restarts": len(all_rows),
            "all_restart_success_rate": float(np.mean([bool(r["optimizer_success"]) for r in all_rows])),
            "gamma_unit_median": np.median(g, axis=0).tolist(),
            "gamma_unit_std": np.std(g, axis=0).tolist(),
            "beta_median": np.median(b, axis=0).tolist(),
            "beta_std": np.std(b, axis=0).tolist(),
            "alpha_unit_median": np.median(a, axis=0).tolist(),
            "alpha_unit_std": np.std(a, axis=0).tolist(),
            "alpha_unit_iqr": (np.quantile(a, 0.75, axis=0) - np.quantile(a, 0.25, axis=0)).tolist(),
            "medoid_record_id": best[medoid_index]["id"],
            "median_distance_to_medoid": float(np.median(distances[medoid_index])),
            "interpretation": "Best restart per instance; normalized but not symmetry/periodicity aligned.",
        }
    return output
