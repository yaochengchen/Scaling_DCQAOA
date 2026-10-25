"""Offline training ONLY: scipy minimization against exact statevector energy."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .instances import IsingInstance, features
from .simulator import Angles, StatevectorSimulator
from .transfer import qaoa_csv


@dataclass
class OptimizationResult:
    angles: Angles
    energy: float
    evaluations: int
    success: bool
    message: str
    gradient_evaluations: int = 0
    trace: list | None = None
    initial_angles: dict | None = None


def optimize(inst: IsingInstance, p: int, *, cd: bool, seed: int,
             restarts: int = 2, maxiter: int = 45) -> list[OptimizationResult]:
    sim = StatevectorSimulator(inst)
    rng = np.random.default_rng(seed)
    outputs = []
    for run in range(restarts):
        # A standalone one-restart call MUST still randomize across different
        # seeds. This supports durable record-level checkpoint/restart.
        style = int(rng.integers(0, 3))
        ref = qaoa_csv(inst, p, sign=(-1 if style != 1 else +1))
        if style == 2:
            gamma = rng.uniform(-0.8, 0.8, p)
            beta = rng.uniform(-np.pi/4, np.pi/4, p)
        else:
            gamma = ref.gamma + rng.normal(0.0, 0.10, p)
            beta = ref.beta + rng.normal(0.0, 0.10, p)
        scale = max(np.sqrt(np.mean(inst.weights**2)), 1e-8)
        alpha = rng.normal(0, 0.15, p)/scale if cd else np.zeros(p)
        a = Angles(gamma, beta, alpha)
        x0 = a.pack(include_cd=cd)
        bounds = [(-3*np.pi, 3*np.pi)]*p + [(-np.pi, np.pi)]*p
        if cd:
            bounds += [(-np.pi*2, np.pi*2)]*p
        calls = 0

        def objective(x):
            nonlocal calls
            calls += 1
            return sim.evaluate(Angles.unpack(x, p, include_cd=cd))

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": int(maxiter), "maxfun": 25*int(maxiter)+50,
                                   "ftol": 1e-10})
        outputs.append(OptimizationResult(Angles.unpack(result.x, p, include_cd=cd),
                                          float(result.fun), calls, bool(result.success),
                                          str(result.message)))
    return outputs


def select_best_per_instance(records: list[dict], *, p: int, cd: bool) -> list[dict]:
    grouped = {}
    for rec in records:
        if rec["p"] == p and bool(rec["cd"]) == cd:
            key = rec["id"]
            if key not in grouped or rec["energy"] < grouped[key]["energy"]:
                grouped[key] = rec
    return list(grouped.values())


def template_from_records(records: list[dict], *, p: int, cd: bool):
    """Medoid of normalized per-instance best solutions; never test data.

    Distances are raw Euclidean in normalized coordinates, not a proven metric
    on symmetry-equivalent circuits; useful starting hypothesis only.
    """
    rows = select_best_per_instance(records, p=p, cd=cd)
    if not rows:
        raise ValueError(f"Missing training records p={p}, cd={cd}")
    from .transfer import degree_factor
    vectors = []
    degrees = []
    for row in rows:
        f = row["features"]
        s = f["rms"]
        df = degree_factor(f["degree"])
        a = row["angles"]
        vectors.append(np.concatenate([np.asarray(a["gamma"])*s/df,
                                       a["beta"], np.asarray(a["alpha"])*s]))
        degrees.append(f["degree"])
    x = np.asarray(vectors)
    d = np.sum((x[:, None, :] - x[None, :, :])**2, axis=2)
    best = int(np.argmin(np.sum(np.sqrt(d), axis=1)))
    ref = x[best]
    return {"gamma_unit": ref[:p].tolist(), "beta": ref[p:2*p].tolist(),
            "alpha_unit": ref[2*p:3*p].tolist(),
            "reference_degree": float(np.median(degrees)),
            "instances": len(rows), "medoid_record_id": rows[best]["id"],
            "normalization": "gamma_unit=gamma*rms/degree_factor, alpha_unit=alpha*rms"}


# --------------------------------------------------------------------------
# Equal-depth study: per-instance optimization chain over ascending depths.
# Convention (verified on the V1 CPU optima): good schedules have gamma<0,
# beta>0 in this code's U_C=exp(-i gamma H_C), U_B=exp(-i beta X) convention.
# The map (gamma, beta) -> (-gamma, -beta) leaves the energy unchanged, so the
# random box below loses no generality for gamma/beta.
# --------------------------------------------------------------------------
INIT_STYLES = ("ramp", "interp", "from_qaoa", "random")


def _units(inst):
    from .transfer import degree_factor
    f = features(inst)
    s = max(f["rms"], 1e-12)
    return degree_factor(f["degree"]) / s, 1.0 / s     # gamma unit, alpha unit


def bounds_for(p: int, cd: bool):
    b = [(-3*np.pi, 3*np.pi)]*p + [(-np.pi, np.pi)]*p
    return b + ([(-2*np.pi, 2*np.pi)]*p if cd else [])


def ramp_start(inst, p, cd, rng=None):
    """Linear-ramp (annealing-like) schedule; jittered when rng is given."""
    gu, au = _units(inst)
    t = (np.arange(p) + 0.5) / p
    dg, db, da = 0.8, 0.6, -0.1
    if rng is not None:
        dg, db, da = rng.uniform(0.4, 1.2), rng.uniform(0.3, 0.9), rng.uniform(-0.3, 0.1)
    parts = [-dg * t * gu, db * (1 - t)]
    if cd:
        parts.append(da * np.ones(p) * au)
    return np.concatenate(parts)


def random_start(inst, p, cd, rng):
    """Independent random start in a normalized box (no ordering imposed)."""
    gu, au = _units(inst)
    parts = [rng.uniform(-1.2, 0.2, p) * gu, rng.uniform(-0.1, 0.9, p)]
    if cd:
        parts.append(rng.uniform(-0.4, 0.2, p) * au)
    return np.concatenate(parts)


def interp_start(prev_x, p_prev, p, cd):
    """INTERP (Zhou et al. 2020): resample the best depth-p_prev schedule to depth p."""
    prev_x = np.asarray(prev_x, float)
    tp = (np.arange(p_prev) + 0.5) / p_prev
    t = (np.arange(p) + 0.5) / p
    blocks = 3 if cd else 2
    out = [np.interp(t, tp, prev_x[k*p_prev:(k+1)*p_prev]) for k in range(blocks)]
    return np.concatenate(out)


def start_plan(p: int, cd: bool, restarts: int, has_prev: bool) -> list[str]:
    """Restart r -> initialization style. Structured starts first, then random.

    QAOA : ramp, interp(p-1) | random ...
    CD   : from_qaoa (best QAOA at the same p, alpha=0), interp(p-1) or ramp, random ...
    The CD warm start guarantees min E_CD(p) <= min E_QAOA(p) up to optimizer
    tolerance (CD contains QAOA at alpha=0), which the equal-depth comparison needs.
    """
    if cd:
        head = ["from_qaoa", "interp" if has_prev else "ramp"]
    else:
        head = ["ramp"] + (["interp"] if has_prev else [])
    plan = head[:restarts]
    plan += ["random"] * (restarts - len(plan))
    return plan


def run_lbfgs(inst, p, cd, x0, sim, *, maxiter=200, trace_callback=None, ftol=1e-10, gtol=1e-5):
    """One L-BFGS-B run with analytic (adjoint) gradients from `sim.value_and_grad`.

    ftol/gtol: scipy L-BFGS-B tolerances.  For complex64 statevectors use about ftol=1e-6,
    gtol=1e-3 (energy noise ~5e-6 relative, gradient noise ~1e-3 absolute); tighter values
    only add ABNORMAL line-search stops and wasted calls.
    """
    calls = 0
    best = float("inf")

    def objective(x):
        nonlocal calls, best
        calls += 1
        value, grad = sim.value_and_grad(x, p, cd)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Non-finite energy or gradient")
        best = min(best, float(value))
        if trace_callback is not None:
            trace_callback({"evaluation": calls, "angles": np.asarray(x).tolist(),
                            "energy": float(value), "best_so_far_energy": best,
                            "grad_norm": float(np.linalg.norm(grad))})
        return value, grad

    x0 = np.clip(np.asarray(x0, float), [b[0] for b in bounds_for(p, cd)],
                 [b[1] for b in bounds_for(p, cd)])
    result = minimize(objective, x0, method="L-BFGS-B", jac=True, bounds=bounds_for(p, cd),
                      options={"maxiter": int(maxiter), "maxfun": 4*int(maxiter)+50,
                               "ftol": float(ftol), "gtol": float(gtol)})
    return OptimizationResult(Angles.unpack(result.x, p, include_cd=cd), float(result.fun), calls,
                              bool(result.success), str(result.message), calls, None,
                              Angles.unpack(x0, p, include_cd=cd).to_dict())
