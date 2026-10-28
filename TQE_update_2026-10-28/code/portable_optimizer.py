import numpy as np,math,time,hashlib
from scipy.optimize import minimize
from cdtransfer.simulator import Angles
from cdtransfer.instances import IsingInstance
from pennylane_n12p2 import make_pennylane_evaluator
class BudgetStop(Exception):pass
def optimizer_options(budget):return {"maxiter":80 if budget<=200 else budget,"ftol":1e-10}
BASE_SEED=20261022
def run_one(row, fixed_angles, start, method, budget=500):
    inst=IsingInstance.from_dict({k:row[k] for k in ('n','edges','weights','family','weight_kind','seed','split')})
    evaluate_energy=make_pennylane_evaluator(inst)
    fixed_energy=evaluate_energy(fixed_angles)
    optimum=(row['max_energy']-row['ground_energy'])/2
    ratio=lambda e:(row['max_energy']-e)/(2*optimum)
    fixed_ratio=ratio(fixed_energy)
    backend='pennylane'
    bounds = [(-math.pi, math.pi)] * 2 + [(-math.pi / 2, math.pi / 2)] * 2 + [(-math.pi, math.pi)] * 2
    seed = int.from_bytes(hashlib.sha256(f"{BASE_SEED}:{inst.fingerprint}:{start}".encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    x0 = np.array([rng.uniform(low, high) for low, high in bounds])
    evaluations = []
    best = -np.inf
    error = None
    optimizer_result = None
    began = time.perf_counter()

    def objective(x):
        nonlocal best
        if len(evaluations) >= budget:
            raise BudgetStop()
        energy = evaluate_energy(Angles.unpack(x, 2, True))
        value = ratio(energy)
        if not np.isfinite(value) or not -1e-7 <= value <= 1 + 1e-7:
            raise ValueError("Invalid approximation ratio")
        best = max(best, value)
        evaluations.append({"evaluation": len(evaluations) + 1,
                            "ratio": float(value), "best_ratio": float(best)})
        return energy

    try:
        if method == "Adam":
            x = x0.copy()
            first_moment = np.zeros_like(x)
            second_moment = np.zeros_like(x)
            step = 0
            while True:
                fx = objective(x)
                gradient = np.zeros_like(x)
                for coordinate, (lower, upper) in enumerate(bounds):
                    probe = x.copy()
                    probe[coordinate] = (x[coordinate] + 1e-4 if x[coordinate] + 1e-4 <= upper
                                         else x[coordinate] - 1e-4)
                    gradient[coordinate] = (objective(probe) - fx) / (probe[coordinate] - x[coordinate])
                step += 1
                first_moment = .9 * first_moment + .1 * gradient
                second_moment = .999 * second_moment + .001 * gradient * gradient
                correction = (.08 * (first_moment / (1 - .9 ** step))
                              / (np.sqrt(second_moment / (1 - .999 ** step)) + 1e-8))
                x = np.clip(x - correction, [bound[0] for bound in bounds],
                            [bound[1] for bound in bounds])
        elif method == "COBYLA":
            optimizer_result = minimize(objective, x0, method="COBYLA", bounds=bounds,
                                        options={"maxiter": budget, "rhobeg": .5, "tol": 1e-4})
        else:
            optimizer_result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                        options=optimizer_options(budget))
    except BudgetStop:
        pass
    except Exception as exc:
        error = repr(exc)
    return {"id": row["id"], "fingerprint": inst.fingerprint, "family": inst.family,
            "weight_kind": inst.weight_kind, "n": inst.n, "p": 2, "start": start, "seed": seed,
            "method": method, "backend": backend, "initial_angles": x0.tolist(),
            "fixed_angles": fixed_angles.to_dict(), "fixed_ratio": float(fixed_ratio),
            "fixed_energy": float(fixed_energy), "optimum": float(optimum),
            "evaluations": evaluations, "nfev": len(evaluations),
            "nit": (int(optimizer_result.nit) if optimizer_result is not None
                    and getattr(optimizer_result, "nit", None) is not None else None),
            "optimizer_status": int(optimizer_result.status) if optimizer_result is not None else None,
            "optimizer_success": bool(optimizer_result.success) if optimizer_result else False,
            "optimizer_message": str(optimizer_result.message) if optimizer_result else None,
            "budget_stop": len(evaluations) >= budget, "error": error,
            "seconds": time.perf_counter() - began}
