"""Exact positive-weight MaxCut helpers with independent solver verification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .instances import IsingInstance


@dataclass(frozen=True)
class VerifiedMaxCut:
    optimum: float
    scip_objective: float
    gurobi_objective: float
    scip_status: str
    gurobi_status: str
    scip_seconds: float
    gurobi_seconds: float
    scip_version: str
    gurobi_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_positive(inst: IsingInstance) -> None:
    if not np.all(np.asarray(inst.weights) > 0):
        raise ValueError("MaxCut approximation ratios require strictly positive weights")


def expected_cut_from_ising(inst: IsingInstance, ising_energy: float) -> float:
    _validate_positive(inst)
    value = 0.5 * (float(np.sum(inst.weights)) - float(ising_energy))
    if not np.isfinite(value):
        raise ValueError("Expected cut value must be finite")
    return value


def _solve_scip(inst: IsingInstance) -> tuple[float, str, float, str]:
    from pyscipopt import Model, quicksum

    started = perf_counter()
    model = Model("maxcut")
    model.hideOutput()
    x = {i: model.addVar(vtype="B", name=f"x_{i}") for i in range(inst.n)}
    y = {
        tuple(e): model.addVar(vtype="B", name=f"y_{e[0]}_{e[1]}")
        for e in inst.edges.tolist()
    }
    model.addCons(x[0] == 0)
    for i, j in y:
        model.addCons(y[i, j] <= x[i] + x[j])
        model.addCons(y[i, j] <= 2 - x[i] - x[j])
        model.addCons(y[i, j] >= x[i] - x[j])
        model.addCons(y[i, j] >= x[j] - x[i])
    model.setObjective(
        quicksum(
            float(w) * y[tuple(e)]
            for e, w in zip(inst.edges.tolist(), inst.weights)
        ),
        "maximize",
    )
    model.optimize()
    status = str(model.getStatus()).lower()
    objective = float(model.getObjVal()) if status == "optimal" else float("nan")
    version = ".".join(
        map(
            str,
            (
                model.getMajorVersion(),
                model.getMinorVersion(),
                model.getTechVersion(),
            ),
        )
    )
    return objective, status, perf_counter() - started, version


def _solve_gurobi(inst: IsingInstance) -> tuple[float, str, float, str]:
    import gurobipy as gp
    from gurobipy import GRB

    started = perf_counter()
    model = gp.Model("maxcut")
    model.Params.OutputFlag = 0
    x = model.addVars(inst.n, vtype=GRB.BINARY, name="x")
    edges = [tuple(e) for e in inst.edges.tolist()]
    y = model.addVars(edges, vtype=GRB.BINARY, name="y")
    model.addConstr(x[0] == 0)
    for i, j in edges:
        model.addConstr(y[i, j] <= x[i] + x[j])
        model.addConstr(y[i, j] <= 2 - x[i] - x[j])
        model.addConstr(y[i, j] >= x[i] - x[j])
        model.addConstr(y[i, j] >= x[j] - x[i])
    model.setObjective(
        gp.quicksum(
            float(w) * y[tuple(e)]
            for e, w in zip(inst.edges.tolist(), inst.weights)
        ),
        GRB.MAXIMIZE,
    )
    model.optimize()
    status = "optimal" if model.Status == GRB.OPTIMAL else f"status-{model.Status}"
    objective = float(model.ObjVal) if model.Status == GRB.OPTIMAL else float("nan")
    return (
        objective,
        status,
        perf_counter() - started,
        ".".join(map(str, gp.gurobi.version())),
    )


def solve_maxcut_verified(
    inst: IsingInstance, *, atol: float = 1e-8, rtol: float = 1e-8
) -> VerifiedMaxCut:
    _validate_positive(inst)
    s_obj, s_status, s_time, s_version = _solve_scip(inst)
    g_obj, g_status, g_time, g_version = _solve_gurobi(inst)
    if s_status != "optimal" or g_status != "optimal":
        raise RuntimeError(f"Exact solver failure: SCIP={s_status}, Gurobi={g_status}")
    if not np.isfinite(s_obj) or not np.isfinite(g_obj):
        raise RuntimeError(
            f"Exact solver returned non-finite objectives: SCIP={s_obj}, Gurobi={g_obj}"
        )
    if not np.isclose(s_obj, g_obj, atol=atol, rtol=rtol):
        raise RuntimeError(f"Exact solver disagreement: SCIP={s_obj}, Gurobi={g_obj}")
    return VerifiedMaxCut(
        optimum=float((s_obj + g_obj) / 2),
        scip_objective=s_obj,
        gurobi_objective=g_obj,
        scip_status=s_status,
        gurobi_status=g_status,
        scip_seconds=s_time,
        gurobi_seconds=g_time,
        scip_version=s_version,
        gurobi_version=g_version,
    )
