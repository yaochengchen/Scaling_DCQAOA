"""Explicit unoptimized CX+one-qubit gate decomposition budgets.

Each exp(-i theta Pauli_i Pauli_j) is one basis-change block + 2 CX
and one RZ. No algebraic CX cancellation or routing is assumed. This is
NOT a hardware-optimized transpiler count; see optional Qiskit exporter.
"""
from __future__ import annotations
from .instances import IsingInstance
from .simulator import Angles


def gate_cost(inst: IsingInstance, angles: Angles) -> dict:
    p = angles.p
    cost_rot = inst.m * p
    cd_rot = 2*inst.m*sum(abs(v)>1e-12 for v in angles.alpha)
    two = 2*(cost_rot + cd_rot)
    return {"n": inst.n, "m": inst.m, "p": p,
            "zz_rotations": cost_rot, "cd_rotations": cd_rot,
            "decomposed_cx": two, "model": "2 CX per ZZ/YZ/ZY, no cancels or routing",
            "has_cd": bool(cd_rot)}


def comparable_qaoa_depth(cd_p: int) -> int:
    # Fixed decomposition: QAOA 2*m*p' vs CD 6*m*p
    return 3 * cd_p
