"""Optional genuine transpilation; requires `pip install '.[qiskit]'`.

Compilation targets unconnected all-to-all CX topology unless coupled by caller.
Use counts only under matched transpiler version, gate set, optimization_level.
"""
from __future__ import annotations
from .instances import IsingInstance
from .simulator import Angles


def circuit(inst: IsingInstance, ang: Angles):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("Install qiskit extra for circuit export") from exc
    qc = QuantumCircuit(inst.n)
    for i in range(inst.n): qc.h(i)
    for gamma, beta, alpha in zip(ang.gamma, ang.beta, ang.alpha):
        for (i,j), w in zip(inst.edges, inst.weights):
            qc.rzz(float(2*gamma*w), int(i), int(j))
        if abs(alpha) > 1e-12:
            for (i,j),w in zip(inst.edges, inst.weights):
                for y,z in ((int(i),int(j)),(int(j),int(i))):
                    # V=H S† maps Y->Z; V† exp(-i theta ZZ) V
                    qc.sdg(y)
                    qc.h(y)
                    qc.rzz(float(2*alpha*w),y,z)
                    qc.h(y)
                    qc.s(y)
        for i in range(inst.n): qc.rx(float(2*beta),i)
    return qc


def transpile_cost(inst: IsingInstance, ang: Angles, optimization_level=1):
    from qiskit import transpile
    qc = transpile(circuit(inst, ang), basis_gates=["cx","rz","sx","x"],
                   optimization_level=optimization_level, seed_transpiler=20261011)
    ops=qc.count_ops()
    return {"cx": int(ops.get("cx",0)), "depth": qc.depth(),
            "size": qc.size(), "optimization_level": optimization_level,
            "basis": ["cx","rz","sx","x"]}
