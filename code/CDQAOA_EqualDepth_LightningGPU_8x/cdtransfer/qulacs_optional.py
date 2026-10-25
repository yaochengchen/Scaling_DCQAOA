"""Optional independent Qulacs CPU/GPU-compatible circuit exporter.

IMPORTANT: Qulacs PauliRotation(theta) uses exp(+i theta P/2) in the
convention used here, hence theta=-2*coefficient for exp(-i coeff P).
Validate a small matrix case and the optional cross-backend test when installed.
"""
from __future__ import annotations
from .instances import IsingInstance
from .simulator import Angles


def state(inst: IsingInstance, angles: Angles):
    try:
        from qulacs import QuantumState, QuantumCircuit
        from qulacs.gate import PauliRotation, H
    except ImportError as exc:
        raise ImportError("Install 'qulacs' for optional independent backend validation") from exc
    q=QuantumCircuit(inst.n)
    for i in range(inst.n): q.add_gate(H(i))
    for gamma,beta,alpha in zip(angles.gamma,angles.beta,angles.alpha):
        for (i,j),w in zip(inst.edges,inst.weights):
            q.add_gate(PauliRotation([int(i),int(j)],[3,3],float(-2*gamma*w)))
        if abs(alpha)>1e-12:
            for (i,j),w in zip(inst.edges,inst.weights):
                q.add_gate(PauliRotation([int(i),int(j)],[2,3],float(-2*alpha*w)))
                q.add_gate(PauliRotation([int(i),int(j)],[3,2],float(-2*alpha*w)))
        for i in range(inst.n):
            q.add_gate(PauliRotation([i],[1],float(-2*beta)))
    s=QuantumState(inst.n)
    q.update_quantum_state(s)
    return s.get_vector()
