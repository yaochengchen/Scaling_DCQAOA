"""PennyLane implementation of the project's cost→CD→mixer circuit."""
from __future__ import annotations

import pennylane as qml

from cdtransfer.instances import IsingInstance
from cdtransfer.simulator import Angles


def make_pennylane_evaluator(inst: IsingInstance):
    """Build one noiseless QNode per graph, reusable across angle evaluations."""
    device = qml.device("lightning.qubit", wires=inst.n, shots=None)
    terms = [qml.PauliZ(int(i)) @ qml.PauliZ(int(j)) for i, j in inst.edges]
    observable = qml.Hamiltonian([float(weight) for weight in inst.weights], terms)

    @qml.qnode(device, interface="numpy", diff_method=None)
    def circuit(gamma, beta, alpha):
        for wire in range(inst.n):
            qml.Hadamard(wires=wire)
        for gamma_layer, beta_layer, alpha_layer in zip(gamma, beta, alpha):
            for (i, j), weight in zip(inst.edges, inst.weights):
                qml.PauliRot(float(2 * gamma_layer * weight), "ZZ", wires=[int(i), int(j)])
            for (i, j), weight in zip(inst.edges, inst.weights):
                qml.PauliRot(float(2 * alpha_layer * weight), "YZ", wires=[int(i), int(j)])
                qml.PauliRot(float(2 * alpha_layer * weight), "ZY", wires=[int(i), int(j)])
            for wire in range(inst.n):
                qml.RX(float(2 * beta_layer), wires=wire)
        return qml.expval(observable)

    def evaluate(angles: Angles) -> float:
        return float(circuit(angles.gamma, angles.beta, angles.alpha))

    return evaluate


def energy_pennylane(inst: IsingInstance, angles: Angles) -> float:
    return make_pennylane_evaluator(inst)(angles)
