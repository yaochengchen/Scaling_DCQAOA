"""PennyLane Lightning backend (lightning.gpu on the server, lightning.qubit for CPU tests).

Same circuit and conventions as simulator.StatevectorSimulator:
  |+>^n, then per layer  U_C(gamma) -> U_CD(alpha) -> U_B(beta)
  U_C  = prod_e exp(-i gamma J_e Z_i Z_j)
  U_CD = prod_e exp(-i alpha J_e Y_i Z_j) exp(-i alpha J_e Z_i Y_j)   (fixed edge order)
  U_B  = prod_q exp(-i beta X_q)
Energy <H_C>, H_C = sum_e J_e Z_i Z_j; gradient by the adjoint method (no state transfer to host).
Parameter vector: [gamma(p), beta(p), alpha(p) if cd].
"""
from __future__ import annotations
import numpy as np
from .simulator import Angles


class LightningSimulator:
    def __init__(self, instance, *, device="lightning.gpu", dtype="complex128"):
        import pennylane as qml
        from autograd import value_and_grad
        self.inst, self.n, self.qml = instance, instance.n, qml
        self.dev = qml.device(device, wires=self.n, c_dtype=getattr(np, dtype))
        self.hamiltonian = qml.Hamiltonian(instance.weights.tolist(),
                                           [qml.Z(int(i)) @ qml.Z(int(j)) for i, j in instance.edges])
        self._value_and_grad = value_and_grad
        self._grad_nodes, self._fwd_nodes = {}, {}

    def _circuit(self, theta, p, cd):
        qml = self.qml
        for i in range(self.n):
            qml.Hadamard(i)
        for layer in range(p):
            for (i, j), w in zip(self.inst.edges, self.inst.weights):
                qml.IsingZZ(2 * w * theta[layer], wires=[int(i), int(j)])
            if cd:   # never skip alpha=0: the derivative at alpha=0 is needed
                for (i, j), w in zip(self.inst.edges, self.inst.weights):
                    qml.PauliRot(2 * w * theta[2 * p + layer], "YZ", wires=[int(i), int(j)])
                    qml.PauliRot(2 * w * theta[2 * p + layer], "ZY", wires=[int(i), int(j)])
            for i in range(self.n):
                qml.RX(2 * theta[p + layer], wires=i)

    def value_and_grad(self, theta, p, cd=True):
        key = (p, cd)
        if key not in self._grad_nodes:
            @self.qml.qnode(self.dev, interface="autograd", diff_method="adjoint", cache=False)
            def energy(t):
                self._circuit(t, p, cd)
                return self.qml.expval(self.hamiltonian)
            self._grad_nodes[key] = self._value_and_grad(energy)
        value, grad = self._grad_nodes[key](self.qml.numpy.array(theta, requires_grad=True))
        return float(value), np.asarray(grad, dtype=float)

    def evaluate(self, angles: Angles) -> float:
        cd = bool(np.any(angles.alpha != 0))
        key = (angles.p, cd)
        if key not in self._fwd_nodes:
            p = angles.p
            @self.qml.qnode(self.dev, interface=None, diff_method=None, cache=False)
            def energy(t):
                self._circuit(t, p, cd)
                return self.qml.expval(self.hamiltonian)
            self._fwd_nodes[key] = energy
        return float(self._fwd_nodes[key](angles.pack(include_cd=cd)))

    def state(self, angles: Angles) -> np.ndarray:
        """Correctness checks only: copies the full state to the host, little-endian like NumPy."""
        cd = bool(np.any(angles.alpha != 0))
        @self.qml.qnode(self.dev, interface=None, diff_method=None)
        def state(t):
            self._circuit(t, angles.p, cd)
            return self.qml.state()
        psi = np.asarray(state(angles.pack(include_cd=cd)))
        return psi.reshape([2] * self.n).transpose(list(reversed(range(self.n)))).reshape(-1)


class LightningTapeSimulator:
    """Same circuit, energy and adjoint gradient as LightningSimulator, without the per-call
    PennyLane Python layer (QNode construction, autograd tracing, diff-method resolution and
    re-decomposition of every PauliRot).  For each (p, cd) the tape is built ONCE from natively
    supported, differentiable Lightning gates; a call only rebinds the gate angles and runs
    Lightning's C++ adjoint (`execute_and_compute_derivatives`).  The layer gradient follows by
    the chain rule  d/dtheta_k = sum_g (d angle_g / d theta_k) * d E / d angle_g.

    exp(-i a Y_i Z_j) = V_i exp(-i a Z_i Z_j) V_i^dagger with V = SX^dagger (V Z V^dagger = Y),
    i.e. SX(i) -> IsingZZ(2a)(i,j) -> SX^dagger(i); identical unitary to PauliRot(2a, "YZ").
    """

    def __init__(self, instance, *, device="lightning.gpu", dtype="complex128"):
        import pennylane as qml
        from pennylane.devices import ExecutionConfig
        self.inst, self.n, self.qml = instance, instance.n, qml
        self.dev = qml.device(device, wires=self.n, c_dtype=getattr(np, dtype))
        self.hamiltonian = qml.Hamiltonian(instance.weights.tolist(),
                                           [qml.Z(int(i)) @ qml.Z(int(j)) for i, j in instance.edges])
        self._grad_cfg = self.dev.setup_execution_config(ExecutionConfig(gradient_method="adjoint"))
        self._fwd_cfg = self.dev.setup_execution_config(ExecutionConfig())
        self._tapes = {}

    def _build(self, p, cd):
        """Tape with placeholder angles + linear map angles = A @ theta (theta packed as usual)."""
        qml = self.qml
        edges = [(int(i), int(j)) for i, j in self.inst.edges]
        w = [float(v) for v in self.inst.weights]
        ops = [qml.Hadamard(i) for i in range(self.n)]
        coef = []      # (theta index, factor) for each parametrized gate, in tape order
        for layer in range(p):
            for (i, j), wij in zip(edges, w):
                ops.append(qml.IsingZZ(0.0, wires=[i, j]))
                coef.append((layer, 2 * wij))
            if cd:
                for (i, j), wij in zip(edges, w):
                    for y, z in ((i, j), (j, i)):            # YZ then ZY, fixed edge order
                        ops.append(qml.SX(y))
                        ops.append(qml.IsingZZ(0.0, wires=[y, z]))
                        ops.append(qml.adjoint(qml.SX(y)))
                        coef.append((2 * p + layer, 2 * wij))
            for i in range(self.n):
                ops.append(qml.RX(0.0, wires=i))
                coef.append((p + layer, 2.0))
        tape = qml.tape.QuantumScript(ops, [qml.expval(self.hamiltonian)])
        npar = (3 if cd else 2) * p
        A = np.zeros((len(coef), npar))
        for g, (k, f) in enumerate(coef):
            A[g, k] = f
        # Gate angles come first in the tape's parameter list; Hamiltonian coefficients
        # (fixed data) follow and are excluded from differentiation.
        n_gate = len(coef)
        tape.trainable_params = list(range(n_gate))
        return tape, A, n_gate

    def _tape(self, theta, p, cd):
        key = (p, cd)
        if key not in self._tapes:
            self._tapes[key] = self._build(p, cd)
        tape, A, n_gate = self._tapes[key]
        angles = A @ np.asarray(theta, dtype=float)
        return tape.bind_new_parameters(list(angles), list(range(n_gate))), A

    def value_and_grad(self, theta, p, cd=True):
        tape, A = self._tape(theta, p, cd)
        values, jacs = self.dev.execute_and_compute_derivatives((tape,), self._grad_cfg)
        jac = np.asarray(jacs[0], dtype=float).reshape(-1)
        return float(np.asarray(values[0])), A.T @ jac

    def evaluate(self, angles: Angles) -> float:
        cd = bool(np.any(angles.alpha != 0))
        tape, _ = self._tape(angles.pack(include_cd=cd), angles.p, cd)
        return float(np.asarray(self.dev.execute((tape,), self._fwd_cfg)[0]))

    def state(self, angles: Angles) -> np.ndarray:
        cd = bool(np.any(angles.alpha != 0))
        tape, _ = self._tape(angles.pack(include_cd=cd), angles.p, cd)
        st = self.qml.tape.QuantumScript(tape.operations, [self.qml.state()])
        psi = np.asarray(self.dev.execute((st,), self._fwd_cfg)[0])
        return psi.reshape([2] * self.n).transpose(list(reversed(range(self.n)))).reshape(-1)


def make_simulator(inst, settings):
    """`settings["engine"]`: "tape" (default, fast path) or "qnode" (plain QNode + autograd)."""
    backend = settings.get("backend", "lightning.gpu")
    if backend not in ("lightning.gpu", "lightning.qubit"):
        raise ValueError(f"Unsupported backend for the GPU study: {backend}")
    cls = LightningSimulator if settings.get("engine", "tape") == "qnode" else LightningTapeSimulator
    return cls(inst, device=backend, dtype=settings.get("dtype", "complex128"))
