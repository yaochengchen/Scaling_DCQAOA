"""Pure NumPy exact-statevector backend, for n<=~20 CPU pilot studies.

State index bit i is qubit i. Layer: U_B(beta) U_CD(alpha) U_C(gamma).
U_CD is FIXED-ORDER product of exp(-i alpha*J Y_i Z_j) then
exp(-i alpha*J Z_i Y_j) for each lexicographically sorted edge.
No claim this Trotter circuit equals exact exp(-i alpha*H_CD).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .instances import IsingInstance


@lru_cache(maxsize=32)
def _indices_zero(n: int, bit: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(1 << n, dtype=np.int64)
    zero = idx[(idx & (1 << bit)) == 0]
    one = zero | (1 << bit)
    return zero, one


@dataclass(frozen=True)
class Angles:
    gamma: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray

    def __post_init__(self):
        g, b, a = [np.asarray(v, dtype=np.float64).reshape(-1).copy() for v in
                   (self.gamma, self.beta, self.alpha)]
        if not (len(g) == len(b) == len(a) and len(g) > 0):
            raise ValueError("Each angle vector must have the same positive length")
        if not all(np.isfinite(v).all() for v in (g, b, a)):
            raise ValueError("Angles must be finite")
        object.__setattr__(self, "gamma", g)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "alpha", a)

    @property
    def p(self): return len(self.gamma)

    def pack(self, include_cd=True) -> np.ndarray:
        return np.concatenate([self.gamma, self.beta, self.alpha] if include_cd
                              else [self.gamma, self.beta])

    def to_dict(self):
        return {"gamma": self.gamma.tolist(), "beta": self.beta.tolist(), "alpha": self.alpha.tolist()}

    @classmethod
    def unpack(cls, theta: np.ndarray, p: int, include_cd=True):
        theta = np.asarray(theta, dtype=float)
        return cls(theta[:p], theta[p:2*p], theta[2*p:3*p] if include_cd else np.zeros(p))


class StatevectorSimulator:
    """Reusable energy-diagonal and Pauli masks for fixed instance."""
    def __init__(self, instance: IsingInstance):
        self.inst = instance
        self.n = instance.n
        if self.n > 23:
            raise ValueError("NumPy reference backend intended for n<=23. Use GPU later.")
        idx = np.arange(1 << self.n, dtype=np.int64)
        energy = np.zeros(len(idx), dtype=np.float64)
        for (i, j), w in zip(instance.edges, instance.weights):
            zi = 1.0 - 2.0 * ((idx >> int(i)) & 1)
            zj = 1.0 - 2.0 * ((idx >> int(j)) & 1)
            energy += float(w) * zi * zj
        self.energy_vector = energy
        self._xy = []
        for (i, j), w in zip(instance.edges, instance.weights):
            i, j = int(i), int(j)
            # Both CD rotations: first Yi Zj, then Zi Yj
            for y, z in ((i, j), (j, i)):
                zero, one = _indices_zero(self.n, y)
                zsign = 1.0 - 2.0 * ((zero >> z) & 1)
                self._xy.append((zero, one, zsign, float(w)))

    def state(self, angles: Angles) -> np.ndarray:
        psi = np.ones(1 << self.n, dtype=np.complex128) / np.sqrt(1 << self.n)
        for gamma, beta, alpha in zip(angles.gamma, angles.beta, angles.alpha):
            psi *= np.exp(-1j * gamma * self.energy_vector)
            if abs(alpha) > 1e-15:
                for zero, one, zsign, weight in self._xy:
                    theta = alpha * weight
                    co, si = np.cos(theta), np.sin(theta)
                    a = psi[zero].copy()
                    b = psi[one].copy()
                    psi[zero] = co * a - si * zsign * b
                    psi[one] = si * zsign * a + co * b
            co, si = np.cos(beta), np.sin(beta)
            for bit in range(self.n):
                zero, one = _indices_zero(self.n, bit)
                a, b = psi[zero].copy(), psi[one].copy()
                psi[zero] = co * a - 1j * si * b
                psi[one] = co * b - 1j * si * a
        return psi

    def evaluate(self, angles: Angles) -> float:
        psi = self.state(angles)
        return float(np.dot(np.abs(psi)**2, self.energy_vector))

    def sample_energies(self, angles: Angles, shots: int, seed: int) -> np.ndarray:
        psi = self.state(angles)
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(psi), shots, p=np.abs(psi)**2)
        return self.energy_vector[selected]

    def brute_force_min(self) -> float:
        return float(np.min(self.energy_vector))


def cost_and_evaluator(inst: IsingInstance):
    return StatevectorSimulator(inst)
