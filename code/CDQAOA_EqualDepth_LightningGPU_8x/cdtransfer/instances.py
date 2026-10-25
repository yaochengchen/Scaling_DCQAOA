"""Ising instances and leakage-resistant deterministic split generation.

Convention: H_C = sum_edges J[i,j] Z_i Z_j; J>0 rewards cut edges.
Vertex i corresponds to bit i (little-endian statevector).
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class IsingInstance:
    n: int
    edges: np.ndarray  # m x 2, indices sorted
    weights: np.ndarray
    family: str = "custom"
    weight_kind: str = "custom"
    seed: int = -1
    split: str = "custom"

    def __post_init__(self):
        e = np.asarray(self.edges, dtype=np.int32).reshape(-1, 2)
        w = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if self.n < 2 or e.shape[0] != w.size or w.size == 0:
            raise ValueError("Must have n>=2 and matching nonempty edges and weights")
        if np.any(e[:, 0] == e[:, 1]) or np.any(e < 0) or np.any(e >= self.n):
            raise ValueError("Edges must be distinct vertices in [0, n)")
        e = np.sort(e, axis=1)
        idx = np.lexsort((e[:, 1], e[:, 0]))
        e, w = e[idx], w[idx]
        if len(set(map(tuple, e))) != len(e):
            raise ValueError("Duplicate undirected edges")
        if not np.isfinite(w).all() or not np.any(w):
            raise ValueError("Weights must be finite and not all zero")
        e.setflags(write=False)
        w.setflags(write=False)
        object.__setattr__(self, "edges", e)
        object.__setattr__(self, "weights", w)

    @property
    def m(self) -> int:
        return len(self.weights)

    def to_dict(self) -> dict:
        return {"n": self.n, "edges": self.edges.tolist(), "weights": self.weights.tolist(),
                "family": self.family, "weight_kind": self.weight_kind,
                "seed": self.seed, "split": self.split}

    @classmethod
    def from_dict(cls, obj: dict) -> "IsingInstance":
        return cls(**obj)

    def scaled(self, scale: float) -> "IsingInstance":
        if scale <= 0:
            raise ValueError("Weight scale must be positive")
        return IsingInstance(self.n, self.edges.copy(), self.weights * scale,
                             self.family, self.weight_kind, self.seed, self.split)

    @property
    def fingerprint(self) -> str:
        payload = {"n": self.n, "edges": self.edges.tolist(), "weights": self.weights.tolist()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def features(inst: IsingInstance) -> dict:
    w = inst.weights
    s = float(np.sqrt(np.mean(w * w)))
    mu = float(np.mean(np.abs(w)))
    return {"n": inst.n, "m": inst.m, "degree": 2.0 * inst.m / inst.n,
            "rms": s, "cv_abs": float(np.std(np.abs(w)) / max(mu, 1e-12)),
            "positive_fraction": float(np.mean(w > 0)),
            "mean_weight": float(np.mean(w)),
            "max_abs": float(np.max(np.abs(w)))}


def make_instance(n: int, family: str, weight_kind: str, seed: int, split="train") -> IsingInstance:
    rng = np.random.default_rng(int(seed))
    if family == "tree":
        if hasattr(nx, "random_labeled_tree"):
            graph = nx.random_labeled_tree(n, seed=int(seed))
        else:
            graph = nx.random_tree(n, seed=int(seed))
    elif family.startswith("regular"):
        d = int(family.replace("regular", ""))
        if n <= d or (n * d) % 2:
            raise ValueError(f"Invalid n={n}, d={d} regular graph")
        graph = nx.random_regular_graph(d, n, seed=int(seed))
    elif family == "er3":
        graph = nx.gnp_random_graph(n, min(3.0 / (n - 1), 1.0), seed=int(seed))
    elif family == "er6":
        graph = nx.gnp_random_graph(n, min(6.0 / (n - 1), 1.0), seed=int(seed))
    elif family in ("er05", "community", "bipartite"):
        if family == "er05":
            graph = nx.gnp_random_graph(n, 0.5, seed=int(seed))
        elif family == "community":
            graph = nx.stochastic_block_model([n//2,n-n//2],[[0.45,0.08],[0.08,0.45]],seed=int(seed))
        else:
            graph = nx.bipartite.random_graph(n//2,n-n//2,0.35,seed=int(seed))
    elif family == "complete":
        graph = nx.complete_graph(n)
    elif family == "ring":
        graph = nx.cycle_graph(n)
    else:
        raise ValueError(f"Unknown graph family: {family}")
    if graph.number_of_edges() == 0:
        # rare empty ER draw: retry with a deterministic ring so no invalid energy evaluator
        graph = nx.cycle_graph(n)
    # Random vertex relabeling prevents literal duplicates for fixed topologies
    # (e.g. unit-weight ring); isomorphic-equivalent instances remain possible.
    permutation = rng.permutation(n)
    edges = np.asarray(sorted(tuple(sorted((int(permutation[i]), int(permutation[j]))))
                              for i,j in graph.edges()), dtype=np.int32)
    m = len(edges)
    if weight_kind == "unit":
        w = np.ones(m)
    elif weight_kind == "uniform":
        w = rng.uniform(0.5, 1.5, m)
    elif weight_kind == "bimodal":
        w = rng.choice([0.5, 1.5], m)
    elif weight_kind == "lognormal":
        w = np.clip(rng.lognormal(0.0, 0.65, m), 0.2, 5.0)
    elif weight_kind == "signed_normal":
        w = rng.normal(0.0, 1.0, m)
    elif weight_kind == "heavy_tail":
        w = np.clip(rng.lognormal(0.0, 1.2, m), 0.2, 10.0)
    else:
        raise ValueError(f"Unknown weight distribution: {weight_kind}")
    if family == "complete":
        # SK-like extensive normalization; intentional n dependence
        w = w / np.sqrt(n)
    return IsingInstance(n, edges, w, family, weight_kind, int(seed), split)


def make_splits(config: dict) -> dict[str, list[dict]]:
    """Independent seeds per split. Held-out families/weights determined by config."""
    base = int(config.get("seed", 20261011))
    splits: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for offset, split in enumerate(("train", "val", "test")):
        spec = config["datasets"][split]
        n_values = spec["n"]
        families = spec["families"]
        weights = spec["weights"]
        samples = int(spec["samples_per_cell"])
        batch: list[dict] = []
        for n in n_values:
            for family in families:
                if family.startswith("regular") and (n <= int(family[7:]) or n * int(family[7:]) % 2):
                    continue
                for weight_kind in weights:
                    for j in range(samples):
                        # deterministic no-overlap seeds even if n/family/weight parameters overlap
                        token = f"{base}:{offset}:{n}:{family}:{weight_kind}:{j}".encode()
                        seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
                        inst = make_instance(n, family, weight_kind, seed, split)
                        if inst.fingerprint in seen:
                            raise RuntimeError("Duplicate graph/weights across splits")
                        seen.add(inst.fingerprint)
                        batch.append({"id": f"{split}-{len(batch):05d}", **inst.to_dict(),
                                      "fingerprint": inst.fingerprint})
        if not batch:
            raise ValueError(f"Empty dataset split: {split}")
        splits[split] = batch
    return splits


def dump_json(path: str | Path, data: object):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.floating): return float(value)
        if isinstance(value, np.bool_): return bool(value)
        if isinstance(value, np.ndarray): return value.tolist()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                              sort_keys=True, default=convert, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Equal-depth GPU study helpers (stable IDs, richer static features, exact E_min)
# --------------------------------------------------------------------------
def stable_id(split: str, n: int, family: str, weight_kind: str, j: int) -> str:
    """ID independent of how many samples per cell are requested.

    Instance j of a cell is identical for samples_per_cell=20 or 50, so a
    20-sample run can be extended to 50 without recomputing (see --reuse-from).
    """
    return f"{split}-n{n}-{family}-{weight_kind}-{j:03d}"


def make_splits_stable(config: dict) -> dict[str, list[dict]]:
    base = int(config.get("seed", 20261011))
    splits: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for offset, split in enumerate(("train", "val", "test")):
        spec = config["datasets"][split]
        batch: list[dict] = []
        for n in spec["n"]:
            for family in spec["families"]:
                if family.startswith("regular") and (n <= int(family[7:]) or n * int(family[7:]) % 2):
                    continue
                for weight_kind in spec["weights"]:
                    for j in range(int(spec["samples_per_cell"])):
                        token = f"{base}:{offset}:{n}:{family}:{weight_kind}:{j}".encode()
                        seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
                        inst = make_instance(n, family, weight_kind, seed, split)
                        if inst.fingerprint in seen:
                            raise RuntimeError("Duplicate graph/weights across splits")
                        seen.add(inst.fingerprint)
                        batch.append({"id": stable_id(split, n, family, weight_kind, j),
                                      **inst.to_dict(), "fingerprint": inst.fingerprint,
                                      "graph_features": graph_features(inst)})
        if not batch:
            raise ValueError(f"Empty dataset split: {split}")
        splits[split] = batch
    return splits


def graph_features(inst: IsingInstance) -> dict:
    """Static, target-instance-only features for later formula search.

    No quantum objective is evaluated. `features()` (used by the frozen V1
    formula) is left unchanged; this dict only adds candidate predictors.
    """
    n, e, w = inst.n, inst.edges, inst.weights
    deg = np.bincount(e.ravel(), minlength=n).astype(float)
    wdeg = np.zeros(n)
    np.add.at(wdeg, e[:, 0], np.abs(w))
    np.add.at(wdeg, e[:, 1], np.abs(w))
    A = np.zeros((n, n))
    A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1.0
    W = np.zeros((n, n))
    W[e[:, 0], e[:, 1]] = W[e[:, 1], e[:, 0]] = w
    lap = np.diag(np.abs(W).sum(1)) - np.abs(W)
    lap_eig = np.linalg.eigvalsh(lap)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, e.tolist()))
    tri = sum(nx.triangles(graph).values()) // 3
    aw = np.abs(w)
    rms = float(np.sqrt(np.mean(w * w)))
    mu = float(np.mean(aw))
    sd = float(np.std(aw))
    return {
        "n": n, "m": inst.m, "density": 2.0 * inst.m / (n * (n - 1)),
        "degree_mean": float(deg.mean()), "degree_std": float(deg.std()),
        "degree_min": float(deg.min()), "degree_max": float(deg.max()),
        "degree_cv": float(deg.std() / max(deg.mean(), 1e-12)),
        "weighted_degree_mean": float(wdeg.mean()), "weighted_degree_std": float(wdeg.std()),
        "clustering": float(nx.average_clustering(graph)), "triangles": int(tri),
        "components": int(nx.number_connected_components(graph)),
        "is_bipartite": bool(nx.is_bipartite(graph)),
        "adjacency_spectral_radius": float(np.max(np.abs(np.linalg.eigvalsh(A)))),
        "weighted_spectral_radius": float(np.max(np.abs(np.linalg.eigvalsh(W)))),
        "laplacian_max": float(lap_eig[-1]), "algebraic_connectivity": float(lap_eig[1]),
        "weight_rms": rms, "weight_mean_abs": mu, "weight_cv": sd / max(mu, 1e-12),
        "weight_skew": float(np.mean((aw - mu) ** 3) / max(sd, 1e-12) ** 3),
        "weight_max_over_rms": float(aw.max() / max(rms, 1e-12)),
        "weight_mean_over_rms": mu / max(rms, 1e-12),
        "positive_fraction": float(np.mean(w > 0)),
        "sum_abs_weights": float(aw.sum()),
    }


def exact_extrema(inst: IsingInstance, chunk_bits: int = 20) -> tuple[float, float]:
    """Exact (min, max) of H_C = sum J Z_i Z_j by exhaustive enumeration of 2^n spins.

    CPU only, chunked (2^chunk_bits states at a time); n=24 takes seconds.
    The global spin flip is a symmetry, so only states with spin n-1 = +1
    (bit n-1 = 0) are enumerated.
    """
    n = inst.n
    total = 1 << (n - 1)
    step = 1 << min(chunk_bits, n - 1)
    lo, hi = np.inf, -np.inf
    ii = inst.edges[:, 0].astype(np.int64)
    jj = inst.edges[:, 1].astype(np.int64)
    w = inst.weights
    for start in range(0, total, step):
        idx = np.arange(start, min(start + step, total), dtype=np.int64)
        energy = np.zeros(len(idx))
        for i, j, wij in zip(ii, jj, w):
            energy += wij * (1.0 - 2.0 * (((idx >> i) ^ (idx >> j)) & 1))
        lo = min(lo, float(energy.min()))
        hi = max(hi, float(energy.max()))
    return lo, hi
