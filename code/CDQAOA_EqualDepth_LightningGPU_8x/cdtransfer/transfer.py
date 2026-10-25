"""Fixed-angle predictors. Their inference NEVER invokes an energy evaluator.

Original CSV is shipped verbatim under assets/original_transfer_data.csv.
Angle conventions are explicit; input target contains STATIC instance features only.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .instances import IsingInstance, features
from .simulator import Angles

SOURCE_CSV = Path(__file__).resolve().parents[1] / "assets" / "original_transfer_data.csv"


def load_source_angles(p: int, q: int = 2, path: str | Path = SOURCE_CSV) -> Angles:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for row in rows[2:]:
        if len(row) < 3 or not row[0].strip() or not row[1].strip():
            continue
        if int(float(row[0])) == q and int(float(row[1])) == p:
            raw = np.asarray([float(t) for t in row[3:3+2*p]], dtype=float)
            if len(raw) != 2*p:
                raise ValueError("Malformed CSV row: gamma and beta entries missing")
            return Angles(raw[:p], raw[p:], np.zeros(p))
    raise KeyError(f"Missing q={q}, p={p} in {path}")


def degree_factor(degree: float) -> float:
    return 1.0 if degree <= 1.0 else float(np.arctan(1.0 / np.sqrt(degree - 1.0)))


def qaoa_csv(inst: IsingInstance, p: int, sign: int = -1,
             scale: str = "rms_degree") -> Angles:
    """Legacy derived transfer with CORRECT CSV layout.

    Historical MindQuantum scoring used exp(+i gamma_code H_C); our circuit
    is exp(-i gamma H_C), hence sign=-1 by default. The source CSV's native
    Hamiltonian convention may differ, so sign must be selected on *validation*
    data, NOT on test instances. Also tests alternative +1.
    """
    ref = load_source_angles(p)
    f = features(inst)
    denom = f["rms"] if scale in ("rms", "rms_degree") else 1.0
    df = degree_factor(f["degree"]) if scale == "rms_degree" else 1.0
    return Angles(sign * ref.gamma * df / denom, ref.beta, np.zeros(p))


def learned_angles(inst: IsingInstance, p: int, template: dict,
                   cd: bool = True, alpha_degree_power: float = 0.0,
                   alpha_multiplier: float = 1.0) -> Angles:
    """Templates stored at unit RMS: gamma_unscaled=gamma*rms/degree_factor,
    alpha_unscaled=alpha*rms (optional degree correction for alpha).
    """
    entry = template[str(p)]
    if isinstance(entry, dict) and "qaoa" in entry:
        entry = entry["cd" if cd else "qaoa"]
    f = features(inst)
    s = f["rms"]
    df = degree_factor(f["degree"])
    d0 = float(entry.get("reference_degree", f["degree"]))
    degree_ratio = max(f["degree"], 0.1) / max(d0, 0.1)
    gamma = np.asarray(entry["gamma_unit"], float) * df / s
    beta = np.asarray(entry["beta"], float)
    alpha = (np.asarray(entry["alpha_unit"], float) / s) * \
            (degree_ratio ** alpha_degree_power) * alpha_multiplier
    return Angles(gamma, beta, alpha if cd else np.zeros(p))


def bch_angles(inst: IsingInstance, p: int, c: float = -0.5, sign: int = -1) -> Angles:
    ref = qaoa_csv(inst, p, sign)
    return Angles(ref.gamma, ref.beta, c * ref.gamma * ref.beta)


def predict(inst: IsingInstance, p: int, method: str, model: dict | None = None) -> Angles:
    """Production inference API. Absolutely no simulator or energy evaluation.

    Since this module imports only instance/angle datatypes, monitoring tests
    can enforce no hidden per-instance optimization.
    """
    model = model or {}
    sign = int(model.get("csv_sign", -1))
    if method == "qaoa_csv":
        return qaoa_csv(inst, p, sign=sign, scale="rms_degree")
    if method == "qaoa_csv_rms":
        return qaoa_csv(inst, p, sign=sign, scale="rms")
    if method == "qaoa_trained":
        return learned_angles(inst, p, model["templates"], cd=False)
    if method == "cd_bch":
        return bch_angles(inst, p, c=float(model.get("bch_c", -0.5)), sign=sign)
    if method == "cd_reference":
        return learned_angles(inst, p, model["templates"], cd=True,
                              alpha_degree_power=0.0)
    if method == "cd_proposed":
        return learned_angles(inst, p, model["templates"], cd=True,
                              alpha_degree_power=float(model.get("alpha_degree_power", 0.0)),
                              alpha_multiplier=float(model.get("alpha_multiplier", 1.0)))
    if method == "cd_zero":
        return qaoa_csv(inst, p, sign=sign)
    raise ValueError(f"Unknown method {method!r}")
