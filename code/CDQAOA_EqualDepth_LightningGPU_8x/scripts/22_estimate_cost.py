"""Estimate GPU wall time of a same_depth config from a finished timing pilot.

    python scripts/22_estimate_cost.py --pilot results/eqdepth_pilot_timing \
        --config configs/eqdepth_full50.json --gpus 8

Model (fitted on the pilot's own records, nothing assumed; run the pilot with the same
--slots-per-gpu / --task-weights you will use, because the measured times include GPU sharing):
  seconds per energy+gradient call  t(n, G) = c0(n) + c1(n) * G,  G = p*(m + 2m*[CD] + n) gates
  calls per optimization            mean eval_calls of the pilot for the same (p, method)
For n not in the pilot, c1 is scaled from the nearest measured n with the per-qubit growth
measured between pilot sizes (capped at x2, the statevector limit); treat as rough.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdtransfer.instances import make_instance


def gates(p, m, n, cd):
    return p * (m + (2 * m if cd else 0) + n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    args = ap.parse_args()
    recs = [json.loads(l) for l in (Path(args.pilot) / "offline_optima.jsonl").open() if l.strip()]
    by_n = defaultdict(list)
    calls = defaultdict(list)
    share = {}      # fraction of one GPU a task of size n occupied in the pilot
    for r in recs:
        if r["eval_calls"] > 0 and not r.get("imported_from"):
            by_n[r["n"]].append((gates(r["p"], r["m"], r["n"], r["cd"]), r["wall_seconds"] / r["eval_calls"]))
            calls[(r["p"], r["cd"])].append(r["eval_calls"])
            if r.get("gpu_slots_used") and r.get("gpu_slots_total"):
                share[r["n"]] = r["gpu_slots_used"] / r["gpu_slots_total"]
    coef = {}
    for n, pts in by_n.items():
        g, t = np.asarray(pts).T
        c1, c0 = np.polyfit(g, t, 1) if len(set(g)) > 1 else (t.mean() / g.mean(), 0.0)
        coef[n] = (max(c0, 0.0), max(c1, 1e-12))
        print(f"pilot n={n}: per-call {coef[n][0]*1e3:.2f} ms + {coef[n][1]*1e6:.2f} us/gate  "
              f"({len(pts)} optimizations)")
    nmax = max(coef)
    if len(coef) >= 2:   # growth rate of the per-gate cost measured between pilot sizes
        ns = sorted(coef)
        slope = np.polyfit(ns, np.log([coef[k][1] for k in ns]), 1)[0]
        growth = float(np.clip(np.exp(slope), 1.0, 2.0))
    else:
        growth = 2.0
    print(f"per-gate cost growth per qubit used for unmeasured n: x{growth:.2f}")
    cfg = json.loads(Path(args.config).read_text())
    depths, restarts = cfg["depths"], cfg["optimization"]["restarts"]
    spec = cfg["datasets"]["train"]
    total = 0.0
    table = defaultdict(float)
    for n in spec["n"]:
        near = min(coef, key=lambda k: abs(k - n))
        c0, c1 = coef.get(n, (coef[near][0], coef[near][1] * growth ** (n - near)))
        for fam in spec["families"]:
            if fam.startswith("regular") and (n <= int(fam[7:]) or n * int(fam[7:]) % 2):
                continue
            m = np.mean([make_instance(n, fam, "unit", 1000 + s).m for s in range(5)])
            for _ in spec["weights"]:
                for p in depths:
                    for cd in (False, True):
                        k = calls.get((p, cd)) or calls[min((q for q in calls if q[1] == cd), key=lambda q: abs(q[0] - p))]
                        frac = share.get(n, share.get(near, 1.0))
                        sec = (spec["samples_per_cell"] * restarts * np.mean(k) *
                               (c0 + c1 * gates(p, m, n, cd)) * frac)
                        table[(n, p)] += sec
                        total += sec
    slots = args.gpus
    print(f"\nconfig {args.config}: depths {depths}, restarts {restarts}, "
          f"{spec['samples_per_cell']} train/cell")
    print("GPU-hours by n x p (whole-GPU equivalents):")
    for n in spec["n"]:
        print(f"  n={n}: " + "  ".join(f"p{p}={table[(n, p)]/3600:.0f}" for p in depths) +
              ("   (extrapolated)" if n not in coef else ""))
    print(f"TOTAL ~ {total/3600:.0f} GPU-hours -> ~ {total/3600/slots:.1f} h wall on {slots} "
          f"GPUs (+ validation/test evaluation, typically a few % of this)")


if __name__ == "__main__":
    main()
