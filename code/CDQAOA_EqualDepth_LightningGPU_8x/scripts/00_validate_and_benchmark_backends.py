"""python scripts/00_validate_and_benchmark_backends.py --out reports/BACKENDS.md"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from cdtransfer.instances import make_instance
from cdtransfer.simulator import Angles,StatevectorSimulator


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="reports/BACKENDS.md")
    ap.add_argument("--sizes",nargs="+",type=int,default=[6,8,10,12])
    ap.add_argument("--repetitions",type=int,default=12)
    args=ap.parse_args()
    lines=["# Backend compatibility and throughput (local run)","",
           "Always compare same state/equivalent angles; throughput includes state preparation and expectation.","",
           "| Backend | n | p | repetitions | evaluations/s | max state error vs NumPy |",
           "|---|---:|---:|---:|---:|---:|"]
    backends={"NumPy reference":lambda inst,a:StatevectorSimulator(inst).state(a)}
    try:
        from cdtransfer.qulacs_optional import state as qstate
        import qulacs
        backends["Qulacs CPU"]=qstate
    except ImportError:pass
    try:
        from cdtransfer.qiskit_optional import circuit
        from qiskit.quantum_info import Statevector
        backends["Qiskit Statevector"]=lambda inst,a:Statevector.from_instruction(circuit(inst,a)).data
    except ImportError:pass
    for n in args.sizes:
        inst=make_instance(n,"er3","uniform",1234)
        a=Angles([.21,.34],[-.15,.32],[.13,-.07])
        reference=backends["NumPy reference"](inst,a)
        for label, fn in backends.items():
            t0=time.perf_counter()
            err=0.
            for j in range(args.repetitions):
                psi=fn(inst,a)
                if j==0:err=float(np.max(np.abs(psi-reference)))
            dt=time.perf_counter()-t0
            speed=args.repetitions/max(dt,1e-9)
            lines.append(f"| {label} | {n} | {a.p} | {args.repetitions} | {speed:.2f} | {err:.3g} |")
            print(f"{label}: n={n}, eval/s={speed:.2f}, maxerr={err:.2e}",flush=True)
    lines.extend(["", "This report measures local CPU/optional environments only; GPU measurements deferred.",
                  "NumPy case also includes per-call energy-vector setup. Real offline optimizers cache it."])
    path=Path(args.out);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
