"""Analytical identity sanity check (NOT new scientific discovery).

If J -> s*J and gamma,alpha -> gamma/s,alpha/s, our fixed order
rotations produce the exact same state. This does NOT demonstrate
universality when weight SHAPE or graph structure changes.
"""
import numpy as np
from cdtransfer.instances import make_instance
from cdtransfer.simulator import Angles,StatevectorSimulator

inst=make_instance(8,"regular3","uniform",77)
a=Angles([0.22,0.35],[0.43,-0.21],[0.17,-0.11])
orig=StatevectorSimulator(inst).state(a)
for s in (0.25,0.5,1.0,2.0,4.0):
    b=Angles(a.gamma/s,a.beta,a.alpha/s)
    new=StatevectorSimulator(inst.scaled(s)).state(b)
    err=float(np.max(np.abs(new-orig)))
    print(f"scale={s:.2g} max_state_error={err:.3e}")
    assert err<1e-12
