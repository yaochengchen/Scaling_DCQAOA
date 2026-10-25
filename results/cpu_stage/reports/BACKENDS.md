# Backend compatibility and throughput (local run)

Always compare same state/equivalent angles; throughput includes state preparation and expectation.

| Backend | n | p | repetitions | evaluations/s | max state error vs NumPy |
|---|---:|---:|---:|---:|---:|
| NumPy reference | 6 | 2 | 5 | 2729.41 | 0 |
| Qulacs CPU | 6 | 2 | 5 | 2734.93 | 1.55e-16 |
| Qiskit Statevector | 6 | 2 | 5 | 227.18 | 1.04e-15 |
| NumPy reference | 8 | 2 | 5 | 1249.41 | 0 |
| Qulacs CPU | 8 | 2 | 5 | 2349.85 | 1.21e-16 |
| Qiskit Statevector | 8 | 2 | 5 | 128.91 | 1.13e-15 |
| NumPy reference | 10 | 2 | 5 | 749.82 | 0 |
| Qulacs CPU | 10 | 2 | 5 | 1288.76 | 1.45e-16 |
| Qiskit Statevector | 10 | 2 | 5 | 89.03 | 8.89e-16 |

This report measures local CPU/optional environments only; GPU measurements deferred.
NumPy case also includes per-call energy-vector setup. Real offline optimizers cache it.
