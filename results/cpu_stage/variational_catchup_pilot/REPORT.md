# Variational catch-up pilot

This exploratory pilot reuses saved `regular3/unit`, n=12 instances and does not rerun the prior production experiment.
It is not the planned 70-instance paper-scale comparison.

- Configuration: p=1, 5 instances, 3 starts per method, at most 100 objective evaluations per trajectory.
- Optimizer: Nelder-Mead; deterministic base seed: 20261012.
- Fixed p=1 CD-QAOA target (selected instances counted once each): 0.875905.
- Simulation and validation time: 2.43 seconds.

| Method | Mean catch-up evaluation | Mean best-so-far at evaluation 100 | Trajectories |
|---|---:|---:|---:|
| Variational QAOA | not reached | 0.773084 | 15 |
| Variational CD-QAOA | not reached | 0.846507 | 15 |

Mean curves equally weight all trajectories; runs ending early carry their final best ratio forward to evaluation 100.
