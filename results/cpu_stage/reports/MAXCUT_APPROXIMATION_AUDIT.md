# MaxCut Approximation Production Audit

## Independent checks

- 350/350 unique record IDs and 350/350 unique graph fingerprints.
- 7 cells × 5 qubit counts × 10 samples = 350 exact slots.
- 840 method-depth evaluations: 700 at p=1 and 140 at p=2 only for n=12.
- Methods are exactly `qaoa_csv` and `cd_proposed`; the frozen model certifies no target-instance objective calls.
- All solver statuses are optimal; paired objectives and the saved optimum agree within 1e-8.
- All weights and numeric results are finite, all weights are positive, and all ratios are in [0,1].
- Every expected-cut/Ising-energy and ratio/cut/optimum identity holds within 1e-8.
- Config, frozen-model, and source-CSV SHA-256 values match the manifest and current assets.
- Summary is complete with 84 exact cells; two PNGs and the experiment report exist.
- Plotter AST has no imports of solver, simulator, runner, or `make_instance`.

## Numeric extrema and residuals

- Ratio minimum / maximum: `0.69537896657438603` / `0.93531334059602966`
- Minimum positive edge weight: `0.20000000000000001`
- Maximum |SCIP − Gurobi objective|: `2.1316282072803006e-14`
- Maximum |expected cut − energy-derived cut|: `3.5527136788005009e-15`
- Maximum |ratio − expected cut / optimum|: `0`
- Tolerance for all identity/agreement checks: `1e-8` (absolute)

## Input hashes

- Config: `f1313c6dcdfcfbef274fe55e0b6c1d1aae3253521c08339a1ba21fb4a69046b6`
- Frozen model: `cf9a38dd21d7c159cc535b51677082cb178ed7da4844ca4fbffd2d6fed25d00c`
- Source CSV: `9e3e9c3dcc0b4ee9ce5a39319ae567fa1161ed79a0606355c5b9b6e3f98f15a6`

## Environment

- gurobi_engine_version: `13.0.3`
- gurobipy_package_version: `13.0.3`
- networkx: `3.5`
- numpy: `2.3.5`
- platform: `Windows-11-10.0.26200-SP0`
- pyscipopt_package_version: `6.2.0`
- python: `3.13.9`
- scip_engine_version: `10.0.2`

## Artifact hashes

- `fig01_graph_structures_same_layer.png`: `ba5c6bdb409dfbc96687803d4b93a7315c01a578e2bf1cf4608370f23fce2d83`
- `fig02_weight_distributions_same_layer.png`: `4b7938db43daf4a68e9b0560f8e608edc2561509e19fa950a4c8f033cb056806`
- `EXPERIMENT_REPORT.md`: `c68c8a4a5ede8945bfad8b2f0cd70c311b7b98586de477383736bc15833e49fe`

## Final status

FINAL: PASS
