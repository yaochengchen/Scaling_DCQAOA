# Initial experimental report — AUTOMATED PILOT, not a paper result

- Dataset: train=81, validation=54, test=12; independent random seeds (not necessarily non-isomorphic graphs).
- Held-out evaluation complete: True (24/24 pairs).
- Offline optimization records: 1458 (including restarts and QAOA/CD budgets).
- CD depths: [1, 2]; equal-budget ordinary QAOA depth = 3 × CD depth.
- Two-qubit budget uses **fixed unoptimized 2-CX-per-Pauli-rotation decomposition**, NOT hardware-transpiled counts.
- Validation-selected CSV sign: -1; alpha degree exponent: -0.5; alpha multiplier: 1.0; BCH c: 0.5.
- Fixed model is frozen before test: FROZEN_MODEL.json; no test energy used to predict angles.

## Held-out results (normalized signed Ising energy, lower is better)
- Paired gain (QAOA budget fixed − CD proposed): mean **-0.093541**, median **-0.072035**.
- Bootstrap 95% confidence interval: [-0.117415, -0.074551].
- CD win rate: **0.0%** over 24 paired tests.

| Group | Pairs | Mean normalized gain | Win rate | 95% bootstrap CI |
|---|---:|---:|---:|---|
| all | 24 | -0.093541 | 0.0% | [-0.1174, -0.0746] |
| depth-1 | 12 | -0.072654 | 0.0% | [-0.0898, -0.0611] |
| family-regular3 | 12 | -0.093463 | 0.0% | [-0.1264, -0.0671] |
| weights-bimodal | 8 | -0.071245 | 0.0% | [-0.0861, -0.0589] |
| depth-2 | 12 | -0.114428 | 0.0% | [-0.1541, -0.0796] |
| weights-lognormal | 8 | -0.075164 | 0.0% | [-0.0910, -0.0618] |
| weights-heavy_tail | 8 | -0.134215 | 0.0% | [-0.1844, -0.0836] |
| family-er6 | 12 | -0.093619 | 0.0% | [-0.1266, -0.0668] |

## Interpretation and limitations
- Positive gain indicates proposed fixed CD beats the validation-selected fixed QAOA baseline under a matched unoptimized gate decomposition.
- A negative gain does **not** support the main performance claim; report failures rather than changing test data.
- All tests may be too small for statistical reliability; confirm on larger independent seeds and actual native-basis transpilation.
- The offline CD optimization may be trapped in local minima; multi-start and optimization-budget convergence studies remain necessary.
- This pilot provides a complete pipeline, not validated new physics or publishable evidence by itself.
- GPU scale-up intentionally deferred to collaborator.

## Figures

- `figures/fig02_parameter_concentration.png`
- `figures/fig05_equal_budget_paired.png`
- `figures/fig04_generalization.png`

## Next concrete actions

1. Validate simulator Pauli-rotation matrix conventions and compare against Qulacs/Aer when available.
2. Expand independent n=8–16 discovery training with multi-start and collect normalized angle distributions.
3. Lock validation-fit model and test across independent n=12–20 / OOD distributions with uncertainty intervals.
4. Verify actual CX/CZ compiled budgets; add matching-depth QAOA and noise robustness.
5. Draft manuscript based on observed positive **or** negative evidence; larger GPU jobs are optional.
