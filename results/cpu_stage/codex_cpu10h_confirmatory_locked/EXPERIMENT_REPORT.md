# Initial experimental report — AUTOMATED PILOT, not a paper result

- Dataset: train=81, validation=54, test=81; independent random seeds (not necessarily non-isomorphic graphs).
- Held-out evaluation complete: True (162/162 pairs).
- Offline optimization records: 1458 (including restarts and QAOA/CD budgets).
- CD depths: [1, 2]; equal-budget ordinary QAOA depth = 3 × CD depth.
- Two-qubit budget uses **fixed unoptimized 2-CX-per-Pauli-rotation decomposition**, NOT hardware-transpiled counts.
- Validation-selected CSV sign: -1; alpha degree exponent: -0.5; alpha multiplier: 1.0; BCH c: 0.5.
- Fixed model is frozen before test: FROZEN_MODEL.json; no test energy used to predict angles.

## Held-out results (normalized signed Ising energy, lower is better)
- Paired gain (QAOA budget fixed − CD proposed): mean **-0.085661**, median **-0.074202**.
- Bootstrap 95% confidence interval: [-0.092803, -0.078920].
- CD win rate: **0.6%** over 162 paired tests.

| Group | Pairs | Mean normalized gain | Win rate | 95% bootstrap CI |
|---|---:|---:|---:|---|
| all | 162 | -0.085661 | 0.6% | [-0.0928, -0.0789] |
| depth-1 | 81 | -0.074476 | 1.2% | [-0.0822, -0.0676] |
| family-regular3 | 54 | -0.089049 | 0.0% | [-0.0989, -0.0796] |
| weights-bimodal | 54 | -0.080757 | 0.0% | [-0.0882, -0.0737] |
| depth-2 | 81 | -0.096847 | 0.0% | [-0.1082, -0.0866] |
| weights-lognormal | 54 | -0.078594 | 0.0% | [-0.0882, -0.0698] |
| weights-heavy_tail | 54 | -0.097633 | 1.9% | [-0.1138, -0.0824] |
| family-er3 | 54 | -0.102466 | 1.9% | [-0.1188, -0.0883] |
| family-er6 | 54 | -0.065470 | 0.0% | [-0.0726, -0.0592] |

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
