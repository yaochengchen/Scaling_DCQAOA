# Equal-depth CD-QAOA GPU study — automated report

- Instances: train=8, val=2, test=2 (independent seeds; split fingerprints in manifest.json).
- Depths: [1, 2, 3]. Comparison: CD-QAOA p vs QAOA p (**same number of layers**). CD layers contain 3x the two-qubit rotations of a QAOA layer (cx_* columns); this is NOT an equal-gate-budget comparison.
- Training optimizations: 96 (all restarts kept; traces in traces/).
- Ratio definition: ['maxcut_approximation_ratio']. E_min/E_max are exact (exhaustive enumeration) for n <= 24; ratios are omitted above that.
- Frozen model sha256: d656f5f2c1604a4614eefdc36c42e3236cb23c69b6043a8873a855d5d5ffef0d; alpha degree power -0.5, multiplier 0.5, csv sign -1.

## Held-out fixed-formula results (no test-instance optimization)

gain = (E_QAOA,best-fixed − E_CD,proposed)/Σ|J| at the same p; positive favours CD.

| p | pairs | mean gain [95% CI] | CD win rate | ratio QAOA fixed | ratio CD proposed | ratio CD V1 | gain vs V1 [95% CI] |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 2 | +0.1147 [+0.1034, +0.1260] | 100.0% | 0.7747 | 0.8440 | 0.8434 | -0.0001 [-0.0162, +0.0160] |
| 2 | 2 | +0.0519 [+0.0494, +0.0544] | 100.0% | 0.8464 | 0.8779 | 0.8819 | -0.0068 [-0.0106, -0.0030] |
| 3 | 2 | +0.0207 [+0.0074, +0.0340] | 100.0% | 0.8852 | 0.8972 | – | – |

## Optimized (train, best of restarts) — ceiling for fixed formulas

| p | QAOA ratio | CD ratio | QAOA E/Σ|J| | CD E/Σ|J| |
|---|---:|---:|---:|---:|
| 1 | 0.7973 | 0.8592 | -0.2530 | -0.3540 |
| 2 | 0.8618 | 0.8964 | -0.3566 | -0.4128 |
| 3 | 0.8977 | 0.9256 | -0.4149 | -0.4603 |

Per family / weight / n breakdowns: summary.json and summary_table.csv.

## Figures

- `figures/fig01_graph_structures_same_layer.png`
- `figures/fig02_weight_distributions_same_layer.png`
- `figures/fig01b_qubits_same_layer.png`
- `figures/fig03_optimized_same_layer.png`

## Caveats

- Optima are best-of-restarts local optima (L-BFGS-B); see optimizer_audit.json.
- Do not modify the formula after looking at test results; new formulas need a new frozen model validated on val and a fresh test evaluation.
