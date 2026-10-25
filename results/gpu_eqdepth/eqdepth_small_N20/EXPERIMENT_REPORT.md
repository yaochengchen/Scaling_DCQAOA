# Equal-depth CD-QAOA GPU study — automated report

- Instances: train=4000, val=2000, test=4000 (independent seeds; split fingerprints in manifest.json).
- Depths: [1, 2, 3, 4, 6]. Comparison: CD-QAOA p vs QAOA p (**same number of layers**). CD layers contain 3x the two-qubit rotations of a QAOA layer (cx_* columns); this is NOT an equal-gate-budget comparison.
- Training optimizations: 320000 (all restarts kept; traces in traces/).
- Ratio definition: ['maxcut_approximation_ratio']. E_min/E_max are exact (exhaustive enumeration) for n <= 24; ratios are omitted above that.
- Frozen model sha256: 880d7fedbea6b7ade14016647786fa4d5b49f419693215410a7831ac0a38713d; alpha degree power -0.5, multiplier 1.0, csv sign -1.

## Held-out fixed-formula results (no test-instance optimization)

gain = (E_QAOA,best-fixed − E_CD,proposed)/Σ|J| at the same p; positive favours CD.

| p | pairs | mean gain [95% CI] | CD win rate | ratio QAOA fixed | ratio CD proposed | ratio CD V1 | gain vs V1 [95% CI] |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 4000 | +0.1143 [+0.1127, +0.1157] | 98.2% | 0.7756 | 0.8397 | 0.8383 | +0.0018 [+0.0015, +0.0020] |
| 2 | 4000 | +0.0482 [+0.0466, +0.0498] | 91.7% | 0.8436 | 0.8712 | 0.8705 | +0.0002 [-0.0004, +0.0008] |
| 3 | 4000 | +0.0327 [+0.0309, +0.0344] | 90.6% | 0.8796 | 0.8987 | – | – |
| 4 | 4000 | +0.0228 [+0.0211, +0.0245] | 89.5% | 0.9035 | 0.9169 | – | – |
| 6 | 4000 | +0.0172 [+0.0149, +0.0194] | 81.3% | 0.9303 | 0.9411 | – | – |

## Optimized (train, best of restarts) — ceiling for fixed formulas

| p | QAOA ratio | CD ratio | QAOA E/Σ|J| | CD E/Σ|J| |
|---|---:|---:|---:|---:|
| 1 | 0.7799 | 0.8494 | -0.3758 | -0.5002 |
| 2 | 0.8527 | 0.8918 | -0.5060 | -0.5761 |
| 3 | 0.8935 | 0.9246 | -0.5792 | -0.6351 |
| 4 | 0.9199 | 0.9453 | -0.6266 | -0.6723 |
| 6 | 0.9521 | 0.9691 | -0.6843 | -0.7152 |

Per family / weight / n breakdowns: summary.json and summary_table.csv.

## Figures

- `figures/fig01_graph_structures_same_layer.png`
- `figures/fig02_weight_distributions_same_layer.png`
- `figures/fig01b_qubits_same_layer.png`
- `figures/fig03_optimized_same_layer.png`

## Caveats

- Optima are best-of-restarts local optima (L-BFGS-B); see optimizer_audit.json.
- Do not modify the formula after looking at test results; new formulas need a new frozen model validated on val and a fresh test evaluation.
