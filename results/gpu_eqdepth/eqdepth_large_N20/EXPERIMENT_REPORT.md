# Equal-depth CD-QAOA GPU study — automated report

- Instances: train=3000, val=1500, test=3000 (independent seeds; split fingerprints in manifest.json).
- Depths: [1, 2, 3, 4]. Comparison: CD-QAOA p vs QAOA p (**same number of layers**). CD layers contain 3x the two-qubit rotations of a QAOA layer (cx_* columns); this is NOT an equal-gate-budget comparison.
- Training optimizations: 192000 (all restarts kept; traces in traces/).
- Ratio definition: ['maxcut_approximation_ratio']. E_min/E_max are exact (exhaustive enumeration) for n <= 24; ratios are omitted above that.
- Frozen model sha256: 8542de5e4b01d8696e4bc3d7e1641500264f34e807eddc6ec74af7e3d1f82679; alpha degree power -0.5, multiplier 1.0, csv sign -1.

## Held-out fixed-formula results (no test-instance optimization)

gain = (E_QAOA,best-fixed − E_CD,proposed)/Σ|J| at the same p; positive favours CD.

| p | pairs | mean gain [95% CI] | CD win rate | ratio QAOA fixed | ratio CD proposed | ratio CD V1 | gain vs V1 [95% CI] |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 3000 | +0.1108 [+0.1092, +0.1123] | 99.3% | 0.7668 | 0.8287 | 0.8267 | +0.0034 [+0.0032, +0.0035] |
| 2 | 3000 | +0.0497 [+0.0482, +0.0511] | 94.1% | 0.8319 | 0.8599 | 0.8578 | +0.0036 [+0.0035, +0.0037] |
| 3 | 3000 | +0.0314 [+0.0297, +0.0331] | 89.3% | 0.8673 | 0.8857 | – | – |
| 4 | 3000 | +0.0218 [+0.0199, +0.0236] | 86.8% | 0.8910 | 0.9042 | – | – |

## Optimized (train, best of restarts) — ceiling for fixed formulas

| p | QAOA ratio | CD ratio | QAOA E/Σ|J| | CD E/Σ|J| |
|---|---:|---:|---:|---:|
| 1 | 0.7705 | 0.8381 | -0.3536 | -0.4743 |
| 2 | 0.8394 | 0.8779 | -0.4765 | -0.5455 |
| 3 | 0.8789 | 0.9098 | -0.5473 | -0.6030 |
| 4 | 0.9054 | 0.9308 | -0.5948 | -0.6407 |

Per family / weight / n breakdowns: summary.json and summary_table.csv.

## Figures

- `figures/fig01_graph_structures_same_layer.png`
- `figures/fig02_weight_distributions_same_layer.png`
- `figures/fig01b_qubits_same_layer.png`
- `figures/fig03_optimized_same_layer.png`

## Caveats

- Optima are best-of-restarts local optima (L-BFGS-B); see optimizer_audit.json.
- Do not modify the formula after looking at test results; new formulas need a new frozen model validated on val and a fresh test evaluation.
