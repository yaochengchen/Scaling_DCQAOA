# Equal-depth CD-QAOA GPU study — automated report

- Instances: train=1000, val=500, test=1000 (independent seeds; split fingerprints in manifest.json).
- Depths: [1, 2, 3, 4, 6]. Comparison: CD-QAOA p vs QAOA p (**same number of layers**). CD layers contain 3x the two-qubit rotations of a QAOA layer (cx_* columns); this is NOT an equal-gate-budget comparison.
- Training optimizations: 80000 (all restarts kept; traces in traces/).
- Ratio definition: ['maxcut_approximation_ratio']. E_min/E_max are exact (exhaustive enumeration) for n <= 24; ratios are omitted above that.
- Frozen model sha256: 0e0ccd781ee6ce2fff53fcc4b3268ba116d9ae5abb222117be81fa2ccc03a084; alpha degree power -0.5, multiplier 1.0, csv sign -1.

## Held-out fixed-formula results (no test-instance optimization)

gain = (E_QAOA,best-fixed − E_CD,proposed)/Σ|J| at the same p; positive favours CD.

| p | pairs | mean gain [95% CI] | CD win rate | ratio QAOA fixed | ratio CD proposed | ratio CD V1 | gain vs V1 [95% CI] |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 1000 | +0.1119 [+0.1094, +0.1144] | 99.4% | 0.7681 | 0.8306 | 0.8289 | +0.0029 [+0.0027, +0.0031] |
| 2 | 1000 | +0.0498 [+0.0470, +0.0524] | 92.2% | 0.8341 | 0.8624 | 0.8607 | +0.0023 [+0.0015, +0.0030] |
| 3 | 1000 | +0.0237 [+0.0201, +0.0269] | 85.9% | 0.8707 | 0.8850 | – | – |
| 4 | 1000 | +0.0143 [+0.0103, +0.0181] | 83.9% | 0.8941 | 0.9033 | – | – |
| 6 | 1000 | +0.0113 [+0.0071, +0.0153] | 82.0% | 0.9226 | 0.9300 | – | – |

## Optimized (train, best of restarts) — ceiling for fixed formulas

| p | QAOA ratio | CD ratio | QAOA E/Σ|J| | CD E/Σ|J| |
|---|---:|---:|---:|---:|
| 1 | 0.7723 | 0.8403 | -0.3583 | -0.4798 |
| 2 | 0.8421 | 0.8802 | -0.4830 | -0.5513 |
| 3 | 0.8818 | 0.9122 | -0.5541 | -0.6088 |
| 4 | 0.9080 | 0.9328 | -0.6011 | -0.6459 |
| 6 | 0.9401 | 0.9586 | -0.6588 | -0.6923 |

Per family / weight / n breakdowns: summary.json and summary_table.csv.

## Figures

- `figures/fig01_graph_structures_same_layer.png`
- `figures/fig02_weight_distributions_same_layer.png`
- `figures/fig01b_qubits_same_layer.png`
- `figures/fig03_optimized_same_layer.png`

## Caveats

- Optima are best-of-restarts local optima (L-BFGS-B); see optimizer_audit.json.
- Do not modify the formula after looking at test results; new formulas need a new frozen model validated on val and a fresh test evaluation.
