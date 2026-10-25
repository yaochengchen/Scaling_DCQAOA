# Unified-table held-out evaluation

Model SHA-256: `f2be3446fd992a826dcfec8f6048b27a8ccf21fff146d06faa84fd507d7f1eb4`. No angle optimization was run.
Unique graph-depth pairs: 32,000; unique test graphs: 7,000.

## Overall p=1 across 7,000 graphs

Mean approximation ratio: QAOA 0.771860; unified DC-QAOA 0.835305.
Mean normalized energy gain: 0.113348 [0.112341, 0.114395]; DC win fraction 98.8714%.

## By cohort and depth

| Cohort | p | Pairs | QAOA ratio | Unified DC ratio | Gain / ΣJ [95% CI] | DC wins |
|---|---:|---:|---:|---:|---:|---:|
| large | 1 | 3,000 | 0.766839 | 0.829032 | 0.111053 [0.109563, 0.112559] | 99.17% |
| large | 2 | 3,000 | 0.831857 | 0.859979 | 0.049276 [0.047663, 0.050700] | 92.93% |
| large | 3 | 3,000 | 0.867270 | 0.885232 | 0.030423 [0.028708, 0.032090] | 88.90% |
| large | 4 | 3,000 | 0.890963 | 0.901964 | 0.017385 [0.015192, 0.019413] | 84.03% |
| small | 1 | 4,000 | 0.775626 | 0.840010 | 0.115069 [0.113645, 0.116459] | 98.65% |
| small | 2 | 4,000 | 0.843632 | 0.872309 | 0.050423 [0.048859, 0.051933] | 92.38% |
| small | 3 | 4,000 | 0.879596 | 0.899372 | 0.034045 [0.032489, 0.035629] | 90.97% |
| small | 4 | 4,000 | 0.903467 | 0.916469 | 0.022002 [0.020486, 0.023565] | 88.92% |
| small | 6 | 4,000 | 0.930284 | 0.941731 | 0.018665 [0.016427, 0.020797] | 85.88% |

The test graphs and old QAOA comparator/extrema came from the two archived GPU cohorts. The unified table was constructed after the original cohort analyses, using training records only, and locked before this evaluation. No claim of a previously untouched research-wide test split is made.
