# MaxCut Approximation Experiment Report

## Configuration

- Version: `maxcut-approx-current-v1`
- Samples per cell: 10
- Qubits: [8, 10, 12, 14, 16]
- Layers: [1, 2]
- Config SHA-256: `f1313c6dcdfcfbef274fe55e0b6c1d1aae3253521c08339a1ba21fb4a69046b6`

## Solver agreement

All 350 saved records report optimal SCIP and Gurobi statuses with objectives agreeing to 1e-8.

## Results

| Family | Weight | n | p | Method | Count | Mean | 95% bootstrap CI |
|---|---:|---:|---:|---|---:|---:|---:|
| tree | unit | 8 | 1 | qaoa_csv | 10 | 0.753123 | [0.750049, 0.756197] |
| tree | unit | 8 | 1 | cd_proposed | 10 | 0.831484 | [0.825171, 0.837632] |
| tree | unit | 10 | 1 | qaoa_csv | 10 | 0.742752 | [0.737870, 0.747225] |
| tree | unit | 10 | 1 | cd_proposed | 10 | 0.816639 | [0.803297, 0.827052] |
| tree | unit | 12 | 1 | qaoa_csv | 10 | 0.741143 | [0.734548, 0.746269] |
| tree | unit | 12 | 1 | cd_proposed | 10 | 0.814514 | [0.799381, 0.827227] |
| tree | unit | 12 | 2 | qaoa_csv | 10 | 0.822618 | [0.817658, 0.827096] |
| tree | unit | 12 | 2 | cd_proposed | 10 | 0.859380 | [0.850962, 0.866752] |
| tree | unit | 14 | 1 | qaoa_csv | 10 | 0.739000 | [0.734692, 0.743307] |
| tree | unit | 14 | 1 | cd_proposed | 10 | 0.813645 | [0.802523, 0.824445] |
| tree | unit | 16 | 1 | qaoa_csv | 10 | 0.736803 | [0.733391, 0.739829] |
| tree | unit | 16 | 1 | cd_proposed | 10 | 0.809320 | [0.801178, 0.816044] |
| regular3 | unit | 8 | 1 | qaoa_csv | 10 | 0.806713 | [0.792607, 0.824927] |
| regular3 | unit | 8 | 1 | cd_proposed | 10 | 0.885458 | [0.870067, 0.899193] |
| regular3 | unit | 10 | 1 | qaoa_csv | 10 | 0.777113 | [0.765647, 0.791071] |
| regular3 | unit | 10 | 1 | cd_proposed | 10 | 0.860321 | [0.848256, 0.872188] |
| regular3 | unit | 12 | 1 | qaoa_csv | 10 | 0.787419 | [0.769627, 0.807291] |
| regular3 | unit | 12 | 1 | cd_proposed | 10 | 0.869602 | [0.853809, 0.886664] |
| regular3 | unit | 12 | 2 | qaoa_csv | 10 | 0.862590 | [0.847633, 0.879388] |
| regular3 | unit | 12 | 2 | cd_proposed | 10 | 0.893043 | [0.879837, 0.907150] |
| regular3 | unit | 14 | 1 | qaoa_csv | 10 | 0.776513 | [0.763451, 0.790977] |
| regular3 | unit | 14 | 1 | cd_proposed | 10 | 0.856213 | [0.844459, 0.869650] |
| regular3 | unit | 16 | 1 | qaoa_csv | 10 | 0.771809 | [0.757414, 0.787275] |
| regular3 | unit | 16 | 1 | cd_proposed | 10 | 0.851817 | [0.840760, 0.863679] |
| er3 | unit | 8 | 1 | qaoa_csv | 10 | 0.819865 | [0.807349, 0.833806] |
| er3 | unit | 8 | 1 | cd_proposed | 10 | 0.878350 | [0.864613, 0.890796] |
| er3 | unit | 10 | 1 | qaoa_csv | 10 | 0.794643 | [0.777496, 0.812962] |
| er3 | unit | 10 | 1 | cd_proposed | 10 | 0.865036 | [0.847132, 0.884969] |
| er3 | unit | 12 | 1 | qaoa_csv | 10 | 0.772389 | [0.756845, 0.788046] |
| er3 | unit | 12 | 1 | cd_proposed | 10 | 0.844357 | [0.830138, 0.857186] |
| er3 | unit | 12 | 2 | qaoa_csv | 10 | 0.851024 | [0.835559, 0.865657] |
| er3 | unit | 12 | 2 | cd_proposed | 10 | 0.888564 | [0.876793, 0.899832] |
| er3 | unit | 14 | 1 | qaoa_csv | 10 | 0.777807 | [0.765221, 0.789159] |
| er3 | unit | 14 | 1 | cd_proposed | 10 | 0.846151 | [0.832789, 0.857834] |
| er3 | unit | 16 | 1 | qaoa_csv | 10 | 0.753444 | [0.740538, 0.766610] |
| er3 | unit | 16 | 1 | cd_proposed | 10 | 0.822496 | [0.805629, 0.838259] |
| er6 | unit | 8 | 1 | qaoa_csv | 10 | 0.821704 | [0.809729, 0.833447] |
| er6 | unit | 8 | 1 | cd_proposed | 10 | 0.837429 | [0.830116, 0.844696] |
| er6 | unit | 10 | 1 | qaoa_csv | 10 | 0.806243 | [0.788759, 0.822037] |
| er6 | unit | 10 | 1 | cd_proposed | 10 | 0.841766 | [0.830074, 0.853393] |
| er6 | unit | 12 | 1 | qaoa_csv | 10 | 0.805193 | [0.792640, 0.818715] |
| er6 | unit | 12 | 1 | cd_proposed | 10 | 0.846555 | [0.833973, 0.860254] |
| er6 | unit | 12 | 2 | qaoa_csv | 10 | 0.872715 | [0.860479, 0.886242] |
| er6 | unit | 12 | 2 | cd_proposed | 10 | 0.885855 | [0.875387, 0.896518] |
| er6 | unit | 14 | 1 | qaoa_csv | 10 | 0.809864 | [0.795809, 0.821866] |
| er6 | unit | 14 | 1 | cd_proposed | 10 | 0.851249 | [0.841096, 0.859218] |
| er6 | unit | 16 | 1 | qaoa_csv | 10 | 0.805737 | [0.796579, 0.815154] |
| er6 | unit | 16 | 1 | cd_proposed | 10 | 0.847462 | [0.839541, 0.856323] |
| regular3 | uniform | 8 | 1 | qaoa_csv | 10 | 0.792039 | [0.776891, 0.806864] |
| regular3 | uniform | 8 | 1 | cd_proposed | 10 | 0.870321 | [0.854123, 0.888128] |
| regular3 | uniform | 10 | 1 | qaoa_csv | 10 | 0.776484 | [0.762733, 0.791393] |
| regular3 | uniform | 10 | 1 | cd_proposed | 10 | 0.856276 | [0.842862, 0.869450] |
| regular3 | uniform | 12 | 1 | qaoa_csv | 10 | 0.768351 | [0.748497, 0.785267] |
| regular3 | uniform | 12 | 1 | cd_proposed | 10 | 0.846358 | [0.833985, 0.858308] |
| regular3 | uniform | 12 | 2 | qaoa_csv | 10 | 0.840532 | [0.822705, 0.856536] |
| regular3 | uniform | 12 | 2 | cd_proposed | 10 | 0.872157 | [0.858223, 0.884136] |
| regular3 | uniform | 14 | 1 | qaoa_csv | 10 | 0.771215 | [0.757930, 0.785998] |
| regular3 | uniform | 14 | 1 | cd_proposed | 10 | 0.856315 | [0.844373, 0.866954] |
| regular3 | uniform | 16 | 1 | qaoa_csv | 10 | 0.771038 | [0.759083, 0.784156] |
| regular3 | uniform | 16 | 1 | cd_proposed | 10 | 0.849083 | [0.839988, 0.858796] |
| regular3 | bimodal | 8 | 1 | qaoa_csv | 10 | 0.791220 | [0.777465, 0.804550] |
| regular3 | bimodal | 8 | 1 | cd_proposed | 10 | 0.867392 | [0.855642, 0.878047] |
| regular3 | bimodal | 10 | 1 | qaoa_csv | 10 | 0.785932 | [0.774378, 0.797885] |
| regular3 | bimodal | 10 | 1 | cd_proposed | 10 | 0.866538 | [0.856947, 0.876660] |
| regular3 | bimodal | 12 | 1 | qaoa_csv | 10 | 0.780941 | [0.761570, 0.801998] |
| regular3 | bimodal | 12 | 1 | cd_proposed | 10 | 0.856835 | [0.842021, 0.873292] |
| regular3 | bimodal | 12 | 2 | qaoa_csv | 10 | 0.851722 | [0.835408, 0.870738] |
| regular3 | bimodal | 12 | 2 | cd_proposed | 10 | 0.883176 | [0.871396, 0.896538] |
| regular3 | bimodal | 14 | 1 | qaoa_csv | 10 | 0.776323 | [0.766880, 0.786191] |
| regular3 | bimodal | 14 | 1 | cd_proposed | 10 | 0.857590 | [0.846319, 0.867951] |
| regular3 | bimodal | 16 | 1 | qaoa_csv | 10 | 0.777157 | [0.767428, 0.786870] |
| regular3 | bimodal | 16 | 1 | cd_proposed | 10 | 0.859745 | [0.848181, 0.870269] |
| regular3 | lognormal | 8 | 1 | qaoa_csv | 10 | 0.809361 | [0.792999, 0.828149] |
| regular3 | lognormal | 8 | 1 | cd_proposed | 10 | 0.877335 | [0.863400, 0.892738] |
| regular3 | lognormal | 10 | 1 | qaoa_csv | 10 | 0.779445 | [0.760398, 0.796508] |
| regular3 | lognormal | 10 | 1 | cd_proposed | 10 | 0.848888 | [0.821156, 0.870008] |
| regular3 | lognormal | 12 | 1 | qaoa_csv | 10 | 0.768159 | [0.747074, 0.788702] |
| regular3 | lognormal | 12 | 1 | cd_proposed | 10 | 0.842830 | [0.824835, 0.857077] |
| regular3 | lognormal | 12 | 2 | qaoa_csv | 10 | 0.840356 | [0.822721, 0.854260] |
| regular3 | lognormal | 12 | 2 | cd_proposed | 10 | 0.876906 | [0.862729, 0.889053] |
| regular3 | lognormal | 14 | 1 | qaoa_csv | 10 | 0.765985 | [0.748976, 0.783509] |
| regular3 | lognormal | 14 | 1 | cd_proposed | 10 | 0.839756 | [0.825597, 0.855595] |
| regular3 | lognormal | 16 | 1 | qaoa_csv | 10 | 0.769061 | [0.757925, 0.780004] |
| regular3 | lognormal | 16 | 1 | cd_proposed | 10 | 0.837064 | [0.825249, 0.849551] |

## Figures

| Figure | Saved artifact |
|---|---|
| Graph structures | `figures/fig01_graph_structures_same_layer.png` |
| Weight distributions | `figures/fig02_weight_distributions_same_layer.png` |

## Interpretation limitation

This is a **same-layer** comparison. It is not a matched or equal two-qubit-gate-budget comparison; CD-QAOA and QAOA at the same p can have different compiled costs.
